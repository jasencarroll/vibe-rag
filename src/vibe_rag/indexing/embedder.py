"""OpenRouter-only embedding provider for vibe-rag.

This module implements all embedding functionality through a single backend:
the OpenRouter embeddings API (``https://openrouter.ai/api/v1/embeddings``).
It exposes:

* :class:`EmbeddingProvider` -- a :class:`typing.Protocol` that any embedding
  backend must satisfy (embed text, embed code, embed code queries, close).
* :class:`OpenRouterEmbeddingProvider` -- the concrete implementation that
  calls the OpenRouter API with automatic batching, an LRU embedding cache,
  and progress reporting.
* :func:`create_embedding_provider` -- factory that reads env vars and returns
  a configured provider instance.
* :func:`embedding_provider_status` -- lightweight health-check dict consumed
  by ``vibe-rag doctor``.
* :func:`resolve_embedding_model`, :func:`resolve_embedding_dimensions`,
  :func:`resolve_embedding_profile` -- helpers that resolve the active model
  and dimension settings from env vars (with defaults).

Environment variables
---------------------
``RAG_OR_API_KEY``
    **Required.**  OpenRouter API key used for authentication.
``RAG_OR_EMBED_MOD``
    Override the embedding model (default: ``perplexity/pplx-embed-v1-4b``).
``RAG_OR_EMBED_DIM``
    Override the embedding dimensions (default: ``2560``).  Must be a positive
    integer when set.

If none of the above are present in the process environment, the module
performs a one-shot login-shell fallback (see :func:`_load_embedding_env_from_shell`)
to inherit them from the user's shell profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
"""Base URL for the OpenRouter embeddings endpoint."""

DEFAULT_OPENROUTER_MODEL = "perplexity/pplx-embed-v1-4b"
"""Default embedding model when ``RAG_OR_EMBED_MOD`` is not set."""

DEFAULT_OPENROUTER_DIMENSIONS = 2560
"""Default embedding vector dimensions when ``RAG_OR_EMBED_DIM`` is not set."""

DEFAULT_PROVIDER = "openrouter"
"""Provider identifier string surfaced in status/profile dicts."""

_TRUSTED_EMBEDDING_SHELLS = frozenset(
    {
        "/bin/sh",
        "/usr/bin/sh",
        "/bin/bash",
        "/usr/bin/bash",
        "/bin/zsh",
        "/usr/bin/zsh",
    }
)
"""Shells considered safe for the login-shell env fallback."""

BATCH_SIZE = 64
"""Maximum number of texts sent in a single OpenRouter API request."""

MAX_CHARS = 16_000
"""Per-text character limit; longer texts are truncated before embedding."""

ProgressCallback = Callable[[dict[str, object]], None]
"""Callback type for reporting embedding batch progress to callers."""

EMBEDDING_ENV_KEYS = (
    "RAG_OR_API_KEY",
    "RAG_OR_EMBED_MOD",
    "RAG_OR_EMBED_DIM",
)
"""Environment variable names inspected for embedding configuration."""

_SHELL_ENV_ATTEMPTED = False
"""Module-level flag ensuring the shell env fallback runs at most once."""


def _emit_progress(progress_callback: ProgressCallback | None, **event: object) -> None:
    """Invoke *progress_callback* with the keyword args as an event dict, if set."""
    if progress_callback is None:
        return
    progress_callback(event)


def _batch_by_limits(
    texts: list[str],
    *,
    max_items: int,
) -> list[list[str]]:
    """Split *texts* into batches of at most *max_items* each.

    Returns an empty list when *texts* is empty.
    """
    if not texts:
        return []

    batches: list[list[str]] = []
    current_batch: list[str] = []

    for text in texts:
        if current_batch and len(current_batch) >= max_items:
            batches.append(current_batch)
            current_batch = []

        current_batch.append(text)

    if current_batch:
        batches.append(current_batch)

    return batches


def _preferred_shell() -> tuple[str, str]:
    """Return ``(shell_path, flag)`` for the login-shell env probe.

    Falls back to ``/bin/sh -lc`` when the user's ``$SHELL`` is not in
    :data:`_TRUSTED_EMBEDDING_SHELLS` or is not executable.
    """
    shell = os.environ.get("SHELL", "").strip() or "/bin/sh"
    shell_path = os.path.realpath(shell)
    if os.path.isabs(shell) and os.access(shell_path, os.X_OK) and shell_path in _TRUSTED_EMBEDDING_SHELLS:
        return shell_path, "-lc"
    return "/bin/sh", "-lc"


def _shell_env_fallback_needed() -> bool:
    """Return ``True`` when none of the :data:`EMBEDDING_ENV_KEYS` are set."""
    return not any(os.environ.get(key, "").strip() for key in EMBEDDING_ENV_KEYS)


def _load_embedding_env_from_shell() -> None:
    """One-shot login-shell fallback for embedding env vars.

    When the current process environment lacks all :data:`EMBEDDING_ENV_KEYS`,
    this function spawns a login shell (``$SHELL -lc 'command env -0'``) to
    capture the user's full environment and copies any relevant keys into
    ``os.environ``.  The probe runs at most once per process thanks to the
    module-level :data:`_SHELL_ENV_ATTEMPTED` guard.
    """
    global _SHELL_ENV_ATTEMPTED
    if _SHELL_ENV_ATTEMPTED or not _shell_env_fallback_needed():
        return
    _SHELL_ENV_ATTEMPTED = True
    shell, flag = _preferred_shell()
    try:
        result = subprocess.run(
            [shell, flag, "command env -0"],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return
    for entry in result.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key_bytes, value_bytes = entry.split(b"=", 1)
        key = key_bytes.decode("utf-8", errors="ignore")
        if key not in EMBEDDING_ENV_KEYS or os.environ.get(key, "").strip():
            continue
        value = value_bytes.decode("utf-8", errors="ignore").strip()
        if value:
            os.environ[key] = value


def resolve_embedding_model() -> str:
    """Return the active embedding model name.

    Reads ``RAG_OR_EMBED_MOD`` (after the shell-env fallback), falling back to
    :data:`DEFAULT_OPENROUTER_MODEL` when the variable is unset or blank.
    """
    _load_embedding_env_from_shell()
    return os.environ.get("RAG_OR_EMBED_MOD", "").strip() or DEFAULT_OPENROUTER_MODEL


def resolve_embedding_dimensions() -> int:
    """Return the active embedding dimensions as a positive integer.

    Reads ``RAG_OR_EMBED_DIM`` (after the shell-env fallback), falling back to
    :data:`DEFAULT_OPENROUTER_DIMENSIONS` when the variable is unset or blank.

    Raises:
        RuntimeError: If the value is not a valid positive integer.
    """
    _load_embedding_env_from_shell()
    raw = os.environ.get("RAG_OR_EMBED_DIM", "").strip()
    if not raw:
        return DEFAULT_OPENROUTER_DIMENSIONS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("RAG_OR_EMBED_DIM must be an integer") from exc
    if value <= 0:
        raise RuntimeError("RAG_OR_EMBED_DIM must be positive")
    return value


def resolve_embedding_profile() -> dict[str, object]:
    """Return a dict describing the active embedding configuration.

    The returned dict contains keys ``provider``, ``model``, and
    ``dimensions`` -- used by the ``index_project`` tool to stamp profile
    metadata into the project database.
    """
    return {
        "provider": DEFAULT_PROVIDER,
        "model": resolve_embedding_model(),
        "dimensions": resolve_embedding_dimensions(),
    }


def _response_error_message(resp: httpx.Response, *json_paths: tuple[str, ...]) -> str:
    """Extract a human-readable error message from an HTTP error response.

    Tries each *json_paths* (sequences of dict keys) against the JSON body.
    Falls back to the first 200 characters of the raw body text, or
    ``"unknown error"`` as a last resort.
    """
    try:
        payload = resp.json()
    except (json.JSONDecodeError, ValueError):
        text = (resp.text or "").strip()
        return text[:200] if text else "unknown error"

    for path in json_paths:
        current: object = payload
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, str) and current.strip():
            return current
    return "unknown error"


class EmbeddingProvider(Protocol):
    """Structural protocol that any embedding backend must satisfy.

    All ``embed_*`` methods accept a list of plain-text strings and return
    a list of equal length, where each element is a ``list[float]`` embedding
    vector.  An optional *progress_callback* receives batch-progress dicts.

    Methods:
        embed_text_sync: Embed natural-language texts (docs, memories).
        embed_code_sync: Embed source-code chunks for indexing.
        embed_code_query_sync: Embed a natural-language query intended to
            search code.  (For OpenRouter's single-model design this
            produces the same vectors as ``embed_code_sync``.)
        close: Release underlying HTTP/transport resources.
    """

    def embed_text_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def embed_code_query_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def close(self) -> None: ...


_EMBED_CACHE_MAX = 256
"""Maximum number of entries in the per-instance embedding LRU cache."""


class _CachedEmbeddingMixin:
    """Per-instance LRU cache for single-text embedding calls.

    Concrete providers mix this in to avoid redundant API round-trips when the
    same text is embedded multiple times (common during search where the query
    is embedded once per scope).

    The cache is keyed by ``SHA-256(method_name + ":" + text)`` and is capped
    at :data:`_EMBED_CACHE_MAX` entries using LRU eviction.  Only single-item
    calls are cached; multi-item batches bypass the cache.
    """

    _embed_cache: OrderedDict[str, list[float]]

    def _init_embed_cache(self) -> None:
        """Initialize the LRU cache.  Must be called from ``__init__``."""
        self._embed_cache = OrderedDict()

    @staticmethod
    def _cache_key(method_name: str, text: str) -> str:
        """Return a deterministic cache key for *method_name* + *text*."""
        digest = hashlib.sha256(f"{method_name}:{text}".encode("utf-8")).hexdigest()
        return digest

    def _get_cached(self, method_name: str, text: str) -> list[float] | None:
        """Look up a cached vector, promoting it to MRU on hit."""
        key = self._cache_key(method_name, text)
        value = self._embed_cache.get(key)
        if value is not None:
            self._embed_cache.move_to_end(key)
        return value

    def _put_cached(self, method_name: str, text: str, vector: list[float]) -> None:
        """Store *vector* in the cache, evicting the LRU entry if full."""
        key = self._cache_key(method_name, text)
        self._embed_cache[key] = vector
        self._embed_cache.move_to_end(key)
        while len(self._embed_cache) > _EMBED_CACHE_MAX:
            self._embed_cache.popitem(last=False)


class OpenRouterEmbeddingProvider(_CachedEmbeddingMixin):
    """Concrete :class:`EmbeddingProvider` backed by the OpenRouter API.

    Features:

    * **Automatic batching** -- input texts are split into chunks of at most
      :data:`BATCH_SIZE` items and sent as sequential HTTP requests.
    * **Text truncation** -- each text is capped at :data:`MAX_CHARS` characters
      before it reaches the API, preventing oversized payloads.
    * **LRU embedding cache** (via :class:`_CachedEmbeddingMixin`) -- when a
      single text is embedded, the result is cached so repeated lookups of the
      same string skip the network entirely.
    * **Progress reporting** -- callers may pass a *progress_callback* that
      receives ``embedding_batch_start`` / ``embedding_batch_complete`` events
      for each batch.

    Args:
        api_key: OpenRouter API key (``RAG_OR_API_KEY``).
        model: Model identifier, e.g. ``"perplexity/pplx-embed-v1-4b"``.
        dimensions: Optional dimension override sent in the API payload.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_OPENROUTER_MODEL, dimensions: int | None = None):
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client: httpx.Client | None = None
        self._init_embed_cache()

    def _get_client(self) -> httpx.Client:
        """Lazily create and return the shared :class:`httpx.Client`."""
        if self._client is None:
            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )
        return self._client

    def _embed_batch(self, texts: list[str], client: httpx.Client) -> list[list[float]]:
        """POST a single batch of texts and return the embedding vectors.

        Raises:
            RuntimeError: On any non-200 response from the API.
        """
        payload: dict[str, object] = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        resp = client.post(OPENROUTER_EMBED_URL, json=payload)
        if resp.status_code != 200:
            msg = _response_error_message(resp, ("error", "message"), ("message",), ("detail",))
            raise RuntimeError(f"Embedding API error {resp.status_code}: {msg}")
        return [item["embedding"] for item in resp.json()["data"]]

    def _embed_all(
        self,
        texts: list[str],
        *,
        input_kind: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Truncate, batch, and embed all *texts*, emitting progress events.

        Args:
            texts: Raw input strings (may exceed :data:`MAX_CHARS`).
            input_kind: Label for progress events (``"text"`` or ``"code"``).
            progress_callback: Optional callback receiving batch-progress dicts.

        Returns:
            A list of embedding vectors with the same length as *texts*.
        """
        truncated = [t[:MAX_CHARS] for t in texts]
        results: list[list[float]] = []
        client = self._get_client()
        batches = _batch_by_limits(truncated, max_items=BATCH_SIZE)
        total_batches = max(1, len(batches))
        completed_items = 0
        for batch_index, batch in enumerate(batches, start=1):
            _emit_progress(
                progress_callback,
                phase="embedding_batch_start",
                provider=DEFAULT_PROVIDER,
                input_kind=input_kind,
                batch_current=batch_index,
                batch_total=total_batches,
                items_completed=completed_items,
                items_total=len(truncated),
                model=self._model,
            )
            results.extend(self._embed_batch(batch, client))
            completed_items += len(batch)
            _emit_progress(
                progress_callback,
                phase="embedding_batch_complete",
                provider=DEFAULT_PROVIDER,
                input_kind=input_kind,
                batch_current=batch_index,
                batch_total=total_batches,
                items_completed=completed_items,
                items_total=len(truncated),
                model=self._model,
            )
        return results

    def embed_text_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        """Embed natural-language texts (docs, memories).

        Single-item calls are served from the LRU cache when available.

        Returns:
            A list of ``list[float]`` vectors, one per input text.
        """
        if len(texts) == 1:
            cached = self._get_cached("embed_text_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(texts, input_kind="text", progress_callback=progress_callback)
            self._put_cached("embed_text_sync", texts[0], result[0])
            return result
        return self._embed_all(texts, input_kind="text", progress_callback=progress_callback)

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        """Embed source-code chunks for indexing.

        Single-item calls are served from the LRU cache when available.

        Returns:
            A list of ``list[float]`` vectors, one per input text.
        """
        if len(texts) == 1:
            cached = self._get_cached("embed_code_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(texts, input_kind="code", progress_callback=progress_callback)
            self._put_cached("embed_code_sync", texts[0], result[0])
            return result
        return self._embed_all(texts, input_kind="code", progress_callback=progress_callback)

    def embed_code_query_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        """Embed a natural-language query for code search.

        Uses the same model and ``"code"`` input kind as
        :meth:`embed_code_sync`.  Single-item calls are served from the LRU
        cache when available.

        Returns:
            A list of ``list[float]`` vectors, one per input text.
        """
        if len(texts) == 1:
            cached = self._get_cached("embed_code_query_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(texts, input_kind="code", progress_callback=progress_callback)
            self._put_cached("embed_code_query_sync", texts[0], result[0])
            return result
        return self._embed_all(texts, input_kind="code", progress_callback=progress_callback)

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None


def create_embedding_provider() -> EmbeddingProvider:
    """Create and return a configured :class:`OpenRouterEmbeddingProvider`.

    Reads ``RAG_OR_API_KEY`` (required), ``RAG_OR_EMBED_MOD``, and
    ``RAG_OR_EMBED_DIM`` from the environment (with shell-env fallback).

    Raises:
        RuntimeError: If ``RAG_OR_API_KEY`` is missing or blank.
    """
    _load_embedding_env_from_shell()
    api_key = os.environ.get("RAG_OR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("RAG_OR_API_KEY not set")
    return OpenRouterEmbeddingProvider(
        api_key=api_key,
        model=resolve_embedding_model(),
        dimensions=resolve_embedding_dimensions(),
    )


def embedding_provider_status() -> dict[str, object]:
    """Return a lightweight health-check dict for the embedding provider.

    Used by ``vibe-rag doctor`` and the ``project_status`` tool.  The returned
    dict always contains:

    * ``provider`` -- ``"openrouter"``
    * ``ok`` -- ``True`` when ``RAG_OR_API_KEY`` is set, ``False`` otherwise
    * ``detail`` -- ``"ready"`` or a human-readable error string
    * ``model`` -- the resolved model name
    * ``dimensions`` -- the resolved embedding dimensions
    """
    _load_embedding_env_from_shell()
    api_key = os.environ.get("RAG_OR_API_KEY", "").strip()
    if not api_key:
        return {
            "provider": DEFAULT_PROVIDER,
            "ok": False,
            "detail": "RAG_OR_API_KEY not set",
            "model": resolve_embedding_model(),
            "dimensions": resolve_embedding_dimensions(),
        }
    return {
        "provider": DEFAULT_PROVIDER,
        "ok": True,
        "detail": "ready",
        "model": resolve_embedding_model(),
        "dimensions": resolve_embedding_dimensions(),
    }


Embedder = OpenRouterEmbeddingProvider
"""Convenience alias kept for backward compatibility with older imports."""
