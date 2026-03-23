from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

import httpx

OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_OPENROUTER_MODEL = "perplexity/pplx-embed-v1-4b"
DEFAULT_OPENROUTER_DIMENSIONS = 2560
DEFAULT_PROVIDER = "openrouter"
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
BATCH_SIZE = 64
MAX_CHARS = 16_000
ProgressCallback = Callable[[dict[str, object]], None]
EMBEDDING_ENV_KEYS = (
    "RAG_OR_API_KEY",
    "RAG_OR_EMBED_MOD",
    "RAG_OR_EMBED_DIM",
)
_SHELL_ENV_ATTEMPTED = False


def _emit_progress(progress_callback: ProgressCallback | None, **event: object) -> None:
    if progress_callback is None:
        return
    progress_callback(event)


def _batch_by_limits(
    texts: list[str],
    *,
    max_items: int,
) -> list[list[str]]:
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
    shell = os.environ.get("SHELL", "").strip() or "/bin/sh"
    shell_path = os.path.realpath(shell)
    if os.path.isabs(shell) and os.access(shell_path, os.X_OK) and shell_path in _TRUSTED_EMBEDDING_SHELLS:
        return shell_path, "-lc"
    return "/bin/sh", "-lc"


def _shell_env_fallback_needed() -> bool:
    return not any(os.environ.get(key, "").strip() for key in EMBEDDING_ENV_KEYS)


def _load_embedding_env_from_shell() -> None:
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
    _load_embedding_env_from_shell()
    return os.environ.get("RAG_OR_EMBED_MOD", "").strip() or DEFAULT_OPENROUTER_MODEL


def resolve_embedding_dimensions() -> int:
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
    return {
        "provider": DEFAULT_PROVIDER,
        "model": resolve_embedding_model(),
        "dimensions": resolve_embedding_dimensions(),
    }


def _response_error_message(resp: httpx.Response, *json_paths: tuple[str, ...]) -> str:
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


class _CachedEmbeddingMixin:
    """Per-instance LRU cache for single-text embedding calls."""

    _embed_cache: OrderedDict[str, list[float]]

    def _init_embed_cache(self) -> None:
        self._embed_cache = OrderedDict()

    @staticmethod
    def _cache_key(method_name: str, text: str) -> str:
        digest = hashlib.sha256(f"{method_name}:{text}".encode("utf-8")).hexdigest()
        return digest

    def _get_cached(self, method_name: str, text: str) -> list[float] | None:
        key = self._cache_key(method_name, text)
        value = self._embed_cache.get(key)
        if value is not None:
            self._embed_cache.move_to_end(key)
        return value

    def _put_cached(self, method_name: str, text: str, vector: list[float]) -> None:
        key = self._cache_key(method_name, text)
        self._embed_cache[key] = vector
        self._embed_cache.move_to_end(key)
        while len(self._embed_cache) > _EMBED_CACHE_MAX:
            self._embed_cache.popitem(last=False)


class OpenRouterEmbeddingProvider(_CachedEmbeddingMixin):
    def __init__(self, api_key: str, model: str = DEFAULT_OPENROUTER_MODEL, dimensions: int | None = None):
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client: httpx.Client | None = None
        self._init_embed_cache()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )
        return self._client

    def _embed_batch(self, texts: list[str], client: httpx.Client) -> list[list[float]]:
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
        if len(texts) == 1:
            cached = self._get_cached("embed_code_query_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(texts, input_kind="code", progress_callback=progress_callback)
            self._put_cached("embed_code_query_sync", texts[0], result[0])
            return result
        return self._embed_all(texts, input_kind="code", progress_callback=progress_callback)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def create_embedding_provider() -> EmbeddingProvider:
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
