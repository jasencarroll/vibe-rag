from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from collections.abc import Callable
from ipaddress import ip_address
from urllib.parse import urlparse
from typing import Protocol

import httpx
from ollama import Client as OllamaClient
from ollama import ResponseError as OllamaResponseError

EMBED_URL = "https://api.mistral.ai/v1/embeddings"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
VOYAGE_EMBED_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "codestral-embed"
DEFAULT_OLLAMA_MODEL = "qwen3-embedding:0.6b"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_VOYAGE_TEXT_MODEL = "voyage-4"
DEFAULT_VOYAGE_CODE_MODEL = "voyage-code-3"
DEFAULT_MISTRAL_DIMENSIONS = 1536
DEFAULT_OLLAMA_DIMENSIONS = 1024
DEFAULT_OPENAI_DIMENSIONS = 1536
DEFAULT_VOYAGE_DIMENSIONS = 1024
REMOTE_OLLAMA_ALLOWLIST = ("localhost", "127.0.0.1", "::1")
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
VOYAGE_BATCH_SIZE = 1000
VOYAGE_MAX_BATCH_TOKENS = 75_000
MAX_CHARS = 16_000
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_HOST_CANDIDATES = (
    "http://localhost:11434",
    "http://127.0.0.1:11434",
)
ProgressCallback = Callable[[dict[str, object]], None]
EMBEDDING_ENV_KEYS = (
    "VIBE_RAG_EMBEDDING_PROVIDER",
    "VIBE_RAG_EMBEDDING_MODEL",
    "VIBE_RAG_CODE_EMBEDDING_MODEL",
    "VIBE_RAG_EMBEDDING_DIMENSIONS",
    "VIBE_RAG_OLLAMA_HOST",
    "OLLAMA_HOST",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "VOYAGE_API_KEY",
)
_SHELL_ENV_ATTEMPTED = False


def _emit_progress(progress_callback: ProgressCallback | None, **event: object) -> None:
    if progress_callback is None:
        return
    progress_callback(event)


def _estimate_tokens(text: str) -> int:
    # Conservative enough for batching against hosted token caps without a provider tokenizer.
    return max(1, (len(text) + 3) // 4)


def _batch_by_limits(
    texts: list[str],
    *,
    max_items: int,
    max_tokens: int | None = None,
) -> list[list[str]]:
    if not texts:
        return []

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        token_estimate = _estimate_tokens(text)
        if current_batch and (
            len(current_batch) >= max_items
            or (max_tokens is not None and current_tokens + token_estimate > max_tokens)
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += token_estimate

    if current_batch:
        batches.append(current_batch)

    return batches


def _allow_remote_ollama_host() -> bool:
    value = os.environ.get("VIBE_RAG_ALLOW_REMOTE_OLLAMA_HOST", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.lower()
    if normalized in REMOTE_OLLAMA_ALLOWLIST:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_ollama_host(host: str) -> str:
    raw_host = host.strip()
    if not raw_host:
        raise RuntimeError("Ollama host is empty")

    parsed = urlparse(raw_host.rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid Ollama host URL: {raw_host}")

    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Invalid Ollama host scheme: {raw_host}")

    hostname = parsed.hostname
    if not hostname:
        raise RuntimeError(f"Invalid Ollama host URL: {raw_host}")

    if not _allow_remote_ollama_host() and not _is_loopback_host(hostname):
        raise RuntimeError(
            "Refusing non-loopback Ollama host. Set VIBE_RAG_ALLOW_REMOTE_OLLAMA_HOST=true to allow remote hosts."
        )

    return raw_host.rstrip("/")


def _resolve_embedding_provider_name() -> str:
    _load_embedding_env_from_shell()
    explicit = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "").strip().lower()
    if explicit:
        return explicit

    try:
        _resolve_ollama_host()
    except RuntimeError as exc:
        raise RuntimeError(
            "No explicit embedding provider configured. "
            "Set VIBE_RAG_EMBEDDING_PROVIDER explicitly when Ollama is unavailable."
        ) from exc
    return "ollama"


def _shell_env_fallback_needed() -> bool:
    explicit = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "").strip().lower()
    if explicit == "mistral":
        return not os.environ.get("MISTRAL_API_KEY", "").strip()
    if explicit == "openai":
        return not os.environ.get("OPENAI_API_KEY", "").strip()
    if explicit == "voyage":
        return not os.environ.get("VOYAGE_API_KEY", "").strip()
    if explicit == "ollama":
        return not (
            os.environ.get("VIBE_RAG_OLLAMA_HOST", "").strip()
            or os.environ.get("OLLAMA_HOST", "").strip()
        )
    return not any(os.environ.get(key, "").strip() for key in EMBEDDING_ENV_KEYS)


def _preferred_shell() -> tuple[str, str]:
    shell = os.environ.get("SHELL", "").strip() or "/bin/sh"
    shell_path = os.path.realpath(shell)
    if os.path.isabs(shell) and os.access(shell_path, os.X_OK) and shell_path in _TRUSTED_EMBEDDING_SHELLS:
        return shell_path, "-lc"
    return "/bin/sh", "-lc"


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


def resolve_embedding_dimensions() -> int:
    raw = os.environ.get("VIBE_RAG_EMBEDDING_DIMENSIONS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError("VIBE_RAG_EMBEDDING_DIMENSIONS must be an integer") from exc
        if value <= 0:
            raise RuntimeError("VIBE_RAG_EMBEDDING_DIMENSIONS must be positive")
        return value

    try:
        provider = _resolve_embedding_provider_name()
    except RuntimeError:
        return DEFAULT_OLLAMA_DIMENSIONS

    if provider == "mistral":
        return DEFAULT_MISTRAL_DIMENSIONS
    if provider == "openai":
        return DEFAULT_OPENAI_DIMENSIONS
    if provider == "voyage":
        return DEFAULT_VOYAGE_DIMENSIONS
    return DEFAULT_OLLAMA_DIMENSIONS


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
    """Per-instance LRU cache for single-text embedding calls.

    When ``embed_text_sync``, ``embed_code_sync``, or ``embed_code_query_sync``
    is called with exactly **one** text, the result is cached keyed on a SHA-256
    digest of the text + method name.  Batch calls (len > 1) bypass the cache
    entirely so indexing workloads are never affected.

    Subclasses must call ``_init_embed_cache()`` in their ``__init__``.
    """

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


class MistralEmbeddingProvider(_CachedEmbeddingMixin):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
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
        resp = client.post(
            EMBED_URL,
            json={"model": self._model, "input": texts},
        )
        if resp.status_code != 200:
            msg = _response_error_message(resp, ("message",), ("error", "message"), ("detail",))
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
                provider="mistral",
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
                provider="mistral",
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


class OllamaEmbeddingProvider(_CachedEmbeddingMixin):
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_OLLAMA_HOST,
        dimensions: int | None = None,
        truncate: bool = True,
    ):
        self._model = model
        self._dimensions = dimensions
        self._truncate = truncate
        self._client = OllamaClient(host=host)
        self._init_embed_cache()

    def _embed_all(
        self,
        texts: list[str],
        *,
        input_kind: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        truncated = [text[:MAX_CHARS] for text in texts]
        results: list[list[float]] = []
        batches = _batch_by_limits(truncated, max_items=BATCH_SIZE)
        total_batches = max(1, len(batches))
        completed_items = 0

        for batch_index, batch in enumerate(batches, start=1):
            kwargs: dict[str, object] = {
                "model": self._model,
                "input": batch,
                "truncate": self._truncate,
            }
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            try:
                _emit_progress(
                    progress_callback,
                    phase="embedding_batch_start",
                    provider="ollama",
                    input_kind=input_kind,
                    batch_current=batch_index,
                    batch_total=total_batches,
                    items_completed=completed_items,
                    items_total=len(truncated),
                    model=self._model,
                )
                response = self._client.embed(**kwargs)
            except OllamaResponseError as exc:
                if exc.status_code == 404:
                    raise RuntimeError(
                        f"Ollama model '{self._model}' is not available. Run: ollama pull {self._model}"
                    ) from exc
                raise RuntimeError(f"Ollama embed failed: {exc.error}") from exc
            except Exception as exc:
                raise RuntimeError(f"Ollama embed failed: {exc}") from exc
            results.extend(list(embedding) for embedding in response["embeddings"])
            completed_items += len(batch)
            _emit_progress(
                progress_callback,
                phase="embedding_batch_complete",
                provider="ollama",
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
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class OpenAIEmbeddingProvider(_CachedEmbeddingMixin):
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL, dimensions: int | None = None):
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
        resp = client.post(OPENAI_EMBED_URL, json=payload)
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
                provider="openai",
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
                provider="openai",
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


class VoyageEmbeddingProvider(_CachedEmbeddingMixin):
    def __init__(
        self,
        api_key: str,
        text_model: str = DEFAULT_VOYAGE_TEXT_MODEL,
        code_model: str = DEFAULT_VOYAGE_CODE_MODEL,
        dimensions: int | None = None,
    ):
        self._text_model = text_model
        self._code_model = code_model
        self._dimensions = dimensions
        self._api_key = api_key
        self._client: httpx.Client | None = None
        self._init_embed_cache()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )
        return self._client

    def _embed_batch(
        self,
        payload: dict[str, object],
        texts: list[str],
        *,
        depth: int = 0,
        max_depth: int = 8,
    ) -> list[list[float]]:
        resp = self._get_client().post(VOYAGE_EMBED_URL, json=payload)
        if resp.status_code == 200:
            return [item["embedding"] for item in resp.json()["data"]]

        msg = _response_error_message(resp, ("detail",), ("message",), ("error", "message"))

        # Voyage enforces a hard per-request token cap. Our estimate is intentionally
        # conservative, but very large markdown/doc batches can still exceed the real
        # tokenizer count. Split and retry instead of failing the whole index run.
        if (
            resp.status_code == 400
            and "max allowed tokens per submitted batch" in msg.lower()
            and len(texts) > 1
        ):
            if depth >= max_depth:
                raise RuntimeError(
                    "Embedding API error 400: voyage batch splitting exhausted max depth"
                )
            midpoint = max(1, len(texts) // 2)
            left = list(texts[:midpoint])
            right = list(texts[midpoint:])
            left_payload = dict(payload)
            left_payload["input"] = left
            right_payload = dict(payload)
            right_payload["input"] = right
            return self._embed_batch(
                left_payload,
                left,
                depth=depth + 1,
                max_depth=max_depth,
            ) + self._embed_batch(
                right_payload,
                right,
                depth=depth + 1,
                max_depth=max_depth,
            )

        raise RuntimeError(f"Embedding API error {resp.status_code}: {msg}")

    def _embed_all(
        self,
        texts: list[str],
        *,
        model: str,
        input_type: str,
        input_kind: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        truncated = [text[:MAX_CHARS] for text in texts]
        results: list[list[float]] = []
        batches = _batch_by_limits(
            truncated,
            max_items=VOYAGE_BATCH_SIZE,
            max_tokens=VOYAGE_MAX_BATCH_TOKENS,
        )
        total_batches = max(1, len(batches))
        completed_items = 0

        for batch_index, batch in enumerate(batches, start=1):
            payload: dict[str, object] = {
                "input": batch,
                "model": model,
                "input_type": input_type,
                "truncation": True,
            }
            if self._dimensions is not None:
                payload["output_dimension"] = self._dimensions
            _emit_progress(
                progress_callback,
                phase="embedding_batch_start",
                provider="voyage",
                input_kind=input_kind,
                batch_current=batch_index,
                batch_total=total_batches,
                items_completed=completed_items,
                items_total=len(truncated),
                model=model,
            )
            results.extend(self._embed_batch(payload, batch))
            completed_items += len(batch)
            _emit_progress(
                progress_callback,
                phase="embedding_batch_complete",
                provider="voyage",
                input_kind=input_kind,
                batch_current=batch_index,
                batch_total=total_batches,
                items_completed=completed_items,
                items_total=len(truncated),
                model=model,
            )
        return results

    def embed_text_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        if len(texts) == 1:
            cached = self._get_cached("embed_text_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(
                texts,
                model=self._text_model,
                input_type="document",
                input_kind="text",
                progress_callback=progress_callback,
            )
            self._put_cached("embed_text_sync", texts[0], result[0])
            return result
        return self._embed_all(
            texts,
            model=self._text_model,
            input_type="document",
            input_kind="text",
            progress_callback=progress_callback,
        )

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        if len(texts) == 1:
            cached = self._get_cached("embed_code_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(
                texts,
                model=self._code_model,
                input_type="document",
                input_kind="code",
                progress_callback=progress_callback,
            )
            self._put_cached("embed_code_sync", texts[0], result[0])
            return result
        return self._embed_all(
            texts,
            model=self._code_model,
            input_type="document",
            input_kind="code",
            progress_callback=progress_callback,
        )

    def embed_code_query_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        if len(texts) == 1:
            cached = self._get_cached("embed_code_query_sync", texts[0])
            if cached is not None:
                return [cached]
            result = self._embed_all(
                texts,
                model=self._code_model,
                input_type="query",
                input_kind="code",
                progress_callback=progress_callback,
            )
            self._put_cached("embed_code_query_sync", texts[0], result[0])
            return result
        return self._embed_all(
            texts,
            model=self._code_model,
            input_type="query",
            input_kind="code",
            progress_callback=progress_callback,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def create_embedding_provider() -> EmbeddingProvider:
    provider = _resolve_embedding_provider_name()
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        model = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return MistralEmbeddingProvider(api_key=api_key, model=model)
    if provider == "ollama":
        model = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
        host = _resolve_ollama_host()
        dimensions_raw = os.environ.get("VIBE_RAG_EMBEDDING_DIMENSIONS", "").strip()
        dimensions = int(dimensions_raw) if dimensions_raw else None
        return OllamaEmbeddingProvider(model=model, host=host, dimensions=dimensions)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        model = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
        dimensions_raw = os.environ.get("VIBE_RAG_EMBEDDING_DIMENSIONS", "").strip()
        dimensions = int(dimensions_raw) if dimensions_raw else None
        return OpenAIEmbeddingProvider(api_key=api_key, model=model, dimensions=dimensions)
    if provider == "voyage":
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set")
        text_model = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", DEFAULT_VOYAGE_TEXT_MODEL).strip() or DEFAULT_VOYAGE_TEXT_MODEL
        code_model = os.environ.get("VIBE_RAG_CODE_EMBEDDING_MODEL", DEFAULT_VOYAGE_CODE_MODEL).strip() or DEFAULT_VOYAGE_CODE_MODEL
        dimensions_raw = os.environ.get("VIBE_RAG_EMBEDDING_DIMENSIONS", "").strip()
        dimensions = int(dimensions_raw) if dimensions_raw else None
        return VoyageEmbeddingProvider(
            api_key=api_key,
            text_model=text_model,
            code_model=code_model,
            dimensions=dimensions,
        )
    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def _resolve_ollama_host() -> str:
    explicit_host = os.environ.get("VIBE_RAG_OLLAMA_HOST", "").strip()
    if explicit_host:
        explicit_host = _validate_ollama_host(explicit_host)
        return explicit_host

    inherited_host = os.environ.get("OLLAMA_HOST", "").strip()
    candidates = [inherited_host] if inherited_host else []
    candidates.extend(host for host in OLLAMA_HOST_CANDIDATES if host not in candidates)

    for host in candidates:
        try:
            validated_host = _validate_ollama_host(host)
            response = httpx.get(f"{validated_host.rstrip('/')}/api/version", timeout=1.0)
        except (httpx.ConnectError, httpx.TimeoutException):
            continue
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama health check failed for {host}: {exc}") from exc
        if response.status_code == 200:
            return host
        raise RuntimeError(f"Ollama reachable at {host} but returned HTTP {response.status_code}")

    searched = ", ".join(candidates)
    raise RuntimeError(f"Ollama not reachable. Set VIBE_RAG_OLLAMA_HOST or OLLAMA_HOST. Tried: {searched}")


def embedding_provider_status() -> dict[str, object]:
    explicit = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "").strip().lower()
    try:
        provider = _resolve_embedding_provider_name()
    except RuntimeError as exc:
        return {
            "provider": explicit or "auto",
            "ok": False,
            "detail": str(exc),
            "model": os.environ.get("VIBE_RAG_EMBEDDING_MODEL", "").strip() or None,
        }
    model = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", "").strip()

    if provider == "mistral":
        return {
            "provider": "mistral",
            "ok": bool(os.environ.get("MISTRAL_API_KEY", "").strip()),
            "detail": "ready" if os.environ.get("MISTRAL_API_KEY", "").strip() else "MISTRAL_API_KEY not set",
            "model": model or DEFAULT_MODEL,
        }

    if provider == "ollama":
        model = model or DEFAULT_OLLAMA_MODEL
        try:
            host = _resolve_ollama_host()
        except RuntimeError as exc:
            return {
                "provider": "ollama",
                "ok": False,
                "detail": str(exc),
                "model": model,
            }
        hostname = urlparse(host).hostname or ""
        remote_host_enabled = bool(hostname and not _is_loopback_host(hostname))
        return {
            "provider": "ollama",
            "ok": True,
            "warning": remote_host_enabled,
            "detail": (
                f"ready ({host}; remote host explicitly allowed)"
                if remote_host_enabled
                else f"ready ({host})"
            ),
            "model": model,
        }

    if provider == "openai":
        return {
            "provider": "openai",
            "ok": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            "detail": "ready" if os.environ.get("OPENAI_API_KEY", "").strip() else "OPENAI_API_KEY not set",
            "model": model or DEFAULT_OPENAI_MODEL,
        }

    if provider == "voyage":
        return {
            "provider": "voyage",
            "ok": bool(os.environ.get("VOYAGE_API_KEY", "").strip()),
            "detail": "ready" if os.environ.get("VOYAGE_API_KEY", "").strip() else "VOYAGE_API_KEY not set",
            "model": model or DEFAULT_VOYAGE_TEXT_MODEL,
        }

    return {
        "provider": provider,
        "ok": False,
        "detail": f"Unsupported embedding provider: {provider}",
        "model": model or None,
    }


Embedder = MistralEmbeddingProvider
