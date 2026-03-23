from __future__ import annotations

import os
from collections.abc import Callable
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


class EmbeddingProvider(Protocol):
    def embed_text_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]: ...


class MistralEmbeddingProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
        self._client: httpx.Client | None = None

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
            try:
                msg = resp.json().get("message", "unknown error")
            except Exception:
                msg = "unknown error"
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
        return self._embed_all(texts, input_kind="text", progress_callback=progress_callback)

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        return self._embed_all(texts, input_kind="code", progress_callback=progress_callback)


class OllamaEmbeddingProvider:
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
        return self._embed_all(texts, input_kind="text", progress_callback=progress_callback)

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        return self._embed_all(texts, input_kind="code", progress_callback=progress_callback)


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL, dimensions: int | None = None):
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client: httpx.Client | None = None

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
            try:
                error = resp.json().get("error", {})
                msg = error.get("message", "unknown error")
            except Exception:
                msg = "unknown error"
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
        return self._embed_all(texts, input_kind="text", progress_callback=progress_callback)

    def embed_code_sync(
        self, texts: list[str], *, progress_callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        return self._embed_all(texts, input_kind="code", progress_callback=progress_callback)


class VoyageEmbeddingProvider:
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

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )
        return self._client

    def _embed_batch(self, payload: dict[str, object], texts: list[str]) -> list[list[float]]:
        resp = self._get_client().post(VOYAGE_EMBED_URL, json=payload)
        if resp.status_code == 200:
            return [item["embedding"] for item in resp.json()["data"]]

        try:
            msg = resp.json().get("detail") or resp.json().get("message") or "unknown error"
        except Exception:
            msg = "unknown error"

        # Voyage enforces a hard per-request token cap. Our estimate is intentionally
        # conservative, but very large markdown/doc batches can still exceed the real
        # tokenizer count. Split and retry instead of failing the whole index run.
        if (
            resp.status_code == 400
            and "max allowed tokens per submitted batch" in msg.lower()
            and len(texts) > 1
        ):
            midpoint = max(1, len(texts) // 2)
            left = list(texts[:midpoint])
            right = list(texts[midpoint:])
            left_payload = dict(payload)
            left_payload["input"] = left
            right_payload = dict(payload)
            right_payload["input"] = right
            return self._embed_batch(left_payload, left) + self._embed_batch(right_payload, right)

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
        return self._embed_all(
            texts,
            model=self._code_model,
            input_type="query",
            input_kind="code",
            progress_callback=progress_callback,
        )


def create_embedding_provider() -> EmbeddingProvider:
    provider = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "ollama").strip().lower() or "ollama"
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
        return explicit_host

    inherited_host = os.environ.get("OLLAMA_HOST", "").strip()
    candidates = [inherited_host] if inherited_host else []
    candidates.extend(host for host in OLLAMA_HOST_CANDIDATES if host not in candidates)

    for host in candidates:
        try:
            response = httpx.get(f"{host.rstrip('/')}/api/version", timeout=1.0)
        except httpx.HTTPError:
            continue
        if response.status_code == 200:
            return host

    searched = ", ".join(candidates)
    raise RuntimeError(f"Ollama not reachable. Set VIBE_RAG_OLLAMA_HOST or OLLAMA_HOST. Tried: {searched}")


def embedding_provider_status() -> dict[str, object]:
    provider = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "ollama").strip().lower() or "ollama"
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
        return {
            "provider": "ollama",
            "ok": True,
            "detail": f"ready ({host})",
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
