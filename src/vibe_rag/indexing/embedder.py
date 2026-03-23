from __future__ import annotations

import os
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
MAX_CHARS = 16_000
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_HOST_CANDIDATES = (
    "http://localhost:11434",
    "http://127.0.0.1:11434",
)


class EmbeddingProvider(Protocol):
    def embed_text_sync(self, texts: list[str]) -> list[list[float]]: ...

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]: ...


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

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        truncated = [t[:MAX_CHARS] for t in texts]
        results: list[list[float]] = []
        client = self._get_client()
        for i in range(0, len(truncated), BATCH_SIZE):
            batch = truncated[i : i + BATCH_SIZE]
            results.extend(self._embed_batch(batch, client))
        return results

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts)

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts)


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

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, object] = {
            "model": self._model,
            "input": texts,
            "truncate": self._truncate,
        }
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        try:
            response = self._client.embed(**kwargs)
        except OllamaResponseError as exc:
            if exc.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{self._model}' is not available. Run: ollama pull {self._model}"
                ) from exc
            raise RuntimeError(f"Ollama embed failed: {exc.error}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama embed failed: {exc}") from exc
        return [list(embedding) for embedding in response["embeddings"]]

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts)

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts)


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

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        truncated = [t[:MAX_CHARS] for t in texts]
        results: list[list[float]] = []
        client = self._get_client()
        for i in range(0, len(truncated), BATCH_SIZE):
            batch = truncated[i : i + BATCH_SIZE]
            results.extend(self._embed_batch(batch, client))
        return results

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts)

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts)


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

    def _embed_all(self, texts: list[str], model: str, input_type: str) -> list[list[float]]:
        payload: dict[str, object] = {
            "input": texts,
            "model": model,
            "input_type": input_type,
            "truncation": True,
        }
        if self._dimensions is not None:
            payload["output_dimension"] = self._dimensions
        resp = self._get_client().post(VOYAGE_EMBED_URL, json=payload)
        if resp.status_code != 200:
            try:
                msg = resp.json().get("detail") or resp.json().get("message") or "unknown error"
            except Exception:
                msg = "unknown error"
            raise RuntimeError(f"Embedding API error {resp.status_code}: {msg}")
        return [item["embedding"] for item in resp.json()["data"]]

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, model=self._text_model, input_type="document")

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, model=self._code_model, input_type="query")


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
