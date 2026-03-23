import pytest
import httpx
from ollama import ResponseError

from vibe_rag.indexing.embedder import (
    Embedder,
    DEFAULT_OLLAMA_MODEL,
    MistralEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
    embedding_provider_status,
    create_embedding_provider,
)


@pytest.fixture
def embedder():
    return Embedder(api_key="test-key")


def test_embed_text_calls_codestral(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1536}]},
    )
    result = embedder.embed_text_sync(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1536
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert "codestral-embed" in body


def test_embed_code_calls_codestral(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.2] * 1536}]},
    )
    result = embedder.embed_code_sync(["def foo(): pass"])
    assert len(result) == 1
    assert len(result[0]) == 1536
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert "codestral-embed" in body


def test_embed_batches_large_input(httpx_mock):
    # Batch size is 64, so 100 texts = 2 batches (64 + 36)
    embedder = Embedder(api_key="test-key")
    for count in [64, 36]:
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 1536} for _ in range(count)]},
        )
    texts = [f"text {i}" for i in range(100)]
    result = embedder.embed_text_sync(texts)
    assert len(result) == 100
    assert len(httpx_mock.get_requests()) == 2


def test_embed_truncates_long_input(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1536}]},
    )
    long_text = "x" * 50_000  # way over 16K limit
    result = embedder.embed_text_sync([long_text])
    assert len(result) == 1
    # Verify the request was truncated
    request = httpx_mock.get_request()
    import json
    body = json.loads(request.content)
    assert len(body["input"][0]) == 16_000


def test_create_embedding_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: "http://localhost:11434",
    )

    provider = create_embedding_provider()

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider._model == DEFAULT_OLLAMA_MODEL


def test_create_embedding_provider_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "unknown")

    with pytest.raises(RuntimeError, match="Unsupported embedding provider"):
        create_embedding_provider()


def test_create_embedding_provider_supports_ollama(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: "http://localhost:11434",
    )

    provider = create_embedding_provider()

    assert isinstance(provider, OllamaEmbeddingProvider)


def test_ollama_provider_calls_embed_with_dimensions(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, host):
            self.host = host

        def embed(self, **kwargs):
            calls.append(kwargs)
            return {"embeddings": [[0.1, 0.2, 0.3]]}

    monkeypatch.setattr("vibe_rag.indexing.embedder.OllamaClient", FakeClient)
    provider = OllamaEmbeddingProvider(
        model="qwen3-embedding:0.6b",
        host="http://localhost:11434",
        dimensions=1024,
    )

    result = provider.embed_text_sync(["hello world"])

    assert result == [[0.1, 0.2, 0.3]]
    assert calls[0]["model"] == "qwen3-embedding:0.6b"
    assert calls[0]["input"] == ["hello world"]
    assert calls[0]["dimensions"] == 1024


def test_resolve_ollama_host_prefers_explicit_host(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_OLLAMA_HOST", "http://192.168.1.5:11434")

    from vibe_rag.indexing.embedder import _resolve_ollama_host

    assert _resolve_ollama_host() == "http://192.168.1.5:11434"


def test_resolve_ollama_host_checks_common_local_hosts(monkeypatch, httpx_mock):
    monkeypatch.delenv("VIBE_RAG_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    httpx_mock.add_exception(httpx.ConnectError("nope"), url="http://localhost:11434/api/version")
    httpx_mock.add_response(url="http://127.0.0.1:11434/api/version", json={"version": "0.6.1"})

    from vibe_rag.indexing.embedder import _resolve_ollama_host

    assert _resolve_ollama_host() == "http://127.0.0.1:11434"


def test_ollama_provider_reports_missing_model(monkeypatch):
    class FakeClient:
        def __init__(self, host):
            self.host = host

        def embed(self, **kwargs):
            raise ResponseError("model not found", status_code=404)

    monkeypatch.setattr("vibe_rag.indexing.embedder.OllamaClient", FakeClient)
    provider = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b")

    with pytest.raises(RuntimeError, match="ollama pull qwen3-embedding:0.6b"):
        provider.embed_text_sync(["hello"])


def test_embedding_provider_status_for_ollama(monkeypatch, httpx_mock):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.delenv("VIBE_RAG_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    httpx_mock.add_exception(httpx.ConnectError("nope"), url="http://localhost:11434/api/version")
    httpx_mock.add_response(url="http://127.0.0.1:11434/api/version", json={"version": "0.6.1"})

    status = embedding_provider_status()

    assert status["provider"] == "ollama"
    assert status["ok"] is True
    assert "127.0.0.1:11434" in status["detail"]


def test_create_embedding_provider_supports_openai(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_DIMENSIONS", "1024")

    provider = create_embedding_provider()

    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_openai_provider_calls_embeddings_endpoint(httpx_mock):
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=1024,
    )
    httpx_mock.add_response(
        url="https://api.openai.com/v1/embeddings",
        json={"data": [{"embedding": [0.4] * 1024}]},
    )

    result = provider.embed_text_sync(["hello world"])

    assert len(result) == 1
    assert len(result[0]) == 1024
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert "text-embedding-3-small" in body
    assert "1024" in body


def test_embedding_provider_status_for_openai(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    status = embedding_provider_status()

    assert status["provider"] == "openai"
    assert status["ok"] is True
    assert status["model"] == "text-embedding-3-small"


def test_create_embedding_provider_supports_voyage(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "voyage-4")
    monkeypatch.setenv("VIBE_RAG_CODE_EMBEDDING_MODEL", "voyage-code-3")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_DIMENSIONS", "1024")

    provider = create_embedding_provider()

    assert isinstance(provider, VoyageEmbeddingProvider)


def test_voyage_provider_uses_document_and_query_modes(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.9, 0.8, 0.7]}]},
        )

    transport = httpx.MockTransport(handler)
    provider = VoyageEmbeddingProvider(
        api_key="test-key",
        text_model="voyage-4",
        code_model="voyage-code-3",
        dimensions=1024,
    )
    provider._client = httpx.Client(transport=transport)

    provider.embed_text_sync(["hello world"])
    provider.embed_code_sync(["def hello(): pass"])

    first = requests[0]
    second = requests[1]
    assert first.url == httpx.URL("https://api.voyageai.com/v1/embeddings")
    assert second.url == httpx.URL("https://api.voyageai.com/v1/embeddings")
    assert b'"model":"voyage-4"' in first.content
    assert b'"input_type":"document"' in first.content
    assert b'"output_dimension":1024' in first.content
    assert b'"model":"voyage-code-3"' in second.content
    assert b'"input_type":"query"' in second.content


def test_embedding_provider_status_for_voyage(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    status = embedding_provider_status()

    assert status["provider"] == "voyage"
    assert status["ok"] is True
    assert status["model"] == "voyage-4"
