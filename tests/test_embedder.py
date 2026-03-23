import subprocess

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
    _batch_by_limits,
    embedding_provider_status,
    create_embedding_provider,
    resolve_embedding_dimensions,
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


def test_embed_batches_emit_progress_events(httpx_mock):
    embedder = MistralEmbeddingProvider(api_key="test-key")
    events = []
    for count in [64, 36]:
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 1536} for _ in range(count)]},
        )

    embedder.embed_text_sync([f"text {i}" for i in range(100)], progress_callback=events.append)

    assert events[0]["phase"] == "embedding_batch_start"
    assert events[1]["phase"] == "embedding_batch_complete"
    assert events[-1]["batch_current"] == 2
    assert events[-1]["items_completed"] == 100


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


def test_create_embedding_provider_requires_explicit_provider_when_ollama_unreachable(monkeypatch):
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: (_ for _ in ()).throw(RuntimeError("Ollama not reachable")),
    )

    with pytest.raises(RuntimeError, match="No explicit embedding provider configured"):
        create_embedding_provider()


def test_embedding_provider_status_reports_no_explicit_provider_when_ollama_unreachable(monkeypatch):
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: (_ for _ in ()).throw(RuntimeError("Ollama not reachable")),
    )

    status = embedding_provider_status()

    assert status["provider"] == "auto"
    assert status["ok"] is False
    assert status["detail"].startswith("No explicit embedding provider configured")


def test_create_embedding_provider_recovers_voyage_from_shell_env_with_explicit_provider(monkeypatch):
    from vibe_rag.indexing.embedder import EMBEDDING_ENV_KEYS
    for key in EMBEDDING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr("vibe_rag.indexing.embedder._SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=b"VOYAGE_API_KEY=test-voyage-key\0VIBE_RAG_EMBEDDING_MODEL=voyage-4\0",
            stderr=b"",
        ),
    )

    provider = create_embedding_provider()

    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider._text_model == "voyage-4"


def test_create_embedding_provider_ignores_untrusted_shell_path(monkeypatch):
    from vibe_rag.indexing.embedder import EMBEDDING_ENV_KEYS

    for key in EMBEDDING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHELL", "/tmp/evil-shell")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr("vibe_rag.indexing.embedder._SHELL_ENV_ATTEMPTED", False)
    captured: dict[str, list[str] | str] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = list(args[0])
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=b"VOYAGE_API_KEY=test-voyage-key\0VIBE_RAG_EMBEDDING_MODEL=voyage-4\0",
            stderr=b"",
        )

    monkeypatch.setattr("vibe_rag.indexing.embedder.subprocess.run", fake_run)

    provider = create_embedding_provider()

    assert captured["args"][0] == "/bin/sh"
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider._text_model == "voyage-4"


def test_embedding_provider_status_recovers_voyage_from_shell_env_with_explicit_provider(monkeypatch):
    from vibe_rag.indexing.embedder import EMBEDDING_ENV_KEYS
    for key in EMBEDDING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr("vibe_rag.indexing.embedder._SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=b"VOYAGE_API_KEY=test-voyage-key\0VIBE_RAG_EMBEDDING_MODEL=voyage-4\0",
            stderr=b"",
        ),
    )

    status = embedding_provider_status()

    assert status["provider"] == "voyage"
    assert status["ok"] is True
    assert status["model"] == "voyage-4"


def test_resolve_embedding_dimensions_defaults_to_ollama_when_ollama_unreachable(monkeypatch):
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: (_ for _ in ()).throw(RuntimeError("Ollama not reachable")),
    )

    assert resolve_embedding_dimensions() == 1024


def test_resolve_embedding_dimensions_explicit_provider_defaults_to_provider_value(monkeypatch):
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.delenv("VIBE_RAG_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    assert resolve_embedding_dimensions() == 1536


def test_resolve_embedding_dimensions_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_DIMENSIONS", "2048")

    assert resolve_embedding_dimensions() == 2048


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


def test_ollama_provider_batches_large_input(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, host):
            self.host = host

        def embed(self, **kwargs):
            calls.append(kwargs)
            return {"embeddings": [[0.1, 0.2, 0.3] for _ in kwargs["input"]]}

    monkeypatch.setattr("vibe_rag.indexing.embedder.OllamaClient", FakeClient)
    provider = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b", host="http://localhost:11434")

    result = provider.embed_code_sync([f"chunk {i}" for i in range(100)])

    assert len(result) == 100
    assert len(calls) == 2
    assert len(calls[0]["input"]) == 64
    assert len(calls[1]["input"]) == 36


def test_ollama_provider_batches_emit_progress_events(monkeypatch):
    calls = []
    events = []

    class FakeClient:
        def __init__(self, host):
            self.host = host

        def embed(self, **kwargs):
            calls.append(kwargs)
            return {"embeddings": [[0.1, 0.2, 0.3] for _ in kwargs["input"]]}

    monkeypatch.setattr("vibe_rag.indexing.embedder.OllamaClient", FakeClient)
    provider = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b", host="http://localhost:11434")

    provider.embed_text_sync([f"text {i}" for i in range(100)], progress_callback=events.append)

    assert len(calls) == 2
    assert events[0]["phase"] == "embedding_batch_start"
    assert events[1]["phase"] == "embedding_batch_complete"
    assert events[-1]["batch_current"] == 2
    assert events[-1]["items_completed"] == 100


def test_resolve_ollama_host_prefers_explicit_host(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_OLLAMA_HOST", "http://localhost:11434")

    from vibe_rag.indexing.embedder import _resolve_ollama_host

    assert _resolve_ollama_host() == "http://localhost:11434"


def test_resolve_ollama_host_rejects_remote_host_without_opt_in(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_OLLAMA_HOST", "http://192.168.1.5:11434")

    from vibe_rag.indexing.embedder import _resolve_ollama_host

    with pytest.raises(RuntimeError, match="Refusing non-loopback Ollama host"):
        _resolve_ollama_host()


def test_resolve_ollama_host_allows_remote_host_with_opt_in(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_OLLAMA_HOST", "http://192.168.1.5:11434")
    monkeypatch.setenv("VIBE_RAG_ALLOW_REMOTE_OLLAMA_HOST", "true")

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


def test_embedding_provider_status_warns_for_remote_ollama_host(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VIBE_RAG_OLLAMA_HOST", "http://192.168.1.5:11434")
    monkeypatch.setenv("VIBE_RAG_ALLOW_REMOTE_OLLAMA_HOST", "true")
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: "http://192.168.1.5:11434",
    )

    status = embedding_provider_status()

    assert status["provider"] == "ollama"
    assert status["ok"] is True
    assert status["warning"] is True
    assert "remote host explicitly allowed" in status["detail"]


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
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "text-embedding-3-small")

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
    provider.embed_code_query_sync(["def hello(): pass"])

    first = requests[0]
    second = requests[1]
    third = requests[2]
    assert first.url == httpx.URL("https://api.voyageai.com/v1/embeddings")
    assert second.url == httpx.URL("https://api.voyageai.com/v1/embeddings")
    assert third.url == httpx.URL("https://api.voyageai.com/v1/embeddings")
    assert b'"model":"voyage-4"' in first.content
    assert b'"input_type":"document"' in first.content
    assert b'"output_dimension":1024' in first.content
    assert b'"model":"voyage-code-3"' in second.content
    assert b'"input_type":"document"' in second.content
    assert b'"model":"voyage-code-3"' in third.content
    assert b'"input_type":"query"' in third.content


def test_embedding_provider_status_for_voyage(monkeypatch):
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "voyage-4")

    status = embedding_provider_status()

    assert status["provider"] == "voyage"
    assert status["ok"] is True
    assert status["model"] == "voyage-4"


def test_voyage_provider_batches_large_input():
    requests = []

    def handler(request):
        requests.append(request)
        body = request.read().decode()
        item_count = body.count('"input_type"')
        assert item_count == 1
        import json

        payload = json.loads(body)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.5, 0.4, 0.3]} for _ in payload["input"]]},
        )

    transport = httpx.MockTransport(handler)
    provider = VoyageEmbeddingProvider(api_key="test-key")
    provider._client = httpx.Client(transport=transport)

    result = provider.embed_code_sync([f"chunk {i}" for i in range(1109)])

    assert len(result) == 1109
    assert len(requests) == 2


def test_voyage_provider_batches_emit_progress_events():
    events = []

    def handler(request):
        import json

        payload = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.5, 0.4, 0.3]} for _ in payload["input"]]},
        )

    transport = httpx.MockTransport(handler)
    provider = VoyageEmbeddingProvider(api_key="test-key")
    provider._client = httpx.Client(transport=transport)

    provider.embed_code_sync([f"chunk {i}" for i in range(1109)], progress_callback=events.append)

    assert events[0]["phase"] == "embedding_batch_start"
    assert events[1]["phase"] == "embedding_batch_complete"
    assert events[-1]["batch_current"] == 2
    assert events[-1]["items_completed"] == 1109


def test_batch_by_limits_splits_on_token_budget_before_item_budget():
    texts = ["x" * 16000 for _ in range(30)]

    batches = _batch_by_limits(texts, max_items=1000, max_tokens=100000)

    assert len(batches) == 2
    assert len(batches[0]) == 25
    assert len(batches[1]) == 5


def test_voyage_provider_retries_with_smaller_batches_on_token_cap():
    requests = []

    def handler(request):
        import json

        payload = json.loads(request.read().decode())
        requests.append(len(payload["input"]))
        if len(payload["input"]) > 1:
            return httpx.Response(
                400,
                json={
                    "detail": (
                        "Request to model 'voyage-code-3' failed. "
                        "The max allowed tokens per submitted batch is 120000. "
                        "Your batch has 131126 tokens after truncation. Please lower the number of tokens in the batch."
                    )
                },
            )
        return httpx.Response(200, json={"data": [{"embedding": [0.5, 0.4, 0.3]}]})

    transport = httpx.MockTransport(handler)
    provider = VoyageEmbeddingProvider(api_key="test-key")
    provider._client = httpx.Client(transport=transport)

    result = provider.embed_text_sync(["x" * 16000, "y" * 16000])

    assert len(result) == 2
    assert requests == [2, 1, 1]


@pytest.mark.parametrize(
    ("provider_factory", "client_attr"),
    [
        (lambda: MistralEmbeddingProvider(api_key="test-key"), "_client"),
        (lambda: OpenAIEmbeddingProvider(api_key="test-key"), "_client"),
        (lambda: VoyageEmbeddingProvider(api_key="test-key"), "_client"),
    ],
)
def test_hosted_embedding_providers_close_httpx_clients(provider_factory, client_attr):
    provider = provider_factory()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    setattr(provider, client_attr, client)

    provider.close()

    assert getattr(provider, client_attr) is None
    assert client.is_closed is True


# ---------------------------------------------------------------------------
# Embedding cache tests
# ---------------------------------------------------------------------------


def test_mistral_embed_cache_avoids_duplicate_api_call(httpx_mock):
    """Embedding the same single text twice should only hit the API once."""
    provider = MistralEmbeddingProvider(api_key="test-key")
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1536}]},
    )

    result1 = provider.embed_text_sync(["hello world"])
    result2 = provider.embed_text_sync(["hello world"])

    assert result1 == result2
    assert len(httpx_mock.get_requests()) == 1


def test_openai_embed_cache_avoids_duplicate_api_call(httpx_mock):
    """OpenAI provider: same single text twice -> one API call."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    httpx_mock.add_response(
        url="https://api.openai.com/v1/embeddings",
        json={"data": [{"embedding": [0.2] * 1536}]},
    )

    result1 = provider.embed_text_sync(["hello world"])
    result2 = provider.embed_text_sync(["hello world"])

    assert result1 == result2
    assert len(httpx_mock.get_requests()) == 1


def test_ollama_embed_cache_avoids_duplicate_call(monkeypatch):
    """Ollama provider: same single text twice -> one embed() call."""
    calls = []

    class FakeClient:
        def __init__(self, host):
            self.host = host

        def embed(self, **kwargs):
            calls.append(kwargs)
            return {"embeddings": [[0.3, 0.4, 0.5]]}

    monkeypatch.setattr("vibe_rag.indexing.embedder.OllamaClient", FakeClient)
    provider = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b", host="http://localhost:11434")

    result1 = provider.embed_text_sync(["hello world"])
    result2 = provider.embed_text_sync(["hello world"])

    assert result1 == result2
    assert len(calls) == 1


def test_voyage_embed_cache_avoids_duplicate_api_call():
    """Voyage provider: same single text twice -> one API call."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [0.6, 0.7, 0.8]}]})

    transport = httpx.MockTransport(handler)
    provider = VoyageEmbeddingProvider(api_key="test-key")
    provider._client = httpx.Client(transport=transport)

    result1 = provider.embed_text_sync(["hello world"])
    result2 = provider.embed_text_sync(["hello world"])

    assert result1 == result2
    assert len(requests) == 1


def test_embed_cache_separates_text_and_code_methods(httpx_mock):
    """embed_text_sync and embed_code_sync for the same text should each call the API."""
    provider = MistralEmbeddingProvider(api_key="test-key")
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1536}]},
    )
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.2] * 1536}]},
    )

    provider.embed_text_sync(["hello world"])
    provider.embed_code_sync(["hello world"])

    assert len(httpx_mock.get_requests()) == 2


def test_embed_cache_skipped_for_batch_calls(httpx_mock):
    """Batch calls (len > 1) should bypass the cache entirely."""
    provider = MistralEmbeddingProvider(api_key="test-key")
    for _ in range(2):
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 1536}, {"embedding": [0.2] * 1536}]},
        )

    provider.embed_text_sync(["hello", "world"])
    provider.embed_text_sync(["hello", "world"])

    assert len(httpx_mock.get_requests()) == 2


def test_embed_cache_evicts_oldest_entries(httpx_mock):
    """Cache should evict oldest entries when exceeding _EMBED_CACHE_MAX."""
    from vibe_rag.indexing.embedder import _EMBED_CACHE_MAX

    provider = MistralEmbeddingProvider(api_key="test-key")

    # Fill cache beyond max
    for i in range(_EMBED_CACHE_MAX + 10):
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/embeddings",
            json={"data": [{"embedding": [float(i)] * 1536}]},
        )
        provider.embed_text_sync([f"text-{i}"])

    assert len(provider._embed_cache) == _EMBED_CACHE_MAX
