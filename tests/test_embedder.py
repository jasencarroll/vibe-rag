import json

import httpx
import pytest

from vibe_rag.indexing import embedder
from vibe_rag.indexing.embedder import (
    DEFAULT_OPENROUTER_DIMENSIONS,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterEmbeddingProvider,
    _batch_by_limits,
    create_embedding_provider,
    embedding_provider_status,
    resolve_embedding_dimensions,
    resolve_embedding_model,
    resolve_embedding_profile,
)

def test_batch_by_limits_splits_by_item_count():
    assert _batch_by_limits(["a", "b", "c"], max_items=2) == [["a", "b"], ["c"]]
    assert _batch_by_limits(["a", "b", "c"], max_items=3) == [["a", "b", "c"]]
    assert _batch_by_limits([], max_items=3) == []


def test_resolve_embedding_model_defaults_to_openrouter(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    assert resolve_embedding_model() == DEFAULT_OPENROUTER_MODEL


def test_resolve_embedding_model_reads_env(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_EMBED_MOD", "custom/openrouter-model")
    assert resolve_embedding_model() == "custom/openrouter-model"


def test_resolve_embedding_model_reads_user_config(isolate_user_embedding_config, monkeypatch):
    isolate_user_embedding_config.parent.mkdir(parents=True, exist_ok=True)
    isolate_user_embedding_config.write_text('[embedding]\nmodel = "custom/openrouter-model"\n')
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    assert resolve_embedding_model() == "custom/openrouter-model"


def test_resolve_embedding_dimensions_defaults_to_openrouter_dim(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    assert resolve_embedding_dimensions() == DEFAULT_OPENROUTER_DIMENSIONS


def test_resolve_embedding_dimensions_accepts_explicit_integer(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_EMBED_DIM", "1024")
    assert resolve_embedding_dimensions() == 1024


def test_resolve_embedding_dimensions_reads_user_config(isolate_user_embedding_config, monkeypatch):
    isolate_user_embedding_config.parent.mkdir(parents=True, exist_ok=True)
    isolate_user_embedding_config.write_text("[embedding]\ndimensions = 1536\n")
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    assert resolve_embedding_dimensions() == 1536


def test_resolve_embedding_dimensions_rejects_invalid_integer(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_EMBED_DIM", "abc")
    with pytest.raises(RuntimeError, match="RAG_OR_EMBED_DIM must be an integer"):
        resolve_embedding_dimensions()


def test_resolve_embedding_dimensions_rejects_non_positive(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_EMBED_DIM", "0")
    with pytest.raises(RuntimeError, match="RAG_OR_EMBED_DIM must be positive"):
        resolve_embedding_dimensions()


def test_resolve_embedding_profile_exposes_provider_metadata(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    profile = resolve_embedding_profile()
    assert profile["provider"] == "openrouter"
    assert profile["model"] == DEFAULT_OPENROUTER_MODEL
    assert profile["dimensions"] == DEFAULT_OPENROUTER_DIMENSIONS


def test_embedding_provider_status_ready_when_key_present(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_API_KEY", "test-key")
    monkeypatch.setenv("RAG_OR_EMBED_MOD", "custom/model")
    monkeypatch.setenv("RAG_OR_EMBED_DIM", "512")

    status = embedding_provider_status()

    assert status["provider"] == "openrouter"
    assert status["ok"] is True
    assert status["detail"] == "ready"
    assert status["model"] == "custom/model"
    assert status["dimensions"] == 512


def test_embedding_provider_status_not_ready_without_key(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)

    status = embedding_provider_status()

    assert status["provider"] == "openrouter"
    assert status["ok"] is False
    assert status["detail"] == "RAG_OR_API_KEY not set"
    assert status["dimensions"] == DEFAULT_OPENROUTER_DIMENSIONS


def test_embedding_provider_status_reports_invalid_user_config(isolate_user_embedding_config, monkeypatch):
    isolate_user_embedding_config.parent.mkdir(parents=True, exist_ok=True)
    isolate_user_embedding_config.write_text("[embedding]\ndimensions = 'abc'\n")
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)

    status = embedding_provider_status()

    assert status["ok"] is False
    assert "[embedding].dimensions must be an integer" in status["detail"]
    assert status["model"] is None
    assert status["dimensions"] is None


def test_create_embedding_provider_builds_openrouter_client(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_API_KEY", "test-key")
    monkeypatch.setenv("RAG_OR_EMBED_MOD", "perplexity/pplx-embed-v1-4b")
    monkeypatch.setenv("RAG_OR_EMBED_DIM", "1024")

    provider = create_embedding_provider()

    assert isinstance(provider, OpenRouterEmbeddingProvider)
    assert provider._model == "perplexity/pplx-embed-v1-4b"
    assert provider._dimensions == 1024


def test_create_embedding_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RAG_OR_API_KEY not set"):
        create_embedding_provider()


def test_create_embedding_provider_reads_user_config(isolate_user_embedding_config, monkeypatch):
    isolate_user_embedding_config.parent.mkdir(parents=True, exist_ok=True)
    isolate_user_embedding_config.write_text(
        '[embedding]\napi_key = "user-key"\nmodel = "custom/model"\ndimensions = 512\n'
    )
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)

    provider = create_embedding_provider()

    assert isinstance(provider, OpenRouterEmbeddingProvider)
    assert provider._api_key == "user-key"
    assert provider._model == "custom/model"
    assert provider._dimensions == 512


def test_create_embedding_provider_combines_env_and_user_config(isolate_user_embedding_config, monkeypatch):
    isolate_user_embedding_config.parent.mkdir(parents=True, exist_ok=True)
    isolate_user_embedding_config.write_text('[embedding]\nmodel = "config/model"\ndimensions = 384\n')
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_API_KEY", "env-key")
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)

    provider = create_embedding_provider()

    assert provider._api_key == "env-key"
    assert provider._model == "config/model"
    assert provider._dimensions == 384


def test_create_embedding_provider_prefers_env_over_user_config(isolate_user_embedding_config, monkeypatch):
    isolate_user_embedding_config.parent.mkdir(parents=True, exist_ok=True)
    isolate_user_embedding_config.write_text('[embedding]\napi_key = "user-key"\nmodel = "config/model"\n')
    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", lambda: None)
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.setenv("RAG_OR_API_KEY", "env-key")
    monkeypatch.setenv("RAG_OR_EMBED_MOD", "env/model")

    provider = create_embedding_provider()

    assert provider._api_key == "env-key"
    assert provider._model == "env/model"


def test_create_embedding_provider_uses_shell_fallback_when_user_config_missing(isolate_user_embedding_config, monkeypatch):
    monkeypatch.setattr(embedder, "_SHELL_ENV_ATTEMPTED", False)
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)

    def fake_shell_load():
        monkeypatch.setenv("RAG_OR_API_KEY", "shell-key")
        monkeypatch.setenv("RAG_OR_EMBED_MOD", "shell/model")
        monkeypatch.setenv("RAG_OR_EMBED_DIM", "256")

    monkeypatch.setattr(embedder, "_load_embedding_env_from_shell", fake_shell_load)

    provider = create_embedding_provider()

    assert provider._api_key == "shell-key"
    assert provider._model == "shell/model"
    assert provider._dimensions == 256


def test_openrouter_provider_posts_model_and_dimensions(httpx_mock, monkeypatch):
    provider = OpenRouterEmbeddingProvider(api_key="test-key", model="perplexity/pplx-embed-v1-4b", dimensions=123)
    payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    httpx_mock.add_response(
        method="POST",
        url="https://openrouter.ai/api/v1/embeddings",
        json=payload,
    )

    result = provider.embed_text_sync(["hello world"])

    request = httpx_mock.get_request()
    body = json.loads(request.content)
    assert result == [[0.1, 0.2, 0.3]]
    assert body["model"] == "perplexity/pplx-embed-v1-4b"
    assert body["input"] == ["hello world"]
    assert body["dimensions"] == 123
    assert request.headers["authorization"] == "Bearer test-key"


def test_openrouter_provider_batches_large_input(httpx_mock):
    provider = OpenRouterEmbeddingProvider(api_key="test-key", model=DEFAULT_OPENROUTER_MODEL, dimensions=2560)
    httpx_mock.add_response(
        method="POST",
        url="https://openrouter.ai/api/v1/embeddings",
        json={"data": [{"embedding": [0.1, 0.2, 0.3]}] * 64},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://openrouter.ai/api/v1/embeddings",
        json={"data": [{"embedding": [0.1, 0.2, 0.3]}] * 36},
    )

    result = provider.embed_text_sync([f"chunk-{index}" for index in range(100)])

    assert len(result) == 100
    assert len(httpx_mock.get_requests()) == 2


def test_openrouter_provider_progress_events_for_batching():
    events = []
    provider = OpenRouterEmbeddingProvider(api_key="test-key", model=DEFAULT_OPENROUTER_MODEL, dimensions=2560)

    def handler(request):
        payload = json.loads(request.read().decode())
        text_count = len(payload["input"])
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(text_count)]},
        )

    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    provider.embed_code_sync([f"chunk {i}" for i in range(100)], progress_callback=events.append)

    assert len(events) == 4
    assert events[0]["phase"] == "embedding_batch_start"
    assert events[1]["phase"] == "embedding_batch_complete"
    assert events[2]["phase"] == "embedding_batch_start"
    assert events[3]["phase"] == "embedding_batch_complete"
    assert events[3]["batch_total"] == 2
    assert events[-1]["items_completed"] == 100


def test_openrouter_provider_caches_single_item_embeddings(httpx_mock):
    provider = OpenRouterEmbeddingProvider(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://openrouter.ai/api/v1/embeddings",
        json={"data": [{"embedding": [0.4, 0.5, 0.6]}]},
    )

    first = provider.embed_text_sync(["cache me"])
    second = provider.embed_text_sync(["cache me"])

    assert first == second
    assert len(httpx_mock.get_requests()) == 1


def test_openrouter_provider_handles_code_and_query_embeddings(httpx_mock):
    provider = OpenRouterEmbeddingProvider(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://openrouter.ai/api/v1/embeddings",
        json={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://openrouter.ai/api/v1/embeddings",
        json={"data": [{"embedding": [0.4, 0.5, 0.6]}]},
    )

    assert provider.embed_code_sync(["def hello()"]) == [[0.1, 0.2, 0.3]]
    assert provider.embed_code_query_sync(["def hello()"]) == [[0.4, 0.5, 0.6]]


def test_openrouter_provider_close_releases_client():
    provider = OpenRouterEmbeddingProvider(api_key="test-key")
    provider._client = httpx.Client()
    provider.close()
    assert provider._client is None
