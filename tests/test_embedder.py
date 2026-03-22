import pytest
import httpx

from vibe_rag.indexing.embedder import Embedder


@pytest.fixture
def embedder():
    return Embedder(mistral_api_key="mk-test", codestral_api_key="ck-test")


def test_embed_text_calls_mistral(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1024}]},
    )
    result = embedder.embed_text_sync(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1024
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert "mistral-embed" in body


def test_embed_code_calls_codestral(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.2] * 1024}]},
    )
    result = embedder.embed_code_sync(["def foo(): pass"])
    assert len(result) == 1
    assert len(result[0]) == 1024
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert "codestral-embed" in body


def test_embed_batches_large_input(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1024} for _ in range(256)]},
    )
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1024} for _ in range(10)]},
    )
    texts = [f"text {i}" for i in range(266)]
    result = embedder.embed_text_sync(texts)
    assert len(result) == 266
    assert len(httpx_mock.get_requests()) == 2
