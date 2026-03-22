import pytest
import httpx

from vibe_rag.indexing.embedder import Embedder


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
