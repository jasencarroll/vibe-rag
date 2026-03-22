import pytest
import httpx

from vibe_rag.indexing.embedder import Embedder


@pytest.fixture
def embedder():
    return Embedder(api_key="test-key")


def test_embed_text_calls_mistral(embedder, httpx_mock):
    httpx_mock.add_response(
        url="https://api.mistral.ai/v1/embeddings",
        json={"data": [{"embedding": [0.1] * 1536}]},
    )
    result = embedder.embed_text_sync(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1536
    request = httpx_mock.get_request()
    body = request.content.decode()
    assert "mistral-embed" in body


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


def test_embed_batches_large_input(embedder, httpx_mock):
    # Batch size is 16, so 35 texts = 3 batches (16 + 16 + 3)
    for count in [16, 16, 3]:
        httpx_mock.add_response(
            url="https://api.mistral.ai/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 1536} for _ in range(count)]},
        )
    texts = [f"text {i}" for i in range(35)]
    result = embedder.embed_text_sync(texts)
    assert len(result) == 35
    assert len(httpx_mock.get_requests()) == 3


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
