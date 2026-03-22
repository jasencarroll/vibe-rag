from __future__ import annotations

import httpx

EMBED_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_MODEL = "mistral-embed"
CODESTRAL_MODEL = "codestral-embed"
BATCH_SIZE = 16
MAX_CHARS = 16_000  # ~8K tokens at ~2 chars/token for code, conservative limit


class Embedder:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def _embed_batch(self, texts: list[str], model: str, client: httpx.Client) -> list[list[float]]:
        resp = client.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text[:500]}")
        return [item["embedding"] for item in resp.json()["data"]]

    def _embed_all(self, texts: list[str], model: str) -> list[list[float]]:
        truncated = [t[:MAX_CHARS] for t in texts]
        results: list[list[float]] = []
        with httpx.Client() as client:
            for i in range(0, len(truncated), BATCH_SIZE):
                batch = truncated[i : i + BATCH_SIZE]
                results.extend(self._embed_batch(batch, model, client))
        return results

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed text/prose using codestral-embed."""
        return self._embed_all(texts, CODESTRAL_MODEL)

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed code using codestral-embed."""
        return self._embed_all(texts, CODESTRAL_MODEL)
