from __future__ import annotations

import httpx

EMBED_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_MODEL = "mistral-embed"
CODESTRAL_MODEL = "codestral-embed"
BATCH_SIZE = 256


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
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def _embed_all(self, texts: list[str], model: str) -> list[list[float]]:
        results: list[list[float]] = []
        with httpx.Client() as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                results.extend(self._embed_batch(batch, model, client))
        return results

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed text/prose using mistral-embed."""
        return self._embed_all(texts, MISTRAL_MODEL)

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed code using codestral-embed."""
        return self._embed_all(texts, CODESTRAL_MODEL)
