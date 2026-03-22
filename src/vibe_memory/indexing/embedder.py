from __future__ import annotations

import httpx

EMBED_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_MODEL = "mistral-embed"
CODESTRAL_MODEL = "codestral-embed"
BATCH_SIZE = 256


class Embedder:
    def __init__(self, mistral_api_key: str, codestral_api_key: str):
        self._mistral_key = mistral_api_key
        self._codestral_key = codestral_api_key

    def _embed_batch(self, texts: list[str], model: str, api_key: str, client: httpx.Client) -> list[list[float]]:
        resp = client.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def _embed_all(self, texts: list[str], model: str, api_key: str) -> list[list[float]]:
        results: list[list[float]] = []
        with httpx.Client() as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                results.extend(self._embed_batch(batch, model, api_key, client))
        return results

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, MISTRAL_MODEL, self._mistral_key)

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, CODESTRAL_MODEL, self._codestral_key)

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                resp = await client.post(
                    EMBED_URL,
                    headers={"Authorization": f"Bearer {self._mistral_key}"},
                    json={"model": MISTRAL_MODEL, "input": batch},
                    timeout=60.0,
                )
                resp.raise_for_status()
                results.extend([item["embedding"] for item in resp.json()["data"]])
        return results

    async def embed_code(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                resp = await client.post(
                    EMBED_URL,
                    headers={"Authorization": f"Bearer {self._codestral_key}"},
                    json={"model": CODESTRAL_MODEL, "input": batch},
                    timeout=60.0,
                )
                resp.raise_for_status()
                results.extend([item["embedding"] for item in resp.json()["data"]])
        return results
