from __future__ import annotations

import httpx

EMBED_URL = "https://api.mistral.ai/v1/embeddings"
CODESTRAL_MODEL = "codestral-embed"
BATCH_SIZE = 64
MAX_CHARS = 16_000  # ~8K tokens at ~2 chars/token for code, conservative limit


class Embedder:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )
        return self._client

    def _embed_batch(self, texts: list[str], model: str, client: httpx.Client) -> list[list[float]]:
        resp = client.post(
            EMBED_URL,
            json={"model": model, "input": texts},
        )
        if resp.status_code != 200:
            try:
                msg = resp.json().get("message", "unknown error")
            except Exception:
                msg = "unknown error"
            raise RuntimeError(f"Embedding API error {resp.status_code}: {msg}")
        return [item["embedding"] for item in resp.json()["data"]]

    def _embed_all(self, texts: list[str], model: str) -> list[list[float]]:
        truncated = [t[:MAX_CHARS] for t in texts]
        results: list[list[float]] = []
        client = self._get_client()
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
