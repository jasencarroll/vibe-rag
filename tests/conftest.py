import pytest
from pathlib import Path

from vibe_rag.db.sqlite import SqliteVecDB


class FakeEmbedder:
    """Returns deterministic 1536-dim vectors. No API calls."""

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 10) / 10.0] * 1536 for i, _ in enumerate(texts)]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 10) / 10.0 + 0.01] * 1536 for i, _ in enumerate(texts)]


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Provides an initialized SqliteVecDB and patches the server global."""
    import vibe_rag.server as srv
    db = SqliteVecDB(tmp_path / "test.db")
    db.initialize()
    old_db = srv._db
    srv._db = db
    yield db
    srv._db = old_db
    db.close()


@pytest.fixture
def mock_embedder():
    """Patches the server's embedder with a fake that needs no API key."""
    import vibe_rag.server as srv
    fake = FakeEmbedder()
    old_embedder = srv._embedder
    srv._embedder = fake
    yield fake
    srv._embedder = old_embedder
