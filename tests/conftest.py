import pytest
from pathlib import Path

from vibe_rag.db.sqlite import SqliteVecDB


class FakeEmbedder:
    """Returns deterministic 1024-dim vectors. No API calls."""

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 10) / 10.0] * 1024 for i, _ in enumerate(texts)]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 10) / 10.0 + 0.01] * 1024 for i, _ in enumerate(texts)]


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Provides an initialized SqliteVecDB and patches the server global."""
    import vibe_rag.server as srv
    project_db = SqliteVecDB(tmp_path / "project.db")
    user_db = SqliteVecDB(tmp_path / "user.db")
    project_db.initialize()
    user_db.initialize()
    old_project_db = srv._project_db
    old_user_db = srv._user_db
    srv._project_db = project_db
    srv._user_db = user_db
    yield project_db
    srv._project_db = old_project_db
    srv._user_db = old_user_db
    project_db.close()
    user_db.close()


@pytest.fixture
def mock_embedder():
    """Patches the server's embedder with a fake that needs no API key."""
    import vibe_rag.server as srv
    fake = FakeEmbedder()
    old_embedder = srv._embedder
    srv._embedder = fake
    yield fake
    srv._embedder = old_embedder
