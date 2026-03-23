import pytest
import hashlib
import math
import re
from pathlib import Path

from vibe_rag.db.sqlite import SqliteVecDB


class FakeEmbedder:
    """Returns deterministic 1024-dim vectors. No API calls."""

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")
    _ACTIVE_DIMENSIONS = 64
    _TOTAL_DIMENSIONS = 1024

    def _vector(self, text: str) -> list[float]:
        active = [0.0] * self._ACTIVE_DIMENSIONS
        for token in self._TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            first = int.from_bytes(digest[:4], "little") % self._ACTIVE_DIMENSIONS
            second = int.from_bytes(digest[4:], "little") % self._ACTIVE_DIMENSIONS
            active[first] += 1.0
            active[second] += 0.5
        norm = math.sqrt(sum(value * value for value in active)) or 1.0
        normalized = [value / norm for value in active]
        return normalized + [0.0] * (self._TOTAL_DIMENSIONS - self._ACTIVE_DIMENSIONS)

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_code_query_sync(self, texts: list[str]) -> list[list[float]]:
        return self.embed_code_sync(texts)

    def close(self) -> None:
        return None


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
