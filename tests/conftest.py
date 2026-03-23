import os
import pytest
import hashlib
import math
import re
from pathlib import Path

from vibe_rag.db.sqlite import SqliteVecDB


class FakeEmbedder:
    """Returns deterministic fixed-width vectors. No API calls."""

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")
    _ACTIVE_DIMENSIONS = 64
    _TOTAL_DIMENSIONS = 2560

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
    project_db = SqliteVecDB(tmp_path / "project.db", embedding_dimensions=2560)
    user_db = SqliteVecDB(tmp_path / "user.db", embedding_dimensions=2560)
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


# ---------------------------------------------------------------------------
# Composite fixtures that eliminate repeated setup across test files
# ---------------------------------------------------------------------------

_SAMPLE_PY_AUTH = "def authenticate(user, password):\n    return True\n"
_SAMPLE_PY_BILLING = "def create_invoice(customer_id):\n    return customer_id\n"
_SAMPLE_MD_GUIDE = "## Deployment\n\nDeploy to Railway with docker.\n"


@pytest.fixture
def indexed_project(tmp_db, mock_embedder, tmp_path: Path):
    """Create a temp project with .py and .md files, index it, and yield the path.

    The fixture handles chdir into the project directory and restores the
    original working directory on teardown.  After yielding, both code and
    doc indexes are populated so search_code / search_docs / load_session_context
    can be called immediately.
    """
    from vibe_rag.tools import index_project

    (tmp_path / "auth.py").write_text(_SAMPLE_PY_AUTH)
    (tmp_path / "billing.py").write_text(_SAMPLE_PY_BILLING)
    (tmp_path / "guide.md").write_text(_SAMPLE_MD_GUIDE)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
        assert result["ok"] is True, f"index_project failed: {result}"
        yield tmp_path
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def populated_memory(tmp_db, mock_embedder):
    """Store a few memories of different kinds and yield their IDs.

    Returns a dict mapping memory kind to the ID returned by
    ``remember_structured``, e.g. ``{"decision": 1, "fact": 2, "note": 3}``.
    Useful for search_memory / forget / update_memory tests.
    """
    from vibe_rag.tools import remember_structured

    ids: dict[str, int] = {}

    result = remember_structured(
        summary="gateway owns auth tokens",
        details="the gateway service validates bearer tokens before routing",
        memory_kind="decision",
    )
    assert result["ok"] is True
    ids["decision"] = result["memory"]["id"]

    result = remember_structured(
        summary="max retry count is 5",
        details="the retry policy caps at 5 attempts with exponential backoff",
        memory_kind="fact",
    )
    assert result["ok"] is True
    ids["fact"] = result["memory"]["id"]

    result = remember_structured(
        summary="need to refactor billing module",
        details="billing module has grown too large and should be split",
        memory_kind="note",
    )
    assert result["ok"] is True
    ids["note"] = result["memory"]["id"]

    yield ids


_PROVIDER_ENV_VARS = (
    "RAG_DB",
    "RAG_USER_DB",
    "RAG_OR_API_KEY",
    "RAG_OR_EMBED_MOD",
    "RAG_OR_EMBED_DIM",
    "RAG_OR_API_BASE_URL",
    "RAG_OR_TIMEOUT_SECONDS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove embedding and persistence env vars for a pristine environment.

    Useful for embedder / provider-selection tests that need deterministic
    environment state.  Each variable is deleted via ``monkeypatch.delenv``
    so the original values are automatically restored on teardown.
    """
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
