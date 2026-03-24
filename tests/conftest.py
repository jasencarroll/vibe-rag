import os
import pytest
import hashlib
import math
import re
from pathlib import Path

from vibe_rag.db.sqlite import SqliteVecDB


class FakeEmbedder:
    """Deterministic embedding stub that replaces the real embedding provider in tests.

    Produces fixed-width 2560-dimensional vectors derived from token hashing
    (blake2b), so identical inputs always yield identical vectors.  Only the
    first 64 dimensions carry signal; the remaining 2496 are zero-padded to
    match the default ``VIBE_RAG_EMBEDDING_DIMENSIONS=2560`` used by the
    project's SqliteVecDB instances.

    No network calls, no API keys, no Ollama -- safe for CI and offline use.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")
    _ACTIVE_DIMENSIONS = 64
    _TOTAL_DIMENSIONS = 2560

    def _vector(self, text: str) -> list[float]:
        """Build a deterministic unit vector from *text* via token hashing.

        Each lowercase alphanumeric token is hashed with blake2b to select
        two slots in the active region, then the active region is L2-normalised
        and zero-padded to ``_TOTAL_DIMENSIONS``.
        """
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
        """Embed a batch of plain-text strings (used for doc chunks and memories)."""
        return [self._vector(text) for text in texts]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of code strings (used for code chunks during indexing)."""
        return [self._vector(text) for text in texts]

    def embed_code_query_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a code search query.  Delegates to :meth:`embed_code_sync`."""
        return self.embed_code_sync(texts)

    def close(self) -> None:
        """No-op cleanup -- nothing to release in the fake."""
        return None


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Provide throwaway project and user SQLite-vec databases for one test.

    Sets up:
        - A fresh ``project.db`` and ``user.db`` inside *tmp_path*, both
          initialised with 2560-dimensional vector tables (matching the
          ``FakeEmbedder`` output width).
        - Patches ``vibe_rag.server._project_db`` and ``._user_db`` so
          that all tool functions see the test databases instead of the
          real ones.

    Yields:
        The project ``SqliteVecDB`` instance (the user DB is accessible
        indirectly via ``vibe_rag.server._user_db``).

    Cleans up:
        Restores the original ``_project_db`` and ``_user_db`` globals on
        the server module and closes both database connections.  The
        underlying files are removed automatically when *tmp_path* is
        garbage-collected by pytest.
    """
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
    """Replace the server's embedding provider with a deterministic fake.

    Sets up:
        Instantiates a :class:`FakeEmbedder` and patches
        ``vibe_rag.server._embedder`` so that all tool functions
        (indexing, search, remember) use it instead of a real provider.
        No API keys or Ollama process required.

    Yields:
        The ``FakeEmbedder`` instance, in case the test needs to inspect
        or further customise it.

    Cleans up:
        Restores the original ``_embedder`` reference on the server module.
    """
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
    """Provide a fully indexed mini-project with code and doc files.

    Sets up:
        - Writes three sample files into *tmp_path*:
          ``auth.py`` (authentication stub), ``billing.py`` (invoice stub),
          and ``guide.md`` (deployment guide).
        - Changes the working directory to *tmp_path* so that
          ``index_project()`` discovers the files correctly.
        - Calls ``index_project()`` to populate the code and doc indexes
          in the test database (provided by ``tmp_db`` and ``mock_embedder``).

    Yields:
        The *tmp_path* ``Path`` pointing to the project root.  At this
        point both the code index and the doc index are populated, so
        ``search_code``, ``search_docs``, ``search``, and
        ``load_session_context`` can be called immediately.

    Cleans up:
        Restores the original working directory.  Database and file
        cleanup is handled by the ``tmp_db`` and ``tmp_path`` fixtures.

    Depends on:
        ``tmp_db``, ``mock_embedder`` (via fixture arguments).
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
    """Seed the memory database with three memories of different kinds.

    Sets up:
        Calls ``remember_structured`` three times to insert:

        - **decision** -- "gateway owns auth tokens" (auth-token routing).
        - **fact** -- "max retry count is 5" (retry-policy detail).
        - **note** -- "need to refactor billing module" (tech-debt note).

    Yields:
        ``dict[str, int]`` mapping each memory kind to its database ID,
        e.g. ``{"decision": 1, "fact": 2, "note": 3}``.  These IDs can
        be used with ``search_memory``, ``forget``, ``update_memory``,
        and ``supersede_memory`` in the consuming test.

    Cleans up:
        Nothing explicit -- the in-memory databases created by ``tmp_db``
        are closed and deleted when that fixture tears down.

    Depends on:
        ``tmp_db``, ``mock_embedder`` (via fixture arguments).
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
    """Strip all vibe-rag-related environment variables for a pristine env.

    Sets up:
        Deletes every variable listed in ``_PROVIDER_ENV_VARS`` (DB paths,
        API keys, embedding model/dimension overrides) via
        ``monkeypatch.delenv``.  This guarantees that provider-selection
        logic and DB-path resolution see a blank slate, regardless of
        what the developer's shell exports.

    Yields:
        Nothing -- the fixture's value is the side-effect of clearing the
        environment.  Tests can then ``monkeypatch.setenv`` individual
        variables to exercise specific configurations.

    Cleans up:
        ``monkeypatch`` automatically restores every deleted variable to
        its original value (or absence) when the test finishes.

    Depends on:
        ``monkeypatch`` (pytest built-in).
    """
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
