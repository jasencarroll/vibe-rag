"""Tests for vibe_rag.server edge cases."""
import hashlib
from pathlib import Path

import pytest

try:
    import vibe_rag.server as srv
    _server_available = True
except ImportError:
    _server_available = False

pytestmark = pytest.mark.skipif(not _server_available, reason="server module has import errors")


def test_project_id_for_path_uses_name_and_path_hash(tmp_path: Path):
    expected = f"{tmp_path.name}-{hashlib.sha1(str(tmp_path.resolve()).encode('utf-8')).hexdigest()[:12]}"
    assert srv._project_id_for_path(tmp_path) == expected


def test_ensure_project_id_returns_path_derived_id():
    old = srv._project_id
    srv._project_id = None
    try:
        pid = srv._ensure_project_id()
        expected = srv._project_id_for_path(Path.cwd())
        assert pid == expected
        assert srv._project_id == pid
    finally:
        srv._project_id = old


def test_get_embedder_without_key_raises(monkeypatch):
    """_get_embedder raises RuntimeError when RAG_OR_API_KEY is unset."""
    old_embedder = srv._embedder
    srv._embedder = None
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    monkeypatch.delenv("RAG_DB", raising=False)
    monkeypatch.delenv("RAG_USER_DB", raising=False)
    try:
        with pytest.raises(RuntimeError, match=r"RAG_OR_API_KEY not set; checked .*config.toml"):
            srv._get_embedder()
    finally:
        srv._embedder = old_embedder


def test_project_db_path_defaults_to_vibe_rag(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_DB", raising=False)
    assert srv._project_db_path() == (tmp_path / ".vibe-rag" / "index.db").resolve()


def test_user_db_path_defaults_to_vibe_rag(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("RAG_USER_DB", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert srv._user_db_path() == (tmp_path / ".vibe-rag" / "memory.db").resolve()


def test_get_db_creates_and_returns_db(tmp_path: Path, monkeypatch):
    old_db = srv._project_db
    srv._project_db = None
    monkeypatch.setenv("RAG_DB", str(tmp_path / "test_srv.db"))
    try:
        db = srv._get_db()
        assert db is not None
        tables = db.list_tables()
        assert "memories" in tables
        assert "code_chunks" in tables
    finally:
        if srv._project_db:
            srv._project_db.close()
        srv._project_db = old_db


def test_get_db_uses_provider_default_dimensions_when_env_unset(tmp_path: Path, monkeypatch):
    old_db = srv._project_db
    srv._project_db = None
    captured = {}

    class FakeDB:
        def __init__(self, path, embedding_dimensions):
            captured["path"] = path
            captured["embedding_dimensions"] = embedding_dimensions

        def initialize(self):
            return None

    monkeypatch.setenv("RAG_DB", str(tmp_path / "test_srv.db"))
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.setenv("RAG_OR_API_KEY", "test-key")
    monkeypatch.setattr(srv, "SqliteVecDB", FakeDB)
    try:
        srv._get_db()
        assert captured["embedding_dimensions"] == 2560
    finally:
        srv._project_db = old_db


def test_get_user_db_creates_and_returns_db(tmp_path: Path, monkeypatch):
    old_db = srv._user_db
    srv._user_db = None
    monkeypatch.setenv("RAG_USER_DB", str(tmp_path / "user_srv.db"))
    try:
        db = srv._get_user_db()
        assert db is not None
        tables = db.list_tables()
        assert "memories" in tables
    finally:
        if srv._user_db:
            srv._user_db.close()
        srv._user_db = old_db


def test_cleanup_closes_embedder_and_clears_global():
    """_cleanup calls embedder.close() and resets _embedder to None."""

    class FakeEmbedder:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    old_project_db = srv._project_db
    old_user_db = srv._user_db
    old_embedder = srv._embedder
    srv._project_db = None
    srv._user_db = None
    fake = FakeEmbedder()
    srv._embedder = fake
    try:
        srv._cleanup()
        assert fake.closed is True
        assert srv._embedder is None
    finally:
        srv._project_db = old_project_db
        srv._user_db = old_user_db
        srv._embedder = old_embedder
