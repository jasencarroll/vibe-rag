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
    old_embedder = srv._embedder
    srv._embedder = None
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: (_ for _ in ()).throw(RuntimeError("Ollama not reachable")),
    )
    try:
        with pytest.raises(RuntimeError, match="Ollama not reachable"):
            srv._get_embedder()
    finally:
        srv._embedder = old_embedder


def test_get_db_creates_and_returns_db(tmp_path: Path, monkeypatch):
    old_db = srv._project_db
    srv._project_db = None
    monkeypatch.setenv("VIBE_RAG_DB", str(tmp_path / "test_srv.db"))
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


def test_get_user_db_creates_and_returns_db(tmp_path: Path, monkeypatch):
    old_db = srv._user_db
    srv._user_db = None
    monkeypatch.setenv("VIBE_RAG_USER_DB", str(tmp_path / "user_srv.db"))
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
