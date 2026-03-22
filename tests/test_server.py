"""Tests for vibe_rag.server edge cases."""
from pathlib import Path

import pytest

try:
    import vibe_rag.server as srv
    _server_available = True
except ImportError:
    _server_available = False

pytestmark = pytest.mark.skipif(not _server_available, reason="server module has import errors")


def test_ensure_project_id_returns_cwd_name():
    old = srv._project_id
    srv._project_id = None
    try:
        pid = srv._ensure_project_id()
        assert pid == Path.cwd().name
        assert srv._project_id == pid
    finally:
        srv._project_id = old


def test_get_embedder_without_key_raises():
    old_embedder = srv._embedder
    old_key = srv._api_key
    srv._embedder = None
    srv._api_key = ""
    try:
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            srv._get_embedder()
    finally:
        srv._embedder = old_embedder
        srv._api_key = old_key


def test_get_db_creates_and_returns_db(tmp_path: Path, monkeypatch):
    old_db = srv._db
    srv._db = None
    monkeypatch.setenv("VIBE_RAG_DB", str(tmp_path / "test_srv.db"))
    try:
        db = srv._get_db()
        assert db is not None
        tables = db.list_tables()
        assert "memories" in tables
        assert "code_chunks" in tables
    finally:
        if srv._db:
            srv._db.close()
        srv._db = old_db
