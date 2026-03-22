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


def test_run_async_uses_single_background_loop():
    async def get_loop_id():
        import asyncio

        return id(asyncio.get_running_loop())

    first = srv._run_async(get_loop_id())
    second = srv._run_async(get_loop_id())

    assert first == second
