from __future__ import annotations
import asyncio
import atexit
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from vibe_rag.db.sqlite import SqliteVecDB
from vibe_rag.db.postgres import PostgresDB
from vibe_rag.indexing.embedder import Embedder

logger = logging.getLogger(__name__)

mcp = FastMCP(name="vibe-rag")

_api_key = os.environ.get("MISTRAL_API_KEY", "")
_database_url = os.environ.get("DATABASE_URL", "")
_db: SqliteVecDB | None = None
_pg: PostgresDB | None = None
_embedder: Embedder | None = None
_project_id: str | None = None
_init_lock = threading.Lock()
_pg_executor = ThreadPoolExecutor(max_workers=1)


def _run_async(coro):
    """Run an async coroutine from sync context, even inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _pg_executor.submit(asyncio.run, coro).result()


def _get_db() -> SqliteVecDB:
    global _db
    if _db is None:
        with _init_lock:
            if _db is None:
                db_path_raw = os.environ.get("VIBE_RAG_DB", "")
                if db_path_raw:
                    db_path = Path(db_path_raw).resolve()
                else:
                    db_path = (Path.cwd() / ".vibe" / "index.db").resolve()
                _db = SqliteVecDB(db_path)
                _db.initialize()
    return _db


def _get_pg() -> PostgresDB | None:
    return _pg


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _init_lock:
            if _embedder is None:
                if not _api_key:
                    raise RuntimeError("MISTRAL_API_KEY not set")
                _embedder = Embedder(api_key=_api_key)
    return _embedder


def _ensure_project_id() -> str:
    global _project_id
    if _project_id is None:
        with _init_lock:
            if _project_id is None:
                _project_id = Path.cwd().name
    return _project_id


def _cleanup() -> None:
    if _db is not None:
        _db.close()
    if _pg is not None:
        try:
            _run_async(_pg.close())
        except Exception:
            pass
    _pg_executor.shutdown(wait=False)


atexit.register(_cleanup)


async def _startup() -> None:
    global _pg, _project_id

    _project_id = Path.cwd().name

    if _database_url:
        try:
            _pg = PostgresDB(_database_url)
            await _pg.connect()
        except Exception:
            logger.warning("pgvector unavailable (connection failed)")
            _pg = None


# MUST be after mcp, _get_db, _get_embedder, _get_pg are defined
import vibe_rag.tools  # noqa: F401, E402


def run_server() -> None:
    asyncio.run(_startup())
    mcp.run(transport="stdio")
