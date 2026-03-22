from __future__ import annotations
import asyncio
import atexit
import hashlib
import logging
import os
import threading
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
_async_loop: asyncio.AbstractEventLoop | None = None
_async_thread: threading.Thread | None = None
_async_ready = threading.Event()


def _ensure_async_loop() -> asyncio.AbstractEventLoop:
    global _async_loop, _async_thread
    if _async_loop is not None:
        return _async_loop

    with _init_lock:
        if _async_loop is not None:
            return _async_loop

        _async_ready.clear()

        def run_loop() -> None:
            global _async_loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _async_loop = loop
            _async_ready.set()
            loop.run_forever()
            loop.close()

        _async_thread = threading.Thread(target=run_loop, name="vibe-rag-async", daemon=True)
        _async_thread.start()
        _async_ready.wait()
        return _async_loop


def _run_async(coro):
    """Run an async coroutine on the server's dedicated event loop."""
    loop = _ensure_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


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


def _project_id_for_path(path: Path) -> str:
    resolved = path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"{resolved.name}-{digest}"


def _ensure_project_id() -> str:
    global _project_id
    if _project_id is None:
        with _init_lock:
            if _project_id is None:
                _project_id = _project_id_for_path(Path.cwd())
    return _project_id


def _cleanup() -> None:
    global _async_loop, _async_thread
    if _db is not None:
        _db.close()
    if _pg is not None:
        try:
            _run_async(_pg.close())
        except Exception:
            pass
    if _async_loop is not None:
        _async_loop.call_soon_threadsafe(_async_loop.stop)
    if _async_thread is not None:
        _async_thread.join(timeout=1)
    _async_loop = None
    _async_thread = None


atexit.register(_cleanup)


async def _startup() -> None:
    global _pg, _project_id

    _project_id = _project_id_for_path(Path.cwd())

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
    _run_async(_startup())
    mcp.run(transport="stdio")
