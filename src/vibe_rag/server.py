"""FastMCP server spine: ``mcp`` instance and lazy-init singletons for DBs + embedder.

Imported by every ``vibe_rag.tools`` submodule to register ``@mcp.tool()``
decorators and access project/user databases.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vibe_rag.db.sqlite import SqliteVecDB
from vibe_rag.indexing.embedder import (
    EmbeddingProvider,
    create_embedding_provider,
    resolve_embedding_dimensions,
)
from vibe_rag.paths import project_index_db_path, user_memory_db_path

mcp = FastMCP(name="vibe-rag")

_project_db: SqliteVecDB | None = None
_user_db: SqliteVecDB | None = None
_embedder: EmbeddingProvider | None = None
_project_id: str | None = None
_init_lock = threading.Lock()  # guards double-checked locking for all singletons
logger = logging.getLogger(__name__)


def _embedding_dimensions() -> int:
    """Resolve configured embedding vector dimensions."""
    return resolve_embedding_dimensions()


def _project_db_path() -> Path:
    """Return resolved path to the project index DB (override via ``RAG_DB`` env var)."""
    db_path_raw = os.environ.get("RAG_DB", "")
    if db_path_raw:
        return Path(db_path_raw).resolve()
    return project_index_db_path().resolve()


def _user_db_path() -> Path:
    """Return resolved path to the user memory DB (override via ``RAG_USER_DB`` env var)."""
    db_path_raw = os.environ.get("RAG_USER_DB", "")
    if db_path_raw:
        return Path(db_path_raw).resolve()
    return user_memory_db_path().resolve()


def _get_db() -> SqliteVecDB:
    """Return the project index DB singleton, creating it on first call (thread-safe)."""
    global _project_db
    if _project_db is None:  # fast path: no lock needed once initialized
        with _init_lock:
            if _project_db is None:  # double-checked locking
                _project_db = SqliteVecDB(
                    _project_db_path(), embedding_dimensions=_embedding_dimensions()
                )
                _project_db.initialize()
    return _project_db


def _get_user_db() -> SqliteVecDB:
    """Return the user memory DB singleton, creating it on first call (thread-safe)."""
    global _user_db
    if _user_db is None:  # fast path
        with _init_lock:
            if _user_db is None:  # double-checked locking
                _user_db = SqliteVecDB(
                    _user_db_path(), embedding_dimensions=_embedding_dimensions()
                )
                _user_db.initialize()
    return _user_db


def _get_embedder() -> EmbeddingProvider:
    """Return the embedding provider singleton, creating it on first call (thread-safe)."""
    global _embedder
    if _embedder is None:  # fast path
        with _init_lock:
            if _embedder is None:  # double-checked locking
                _embedder = create_embedding_provider()
    return _embedder


def _project_id_for_path(path: Path) -> str:
    """Derive a stable project identifier from *path* (``<dirname>-<sha1[:12]>``)."""
    resolved = path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"{resolved.name}-{digest}"


def _ensure_project_id() -> str:
    """Return the current project id, computing it lazily from cwd (thread-safe)."""
    global _project_id
    if _project_id is None:  # fast path
        with _init_lock:
            if _project_id is None:  # double-checked locking
                _project_id = _project_id_for_path(Path.cwd())
    return _project_id


def _cleanup() -> None:
    """Best-effort shutdown of open DB connections and embedder (registered via atexit)."""
    global _embedder
    if _project_db is not None:
        try:
            _project_db.close()
        except Exception as exc:
            logger.warning("project DB close failed: %s", exc)
    if _user_db is not None:
        try:
            _user_db.close()
        except Exception as exc:
            logger.warning("user DB close failed: %s", exc)
    if _embedder is not None:
        close = getattr(_embedder, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("embedder close failed: %s", exc)
        _embedder = None


atexit.register(_cleanup)  # ensure DBs/embedder are closed on interpreter exit


# Register all @mcp.tool() decorators by importing the tools package.
# This MUST happen after ``mcp`` and the singleton accessors (_get_db,
# _get_user_db, _get_embedder, _ensure_project_id) are defined, because
# every tool submodule does ``from vibe_rag.server import mcp, ...`` at
# import time.
import vibe_rag.tools  # noqa: F401, E402


def run_server() -> None:
    """Pin the project id to cwd and start the MCP server on stdio transport."""
    global _project_id
    _project_id = _project_id_for_path(Path.cwd())
    mcp.run(transport="stdio")
