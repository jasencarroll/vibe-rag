from __future__ import annotations
import atexit
import hashlib
import os
import threading
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from vibe_rag.db.sqlite import SqliteVecDB
from vibe_rag.indexing.embedder import EmbeddingProvider, create_embedding_provider

mcp = FastMCP(name="vibe-rag")

_api_key = os.environ.get("MISTRAL_API_KEY", "")
_project_db: SqliteVecDB | None = None
_user_db: SqliteVecDB | None = None
_embedder: EmbeddingProvider | None = None
_project_id: str | None = None
_init_lock = threading.Lock()


def _embedding_dimensions() -> int:
    raw = os.environ.get("VIBE_RAG_EMBEDDING_DIMENSIONS", "1536").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("VIBE_RAG_EMBEDDING_DIMENSIONS must be an integer") from exc
    if value <= 0:
        raise RuntimeError("VIBE_RAG_EMBEDDING_DIMENSIONS must be positive")
    return value


def _project_db_path() -> Path:
    db_path_raw = os.environ.get("VIBE_RAG_DB", "")
    if db_path_raw:
        return Path(db_path_raw).resolve()
    return (Path.cwd() / ".vibe" / "index.db").resolve()


def _user_db_path() -> Path:
    db_path_raw = os.environ.get("VIBE_RAG_USER_DB", "")
    if db_path_raw:
        return Path(db_path_raw).resolve()
    return (Path.home() / ".vibe" / "memory.db").resolve()


def _get_db() -> SqliteVecDB:
    global _project_db
    if _project_db is None:
        with _init_lock:
            if _project_db is None:
                _project_db = SqliteVecDB(_project_db_path(), embedding_dimensions=_embedding_dimensions())
                _project_db.initialize()
    return _project_db


def _get_user_db() -> SqliteVecDB:
    global _user_db
    if _user_db is None:
        with _init_lock:
            if _user_db is None:
                _user_db = SqliteVecDB(_user_db_path(), embedding_dimensions=_embedding_dimensions())
                _user_db.initialize()
    return _user_db


def _get_embedder() -> EmbeddingProvider:
    global _embedder
    if _embedder is None:
        with _init_lock:
            if _embedder is None:
                _embedder = create_embedding_provider()
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
    if _project_db is not None:
        _project_db.close()
    if _user_db is not None:
        _user_db.close()


atexit.register(_cleanup)


# MUST be after mcp, _get_db, _get_user_db, _get_embedder are defined
import vibe_rag.tools  # noqa: F401, E402


def run_server() -> None:
    global _project_id
    _project_id = _project_id_for_path(Path.cwd())
    mcp.run(transport="stdio")
