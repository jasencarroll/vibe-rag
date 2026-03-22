from __future__ import annotations
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from vibe_rag.db.sqlite import SqliteVecDB
from vibe_rag.indexing.embedder import Embedder

mcp = FastMCP(name="vibe-rag")

_db_path = Path(os.environ.get("VIBE_RAG_DB", Path.cwd() / ".vibe" / "index.db"))
_api_key = os.environ.get("MISTRAL_API_KEY", "")
_db: SqliteVecDB | None = None
_embedder: Embedder | None = None

def _get_db() -> SqliteVecDB:
    global _db
    if _db is None:
        _db = SqliteVecDB(_db_path)
        _db.initialize()
    return _db


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        if not _api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        _embedder = Embedder(api_key=_api_key)
    return _embedder

# MUST be after mcp, _get_db, _get_embedder are defined — tools.py imports them
import vibe_rag.tools  # noqa: F401, E402

def run_server() -> None:
    mcp.run(transport="stdio")
