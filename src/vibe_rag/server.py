from __future__ import annotations
import asyncio
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from vibe_rag.db.sqlite import SqliteVecDB
from vibe_rag.db.postgres import PostgresDB
from vibe_rag.indexing.embedder import Embedder

mcp = FastMCP(name="vibe-rag")

_db_path = Path(os.environ.get("VIBE_RAG_DB", Path.cwd() / ".vibe" / "index.db"))
_api_key = os.environ.get("MISTRAL_API_KEY", "")
_database_url = os.environ.get("DATABASE_URL", "")
_db: SqliteVecDB | None = None
_pg: PostgresDB | None = None
_embedder: Embedder | None = None
_project_id: str | None = None


def _get_db() -> SqliteVecDB:
    global _db
    if _db is None:
        _db = SqliteVecDB(_db_path)
        _db.initialize()
    return _db


def _get_pg() -> PostgresDB | None:
    """Returns the pgvector connection, or None if DATABASE_URL not set."""
    return _pg


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        if not _api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        _embedder = Embedder(api_key=_api_key)
    return _embedder


def _get_project_id() -> str | None:
    return _project_id


async def _startup() -> None:
    global _pg, _project_id

    # Resolve project ID from manifest or directory name
    cwd = Path.cwd()
    _project_id = cwd.name  # simple fallback, good enough

    # Connect to pgvector if DATABASE_URL is set
    if _database_url:
        try:
            _pg = PostgresDB(_database_url)
            await _pg.connect()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"pgvector unavailable: {e}")
            _pg = None


# MUST be after mcp, _get_db, _get_embedder, _get_pg are defined
import vibe_rag.tools  # noqa: F401, E402


def run_server() -> None:
    # Run startup (connects pgvector) then start MCP server
    asyncio.get_event_loop().run_until_complete(_startup())
    mcp.run(transport="stdio")
