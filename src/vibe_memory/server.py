from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vibe_memory.config import load_config, Config, resolve_project_id
from vibe_memory.db.postgres import PostgresDB
from vibe_memory.db.sqlite import SqliteVecDB
from vibe_memory.indexing.embedder import Embedder
from vibe_memory.indexing.session_indexer import (
    find_completed_sessions,
    parse_session_messages,
    chunk_session_text,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(name="vibe-memory")

# Global state
_config: Config | None = None
_pg: PostgresDB | None = None
_sqlite: SqliteVecDB | None = None
_embedder: Embedder | None = None


async def _startup() -> None:
    global _config, _pg, _sqlite, _embedder

    _config = load_config()

    if _config.mistral_api_key or _config.codestral_api_key:
        _embedder = Embedder(
            mistral_api_key=_config.mistral_api_key or "",
            codestral_api_key=_config.codestral_api_key or "",
        )
    else:
        logger.warning("No embedding API keys found. Embedding tools disabled.")

    if _config.database_url:
        try:
            _pg = PostgresDB(_config.database_url)
            await _pg.connect()
            logger.info("Connected to pgvector")
        except Exception as e:
            logger.warning(f"pgvector unavailable: {e}")
            _pg = None
    else:
        logger.warning("No DATABASE_URL. Memory features disabled.")

    index_path = Path.cwd() / ".vibe" / "index.db"
    if index_path.exists():
        _sqlite = SqliteVecDB(index_path)
        _sqlite.initialize()
        logger.info(f"Loaded code index: {_sqlite.chunk_count()} chunks")
    else:
        _sqlite = None
        logger.info("No code index found. Use index_project to create one.")

    if _pg and _embedder:
        asyncio.create_task(_catch_up_sessions())


async def _catch_up_sessions() -> None:
    try:
        indexed_ids = await _pg.get_indexed_session_ids()
        completed = find_completed_sessions(_config.session_log_dir)
        unindexed = [s for s in completed if s["session_id"] not in indexed_ids]
        if not unindexed:
            logger.info("Session catch-up: all sessions indexed")
            return
        logger.info(f"Session catch-up: indexing {len(unindexed)} sessions")
        for session in unindexed:
            try:
                await _index_one_session(session)
            except Exception as e:
                logger.warning(f"Failed to index session {session['session_id']}: {e}")
    except Exception as e:
        logger.warning(f"Session catch-up failed: {e}")


async def _index_one_session(session: dict) -> None:
    messages_path = session["session_dir"] / "messages.jsonl"
    if not messages_path.exists():
        return
    text = parse_session_messages(messages_path)
    if not text.strip():
        return
    chunks_text = chunk_session_text(text)
    embeddings = await _embedder.embed_text(chunks_text)
    meta = session["meta"]
    working_dir = meta.get("environment", {}).get("working_directory", "")
    if working_dir:
        project_id = resolve_project_id(Path(working_dir))
    else:
        project_id = None
    chunks = []
    for i, content in enumerate(chunks_text):
        chunks.append({
            "session_id": session["session_id"],
            "chunk_index": i,
            "content": content,
            "project_id": project_id,
            "summary": None,
            "session_start": meta.get("start_time"),
            "session_end": meta.get("end_time"),
        })
    await _pg.upsert_session_chunks(chunks, embeddings)
    logger.info(f"Indexed session {session['session_id']} ({len(chunks)} chunks)")


# Register tools
from vibe_memory.tools.search_code import register as register_search_code
from vibe_memory.tools.search_memory import register as register_search_memory
from vibe_memory.tools.remember import register as register_remember
from vibe_memory.tools.forget import register as register_forget
from vibe_memory.tools.index_project import register as register_index_project
from vibe_memory.tools.ingest_doc import register as register_ingest_doc

register_search_code(mcp)
register_search_memory(mcp)
register_remember(mcp)
register_forget(mcp)
register_index_project(mcp)
register_ingest_doc(mcp)


def run_server() -> None:
    @contextlib.asynccontextmanager
    async def lifespan(server):
        await _startup()
        yield

    mcp._lifespan = lifespan
    mcp.run(transport="stdio")
