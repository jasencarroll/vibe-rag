import pytest
import pytest_asyncio
import os

from vibe_rag.db.postgres import PostgresDB

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    pg = PostgresDB(url)
    await pg.connect()
    async with pg._pool.acquire() as conn:
        await conn.execute("DELETE FROM memories")
        await conn.execute("DELETE FROM sessions")
        await conn.execute("DELETE FROM docs")
    yield pg
    await pg.close()


async def test_remember_and_search(db):
    memory_id = await db.remember(
        content="Use JWT for auth", embedding=[0.1] * 1024,
        project_id="test-project", tags=["decision", "auth"],
    )
    assert memory_id is not None
    results = await db.search_memories(
        query_embedding=[0.1] * 1024, project_id="test-project",
        scope="memories", limit=5,
    )
    assert len(results) >= 1
    assert results[0]["content"] == "Use JWT for auth"


async def test_forget_by_id(db):
    memory_id = await db.remember(
        content="temporary fact", embedding=[0.1] * 1024,
        project_id=None, tags=[],
    )
    deleted = await db.forget_memory(memory_id=memory_id)
    assert deleted["content"] == "temporary fact"


async def test_upsert_session_chunks(db):
    chunks = [{"session_id": "sess-001", "chunk_index": 0,
               "content": "User asked about auth", "project_id": "test-project",
               "summary": "Discussion about authentication",
               "session_start": "2026-03-22T10:00:00Z", "session_end": "2026-03-22T10:30:00Z"}]
    await db.upsert_session_chunks(chunks, [[0.2] * 1024])
    results = await db.search_memories(query_embedding=[0.2] * 1024, scope="sessions", limit=5)
    assert len(results) >= 1


async def test_get_indexed_session_ids(db):
    chunks = [{"session_id": "sess-001", "chunk_index": 0, "content": "chunk",
               "project_id": None, "summary": None, "session_start": None, "session_end": None}]
    await db.upsert_session_chunks(chunks, [[0.1] * 1024])
    ids = await db.get_indexed_session_ids()
    assert "sess-001" in ids


async def test_forget_session(db):
    chunks = [{"session_id": "sess-delete", "chunk_index": 0, "content": "to be deleted",
               "project_id": None, "summary": None, "session_start": None, "session_end": None}]
    await db.upsert_session_chunks(chunks, [[0.1] * 1024])
    count = await db.forget_session("sess-delete")
    assert count == 1
