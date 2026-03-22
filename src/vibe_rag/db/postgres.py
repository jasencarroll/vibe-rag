from __future__ import annotations

import asyncpg


class PostgresDB:
    def __init__(self, database_url: str):
        self._url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._url, min_size=1, max_size=5)
        await self._run_migrations()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _run_migrations(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    tags TEXT DEFAULT '',
                    project_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING hnsw (embedding vector_cosine_ops)
            """)

    async def remember(self, content: str, embedding: list[float], tags: str = "", project_id: str | None = None) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO memories (content, embedding, tags, project_id) VALUES ($1, $2::vector, $3, $4) RETURNING id",
                content, str(embedding), tags, project_id,
            )
            return row["id"]

    async def search_memories(self, query_embedding: list[float], limit: int = 10, project_id: str | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            if project_id:
                rows = await conn.fetch(
                    """SELECT id, content, tags, project_id, created_at,
                              1 - (embedding <=> $1::vector) as score
                       FROM memories
                       WHERE project_id = $3 OR project_id IS NULL
                       ORDER BY embedding <=> $1::vector
                       LIMIT $2""",
                    str(query_embedding), limit, project_id,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, content, tags, project_id, created_at,
                              1 - (embedding <=> $1::vector) as score
                       FROM memories
                       ORDER BY embedding <=> $1::vector
                       LIMIT $2""",
                    str(query_embedding), limit,
                )
            return [dict(r) for r in rows]

    async def forget(self, memory_id: int) -> str | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM memories WHERE id = $1 RETURNING content", memory_id,
            )
            return row["content"] if row else None

    async def memory_count(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM memories")
