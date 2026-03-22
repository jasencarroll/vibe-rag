from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

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

            exists = await conn.fetchval("SELECT to_regclass('public.memories') IS NOT NULL")
            if not exists:
                await self._create_memories_table(conn)
                return

            columns = await conn.fetch(
                """
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'memories'
                """
            )
            column_types = {row["column_name"]: row["udt_name"] for row in columns}
            embedding_dim = await conn.fetchval(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public' AND c.relname = 'memories' AND a.attname = 'embedding'
                """
            )

            compatible = (
                column_types.get("id") == "uuid"
                and column_types.get("tags") == "_text"
                and embedding_dim == 1536
            )
            if not compatible:
                await self._archive_legacy_table(conn)
                await self._create_memories_table(conn)
                return

            await self._ensure_indexes(conn)

    async def _archive_legacy_table(self, conn: asyncpg.Connection) -> None:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        legacy_table = f"memories_legacy_{suffix}"
        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'memories'"
        )
        for row in indexes:
            old_name = row["indexname"]
            new_name = f"{old_name}_legacy_{suffix}"
            await conn.execute(f'ALTER INDEX IF EXISTS "{old_name}" RENAME TO "{new_name}"')
        await conn.execute(f'ALTER TABLE memories RENAME TO "{legacy_table}"')

    async def _create_memories_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute(
            """
            CREATE TABLE memories (
                id UUID PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                tags TEXT[] DEFAULT '{}',
                project_id TEXT,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        await self._ensure_indexes(conn)

    async def _ensure_indexes(self, conn: asyncpg.Connection) -> None:
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON memories USING hnsw (embedding vector_cosine_ops)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_project
            ON memories USING btree (project_id)
            """
        )

    def _normalize_tags(self, tags: str) -> list[str]:
        return [tag.strip() for tag in tags.split(",") if tag.strip()]

    async def remember(
        self,
        content: str,
        embedding: list[float],
        tags: str = "",
        project_id: str | None = None,
    ) -> str:
        memory_id = uuid4()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (id, content, embedding, tags, project_id)
                VALUES ($1, $2, $3::vector, $4::text[], $5)
                RETURNING id
                """,
                memory_id,
                content,
                str(embedding),
                self._normalize_tags(tags),
                project_id,
            )
            return str(row["id"])

    async def search_memories(
        self,
        query_embedding: list[float],
        limit: int = 10,
        project_id: str | None = None,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            if project_id:
                rows = await conn.fetch(
                    """
                    SELECT id, content, tags, project_id, created_at,
                           1 - (embedding <=> $1::vector) as score
                    FROM memories
                    WHERE project_id = $3 OR project_id IS NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(query_embedding),
                    limit,
                    project_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, content, tags, project_id, created_at,
                           1 - (embedding <=> $1::vector) as score
                    FROM memories
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(query_embedding),
                    limit,
                )
            return [dict(row) for row in rows]

    async def forget(self, memory_id: str | UUID) -> str | None:
        memory_uuid = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM memories WHERE id = $1 RETURNING content",
                memory_uuid,
            )
            return row["content"] if row else None

    async def memory_count(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM memories")
