from __future__ import annotations

from datetime import datetime, timezone
import json
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

            await self._ensure_columns(conn, column_types)
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
                memory_kind TEXT DEFAULT 'note',
                summary TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                source_session_id TEXT,
                source_message_id TEXT,
                supersedes UUID,
                superseded_by UUID,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        await self._ensure_indexes(conn)

    async def _ensure_columns(self, conn: asyncpg.Connection, column_types: dict[str, str]) -> None:
        if "memory_kind" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN memory_kind TEXT DEFAULT 'note'")
        if "summary" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN summary TEXT")
        if "metadata" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN metadata JSONB DEFAULT '{}'::jsonb")
        if "source_session_id" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN source_session_id TEXT")
        if "source_message_id" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN source_message_id TEXT")
        if "supersedes" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN supersedes UUID")
        if "superseded_by" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN superseded_by UUID")
        if "updated_at" not in column_types:
            await conn.execute("ALTER TABLE memories ADD COLUMN updated_at TIMESTAMPTZ DEFAULT now()")

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
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_superseded_by
            ON memories USING btree (superseded_by)
            """
        )

    def _normalize_tags(self, tags: str) -> list[str]:
        return [tag.strip() for tag in tags.split(",") if tag.strip()]

    def _normalize_row(self, row: asyncpg.Record | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        metadata = result.get("metadata")
        if isinstance(metadata, str):
            result["metadata"] = json.loads(metadata or "{}")
        elif metadata is None:
            result["metadata"] = {}
        return result

    async def remember(
        self,
        content: str,
        embedding: list[float],
        tags: str = "",
        project_id: str | None = None,
    ) -> str:
        return await self.remember_structured(
            summary=content[:200],
            content=content,
            embedding=embedding,
            tags=tags,
            project_id=project_id,
            memory_kind="note",
        )

    async def remember_structured(
        self,
        summary: str,
        content: str,
        embedding: list[float],
        tags: str = "",
        project_id: str | None = None,
        memory_kind: str = "note",
        metadata: dict | None = None,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        supersedes: str | UUID | None = None,
    ) -> str:
        memory_id = uuid4()
        supersedes_uuid = None if supersedes is None else (supersedes if isinstance(supersedes, UUID) else UUID(str(supersedes)))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (
                    id, content, embedding, tags, project_id, memory_kind, summary,
                    metadata, source_session_id, source_message_id, supersedes, updated_at
                )
                VALUES ($1, $2, $3::vector, $4::text[], $5, $6, $7, $8::jsonb, $9, $10, $11, now())
                RETURNING id
                """,
                memory_id,
                content,
                str(embedding),
                self._normalize_tags(tags),
                project_id,
                memory_kind,
                summary,
                json.dumps(metadata or {}),
                source_session_id,
                source_message_id,
                supersedes_uuid,
            )
            if supersedes_uuid is not None:
                await conn.execute(
                    "UPDATE memories SET superseded_by = $2, updated_at = now() WHERE id = $1",
                    supersedes_uuid,
                    memory_id,
                )
            return str(row["id"])

    async def search_memories(
        self,
        query_embedding: list[float],
        limit: int = 10,
        project_id: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            superseded_filter = "" if include_superseded else "AND superseded_by IS NULL"
            if project_id:
                rows = await conn.fetch(
                    f"""
                    SELECT id, content, tags, project_id, memory_kind, summary, metadata,
                           source_session_id, source_message_id, supersedes, superseded_by,
                           created_at, updated_at,
                           1 - (embedding <=> $1::vector) as score
                    FROM memories
                    WHERE (project_id = $3 OR project_id IS NULL) {superseded_filter}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(query_embedding),
                    limit,
                    project_id,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT id, content, tags, project_id, memory_kind, summary, metadata,
                           source_session_id, source_message_id, supersedes, superseded_by,
                           created_at, updated_at,
                           1 - (embedding <=> $1::vector) as score
                    FROM memories
                    WHERE TRUE {superseded_filter}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(query_embedding),
                    limit,
                )
            return [self._normalize_row(row) for row in rows]

    async def get_memory(self, memory_id: str | UUID) -> dict | None:
        memory_uuid = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, tags, project_id, memory_kind, summary, metadata,
                       source_session_id, source_message_id, supersedes, superseded_by,
                       created_at, updated_at
                FROM memories
                WHERE id = $1
                """,
                memory_uuid,
            )
            return self._normalize_row(row)

    async def get_memory_by_source(
        self, source_session_id: str, source_message_id: str
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, tags, project_id, memory_kind, summary, metadata,
                       source_session_id, source_message_id, supersedes, superseded_by,
                       created_at, updated_at
                FROM memories
                WHERE source_session_id = $1 AND source_message_id = $2
                LIMIT 1
                """,
                source_session_id,
                source_message_id,
            )
            return self._normalize_row(row)

    async def forget(self, memory_id: str | UUID) -> str | None:
        memory_uuid = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM memories WHERE id = $1 RETURNING content",
                memory_uuid,
            )
            return row["content"] if row else None

    async def memory_count(self, include_superseded: bool = False) -> int:
        async with self._pool.acquire() as conn:
            if include_superseded:
                return await conn.fetchval("SELECT COUNT(*) FROM memories")
            return await conn.fetchval("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL")
