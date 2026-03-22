from __future__ import annotations

from datetime import datetime, timezone

import asyncpg
from vibe_memory.db.migrations import run_migrations


def _parse_ts(val: str | datetime | None) -> datetime | None:
    if val is None or isinstance(val, datetime):
        return val
    return datetime.fromisoformat(val.replace("Z", "+00:00"))


class PostgresDB:
    def __init__(self, database_url: str):
        self._url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._url, min_size=1, max_size=5)
        await run_migrations(self._pool)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def remember(self, content: str, embedding: list[float], project_id: str | None, tags: list[str]) -> str:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO memories (content, embedding, project_id, tags)
                   VALUES ($1, $2::vector, $3, $4)
                   RETURNING id""",
                content, str(embedding), project_id, tags,
            )
            return str(row["id"])

    async def search_memories(self, query_embedding: list[float], scope: str = "all",
                               project_id: str | None = None, project_only: bool = False, limit: int = 10) -> list[dict]:
        results = []
        vec_str = str(query_embedding)

        if scope in ("all", "memories"):
            where = "WHERE 1=1"
            params = [vec_str, limit]
            if project_only and project_id:
                where += f" AND (project_id = ${len(params) + 1})"
                params.append(project_id)
            elif project_id:
                where += f" AND (project_id = ${len(params) + 1} OR project_id IS NULL)"
                params.append(project_id)

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT id, content, project_id, tags, created_at,
                               1 - (embedding <=> $1::vector) as score,
                               'memory' as source_type
                        FROM memories {where}
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2""",
                    *params,
                )
                results.extend([dict(r) for r in rows])

        if scope in ("all", "sessions"):
            where = "WHERE 1=1"
            params = [vec_str, limit]
            if project_only and project_id:
                where += f" AND (project_id = ${len(params) + 1})"
                params.append(project_id)
            elif project_id:
                where += f" AND (project_id = ${len(params) + 1} OR project_id IS NULL)"
                params.append(project_id)

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT id, session_id, content, project_id, summary,
                               session_start, session_end, created_at,
                               1 - (embedding <=> $1::vector) as score,
                               'session' as source_type
                        FROM sessions {where}
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2""",
                    *params,
                )
                results.extend([dict(r) for r in rows])

        if scope in ("all", "docs"):
            where = "WHERE 1=1"
            params = [vec_str, limit]
            if project_only and project_id:
                where += f" AND (project_id = ${len(params) + 1})"
                params.append(project_id)
            elif project_id:
                where += f" AND (project_id = ${len(params) + 1} OR project_id IS NULL)"
                params.append(project_id)

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT id, source, content, project_id, created_at,
                               1 - (embedding <=> $1::vector) as score,
                               'doc' as source_type
                        FROM docs {where}
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2""",
                    *params,
                )
                results.extend([dict(r) for r in rows])

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[:limit]

    async def forget_memory(self, memory_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM memories WHERE id = $1::uuid RETURNING content, id",
                memory_id,
            )
            return dict(row) if row else None

    async def find_closest_memory(self, query_embedding: list[float]) -> dict | None:
        async with self._pool.acquire() as conn:
            vec_str = str(query_embedding)
            row = await conn.fetchrow(
                """SELECT id, content, 1 - (embedding <=> $1::vector) as score
                   FROM memories
                   ORDER BY embedding <=> $1::vector
                   LIMIT 1""",
                vec_str,
            )
            return dict(row) if row else None

    async def forget_session(self, session_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE session_id = $1", session_id
            )
            return int(result.split()[-1])

    async def upsert_session_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        async with self._pool.acquire() as conn:
            for chunk, embedding in zip(chunks, embeddings):
                await conn.execute(
                    """INSERT INTO sessions
                       (session_id, chunk_index, content, embedding, project_id,
                        summary, session_start, session_end)
                       VALUES ($1, $2, $3, $4::vector, $5, $6, $7::timestamptz, $8::timestamptz)
                       ON CONFLICT (session_id, chunk_index) DO UPDATE SET
                         content = EXCLUDED.content,
                         embedding = EXCLUDED.embedding""",
                    chunk["session_id"], chunk["chunk_index"], chunk["content"],
                    str(embedding), chunk["project_id"], chunk["summary"],
                    _parse_ts(chunk["session_start"]), _parse_ts(chunk["session_end"]),
                )

    async def upsert_doc_chunks(self, source: str, chunks: list[str],
                                 embeddings: list[list[float]], project_id: str | None = None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM docs WHERE source = $1", source)
            for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
                await conn.execute(
                    """INSERT INTO docs (source, chunk_index, content, embedding, project_id)
                       VALUES ($1, $2, $3, $4::vector, $5)""",
                    source, i, content, str(embedding), project_id,
                )

    async def get_indexed_session_ids(self) -> set[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT session_id FROM sessions")
            return {row["session_id"] for row in rows}

    async def get_stats(self) -> dict:
        async with self._pool.acquire() as conn:
            memories = await conn.fetchval("SELECT COUNT(*) FROM memories")
            sessions = await conn.fetchval("SELECT COUNT(DISTINCT session_id) FROM sessions")
            docs = await conn.fetchval("SELECT COUNT(DISTINCT source) FROM docs")
            return {"memories": memories, "sessions": sessions, "docs": docs}
