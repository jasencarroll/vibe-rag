from __future__ import annotations

import re
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def run_migrations(pool: asyncpg.Pool) -> list[int]:
    applied: list[int] = []
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
        applied_versions = {row["version"] for row in rows}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for f in migration_files:
            match = re.match(r"^(\d+)", f.name)
            if not match:
                continue
            version = int(match.group(1))
            if version in applied_versions:
                continue
            sql = f.read_text()
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)", version
            )
            applied.append(version)
    return applied
