from __future__ import annotations
import os
from pathlib import Path
import asyncio
import asyncpg
from vibe_rag.db.migrations import run_migrations
from vibe_rag.setup.neon import (
    neonctl_available, neon_auth, neon_create_project,
    neon_get_connection_string, neon_enable_pgvector,
)

ENV_PATH = Path.home() / ".vibe" / ".env"


def _save_database_url(url: str) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if ENV_PATH.exists():
        existing = ENV_PATH.read_text()
    lines = existing.splitlines()
    new_lines = [l for l in lines if not l.startswith("DATABASE_URL=")]
    new_lines.append(f"DATABASE_URL='{url}'")
    ENV_PATH.write_text("\n".join(new_lines) + "\n")
    os.chmod(ENV_PATH, 0o600)


async def _run_migrations_with_url(url: str) -> list[int]:
    pool = await asyncpg.create_pool(url, min_size=1, max_size=2)
    try:
        return await run_migrations(pool)
    finally:
        await pool.close()


def setup_with_neon() -> str:
    print("\n  Opening Neon auth in browser...")
    if not neon_auth():
        raise RuntimeError("Neon authentication failed")
    print("  ✓ Authenticated")
    print("  ✓ Creating project...")
    project = neon_create_project()
    project_id = project.get("project", {}).get("id", project.get("id", ""))
    print("  ✓ Created project: vibe-rag")
    conn_str = neon_get_connection_string(project_id)
    neon_enable_pgvector(conn_str)
    print("  ✓ Enabled pgvector extension")
    print("  ✓ Running migrations...")
    applied = asyncio.run(_run_migrations_with_url(conn_str))
    print(f"  ✓ Applied {len(applied)} migrations")
    _save_database_url(conn_str)
    print(f"  ✓ Saved DATABASE_URL to {ENV_PATH}")
    return conn_str


def setup_with_url(url: str) -> str:
    print("\n  Testing connection...")
    try:
        applied = asyncio.run(_run_migrations_with_url(url))
        print("  ✓ Connected")
        print(f"  ✓ Applied {len(applied)} migrations")
    except Exception as e:
        raise RuntimeError(f"Connection failed: {e}")
    _save_database_url(url)
    print(f"  ✓ Saved DATABASE_URL to {ENV_PATH}")
    return url
