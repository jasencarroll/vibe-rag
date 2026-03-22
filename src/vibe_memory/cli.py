from __future__ import annotations
import asyncio
import os
import sys
import click
from vibe_memory import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """vibe-memory — Centralized memory for Mistral Vibe."""
    pass


@main.command()
def setup():
    """Set up the memory database (Neon or BYO Postgres)."""
    from vibe_memory.setup.neon import neonctl_available
    from vibe_memory.setup.interactive import setup_with_neon, setup_with_url

    click.echo(f"\n  vibe-memory {__version__}\n")
    existing = os.environ.get("DATABASE_URL")
    if existing:
        click.echo("  DATABASE_URL already set.")
        if not click.confirm("  Overwrite?", default=False):
            return

    click.echo("  No DATABASE_URL found. How do you want to connect?\n")
    click.echo("  1. Create a free database on Neon (recommended)")
    click.echo("  2. I have a Postgres URL with pgvector\n")
    choice = click.prompt("  Choose", type=click.IntRange(1, 2))

    if choice == 1:
        if not neonctl_available():
            click.echo("\n  neonctl not found. Install it first:")
            click.echo("    npm install -g neonctl\n")
            click.echo("  Or choose option 2 to use your own Postgres URL.")
            return
        setup_with_neon()
    else:
        url = click.prompt("  Paste your DATABASE_URL")
        setup_with_url(url)

    click.echo("\n  Add this to your .vibe/config.toml:\n")
    click.echo('  [[mcp_servers]]')
    click.echo('  name = "memory"')
    click.echo('  transport = "stdio"')
    click.echo('  command = "uvx"')
    click.echo('  args = ["vibe-memory", "serve"]\n')
    click.echo("  Done. Start vibe and your memory is live.\n")


@main.command()
def status():
    """Check database connection and show stats."""
    from vibe_memory.config import load_config
    from vibe_memory.db.postgres import PostgresDB
    from vibe_memory.db.sqlite import SqliteVecDB
    from pathlib import Path

    config = load_config()
    click.echo(f"\n  vibe-memory {__version__}")
    click.echo(f"  Project: {config.project_id}\n")

    if config.database_url:
        try:
            pg = PostgresDB(config.database_url)
            stats = asyncio.run(_async_status(pg))
            click.echo("  pgvector: ✓ connected")
            click.echo(f"    Memories: {stats['memories']}")
            click.echo(f"    Sessions: {stats['sessions']}")
            click.echo(f"    Docs:     {stats['docs']}")
        except Exception as e:
            click.echo(f"  pgvector: ✗ {e}")
    else:
        click.echo("  pgvector: ✗ no DATABASE_URL")

    index_path = Path.cwd() / ".vibe" / "index.db"
    if index_path.exists():
        db = SqliteVecDB(index_path)
        db.initialize()
        click.echo(f"  sqlite-vec: ✓ {db.chunk_count()} code chunks")
        db.close()
    else:
        click.echo("  sqlite-vec: ✗ no index (run index_project)")
    click.echo()


async def _async_status(pg):
    await pg.connect()
    stats = await pg.get_stats()
    await pg.close()
    return stats


@main.command()
@click.argument("project")
def purge(project: str):
    """Delete all data for a project."""
    if not click.confirm(f"  Delete all memories, sessions, and docs for '{project}'?"):
        return
    click.echo(f"  Purging {project}... (deferred to 0.0.2)")


@main.command()
def migrate():
    """Run pending database migrations."""
    from vibe_memory.config import load_config
    from vibe_memory.db.postgres import PostgresDB

    config = load_config()
    if not config.database_url:
        click.echo("  No DATABASE_URL set. Run `vibe-memory setup` first.")
        return

    pg = PostgresDB(config.database_url)
    async def _migrate():
        await pg.connect()
        await pg.close()
    asyncio.run(_migrate())
    click.echo("  ✓ Migrations applied")


@main.command()
def serve():
    """Start the MCP server (called by Vibe, not the user)."""
    from vibe_memory.server import run_server
    run_server()
