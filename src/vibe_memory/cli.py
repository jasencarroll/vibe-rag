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
@click.argument("name", required=False)
def init(name: str | None):
    """Create a new Vibe project with memory, skills, and agents."""
    import shutil
    import subprocess
    from pathlib import Path
    from importlib.resources import files as pkg_files

    templates_dir = Path(str(pkg_files("vibe_memory") / "templates"))
    projects_dir = Path.cwd()

    # Ask for name if not provided
    if not name:
        name = click.prompt("\n  Project name")
    if not name:
        click.echo("  Error: project name required")
        return

    target = projects_dir / name

    # Handle existing directory
    if target.exists():
        click.echo(f"\n  {target} already exists.")
        if (target / ".vibe").exists():
            if click.confirm("  Re-stamp .vibe config? (overwrites .vibe/ and AGENTS.md)", default=False):
                shutil.rmtree(target / ".vibe", ignore_errors=True)
                (target / "AGENTS.md").unlink(missing_ok=True)
            else:
                click.echo("  Aborted.")
                return
        elif click.confirm("  Delete and recreate?", default=False):
            shutil.rmtree(target)
        else:
            click.echo("  Aborted.")
            return

    # Create project
    target.mkdir(parents=True, exist_ok=True)

    # Stamp AGENTS.md
    shutil.copy2(templates_dir / "AGENTS.md", target / "AGENTS.md")

    # Stamp .vibe/config.toml
    vibe_dir = target / ".vibe"
    vibe_dir.mkdir(exist_ok=True)
    config_src = templates_dir / ".vibe" / "config.toml"
    config_dst = vibe_dir / "config.toml"
    config_text = config_src.read_text()

    # Inject env vars (Vibe doesn't expand ${VAR} in MCP env blocks)
    console_key = os.environ.get("MISTRAL_CONSOLE_KEY", os.environ.get("MISTRAL_API_KEY", ""))
    config_text = config_text.replace("__DATABASE_URL__", os.environ.get("DATABASE_URL", ""))
    config_text = config_text.replace("__MISTRAL_CONSOLE_KEY__", console_key)

    # Resolve vibe-memory binary path
    vibe_memory_bin = shutil.which("vibe-memory") or str(Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "vibe-memory")
    config_text = config_text.replace("__VIBE_MEMORY_BIN__", vibe_memory_bin)

    config_dst.write_text(config_text)

    # Stamp skills
    skills_src = templates_dir / ".vibe" / "skills"
    if skills_src.exists():
        skills_dst = vibe_dir / "skills"
        shutil.copytree(skills_src, skills_dst, dirs_exist_ok=True)

    # Install global agents (once)
    global_agents = Path.home() / ".vibe" / "agents"
    agents_src = templates_dir / "agents"
    if agents_src.exists():
        if not global_agents.exists() or not any(global_agents.iterdir()):
            global_agents.mkdir(parents=True, exist_ok=True)
            for f in agents_src.glob("*.toml"):
                shutil.copy2(f, global_agents / f.name)
            click.echo("  Installed global agents to ~/.vibe/agents/")

    # .gitignore
    gitignore = target / ".gitignore"
    if gitignore.exists():
        text = gitignore.read_text()
        if ".vibe/index.db" not in text:
            gitignore.write_text(text.rstrip() + "\n.vibe/index.db\n")
    else:
        gitignore.write_text(".vibe/index.db\n")

    # Init git
    if not (target / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=target, capture_output=True)

    # Summary
    skill_count = sum(1 for _ in (vibe_dir / "skills").rglob("SKILL.md")) if (vibe_dir / "skills").exists() else 0
    click.echo(f"\n  ✓ {name} created at {target}\n")
    click.echo(f"    AGENTS.md            — coding rules")
    click.echo(f"    .vibe/config.toml    — Playwright + Context7 + Memory MCP")
    click.echo(f"    .vibe/skills/        — {skill_count} skills")
    click.echo(f"    .gitignore           — excludes .vibe/index.db")

    if os.environ.get("DATABASE_URL"):
        click.echo(f"    vibe-memory          — connected")
    else:
        click.echo(f"\n    Run `vibe-memory setup` to connect a database")

    click.echo(f"\n  Next:")
    click.echo(f"    cd {target}")
    click.echo(f"    vibe --agent builder\n")


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
