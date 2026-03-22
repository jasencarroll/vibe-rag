from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

import click
from vibe_memory import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """vibe-memory — Memory and semantic code search for Mistral Vibe."""
    pass


@main.command()
@click.argument("name", required=False)
def init(name: str | None):
    """Create a new Vibe project with memory, skills, and agents."""
    from importlib.resources import files as pkg_files

    templates_dir = Path(str(pkg_files("vibe_memory") / "templates"))

    if not name:
        name = click.prompt("\n  Project name")
    if not name:
        click.echo("  Error: project name required")
        return

    target = Path.cwd() / name

    if target.exists():
        click.echo(f"\n  {target} already exists.")
        if (target / ".vibe").exists():
            if click.confirm("  Re-stamp .vibe config?", default=False):
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

    target.mkdir(parents=True, exist_ok=True)

    # AGENTS.md
    shutil.copy2(templates_dir / "AGENTS.md", target / "AGENTS.md")

    # .vibe/config.toml — inject env vars and binary path
    vibe_dir = target / ".vibe"
    vibe_dir.mkdir(exist_ok=True)
    config_text = (templates_dir / ".vibe" / "config.toml").read_text()

    console_key = os.environ.get("MISTRAL_CONSOLE_KEY", os.environ.get("MISTRAL_API_KEY", ""))
    config_text = config_text.replace("__DATABASE_URL__", os.environ.get("DATABASE_URL", ""))
    config_text = config_text.replace("__MISTRAL_CONSOLE_KEY__", console_key)

    vibe_memory_bin = shutil.which("vibe-memory") or str(Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "vibe-memory")
    config_text = config_text.replace("__VIBE_MEMORY_BIN__", vibe_memory_bin)

    (vibe_dir / "config.toml").write_text(config_text)

    # Skills
    skills_src = templates_dir / ".vibe" / "skills"
    if skills_src.exists():
        shutil.copytree(skills_src, vibe_dir / "skills", dirs_exist_ok=True)

    # Global agents
    global_agents = Path.home() / ".vibe" / "agents"
    agents_src = templates_dir / "agents"
    if agents_src.exists() and (not global_agents.exists() or not any(global_agents.iterdir())):
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

    # git init
    if not (target / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=target, capture_output=True)

    skill_count = sum(1 for _ in (vibe_dir / "skills").rglob("SKILL.md")) if (vibe_dir / "skills").exists() else 0
    click.echo(f"\n  ✓ {name} created at {target}\n")
    click.echo(f"    AGENTS.md            — coding rules")
    click.echo(f"    .vibe/config.toml    — Playwright + Context7 + Memory MCP")
    click.echo(f"    .vibe/skills/        — {skill_count} skills")
    click.echo(f"\n  Next:")
    click.echo(f"    cd {target}")
    click.echo(f"    vibe --agent builder\n")


@main.command()
def status():
    """Check memory status for current project."""
    from vibe_memory.db.sqlite import SqliteVecDB

    db_path = Path.cwd() / ".vibe" / "index.db"
    click.echo(f"\n  vibe-memory {__version__}")
    click.echo(f"  DB: {db_path}\n")

    if db_path.exists():
        db = SqliteVecDB(db_path)
        db.initialize()
        click.echo(f"  Code chunks: {db.code_chunk_count()}")
        click.echo(f"  Memories:    {db.memory_count()}")
        db.close()
    else:
        click.echo("  No index yet. Run vibe and use index_project.")
    click.echo()


@main.command()
def serve():
    """Start the MCP server (called by Vibe, not the user)."""
    from vibe_memory.server import run_server
    run_server()
