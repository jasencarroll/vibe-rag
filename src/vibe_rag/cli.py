from __future__ import annotations
import shutil
from pathlib import Path

import click
from vibe_rag import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """vibe-rag — Memory and semantic code search for Mistral Vibe."""
    pass


@main.command()
@click.argument("name", required=False)
def init(name: str | None):
    """Create a new Vibe project with vibe-rag configured."""
    from importlib.resources import files as pkg_files

    templates_dir = Path(str(pkg_files("vibe_rag") / "templates"))
    bundle_dir = Path(str(pkg_files("vibe_rag") / "template_bundle"))

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

    # .vibe/config.toml
    vibe_dir = target / ".vibe"
    vibe_dir.mkdir(exist_ok=True)
    config_text = (bundle_dir / "vibe" / "config.toml").read_text()

    vibe_rag_bin = shutil.which("vibe-rag") or "vibe-rag"
    config_text = config_text.replace("__VIBE_RAG_BIN__", vibe_rag_bin)

    (vibe_dir / "config.toml").write_text(config_text)

    skills_template_dir = bundle_dir / "vibe" / "skills"
    if skills_template_dir.exists():
        shutil.copytree(skills_template_dir, vibe_dir / "skills", dirs_exist_ok=True)

    # .gitignore
    gitignore = target / ".gitignore"
    if gitignore.exists():
        text = gitignore.read_text()
        if ".vibe/index.db" not in text:
            gitignore.write_text(text.rstrip() + "\n.vibe/index.db\n")
    else:
        gitignore.write_text(".vibe/index.db\n")

    click.echo(f"\n  ✓ {name} created at {target}\n")
    click.echo(f"    AGENTS.md          — project coding rules")
    click.echo(f"    .vibe/config.toml  — vibe-rag MCP server")
    click.echo(f"\n  Next:")
    click.echo(f"    cd {target}")
    click.echo(f"    vibe")

@main.command()
def status():
    """Check memory status for current project."""
    from vibe_rag.db.sqlite import SqliteVecDB

    db_path = Path.cwd() / ".vibe" / "index.db"
    user_db_path = Path.home() / ".vibe" / "memory.db"
    click.echo(f"\n  vibe-rag {__version__}")
    click.echo(f"  DB: {db_path}\n")

    if db_path.exists():
        db = SqliteVecDB(db_path)
        db.initialize()
        click.echo(f"  Code chunks: {db.code_chunk_count()}")
        click.echo(f"  Doc chunks:  {db.doc_count()}")
        click.echo(f"  Project:     {db.memory_count()} memories")
        db.close()
    else:
        click.echo("  No index yet. Run vibe and use index_project.")

    if user_db_path.exists():
        user_db = SqliteVecDB(user_db_path)
        user_db.initialize()
        click.echo(f"  User:        {user_db.memory_count()} memories ({user_db_path})")
        user_db.close()
    else:
        click.echo(f"  User:        0 memories ({user_db_path})")
    click.echo()


@main.command()
def serve():
    """Start the MCP server (called by Vibe, not the user)."""
    from vibe_rag.server import run_server
    run_server()


if __name__ == "__main__":
    main()
