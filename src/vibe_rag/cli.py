from __future__ import annotations
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

import click
from vibe_rag import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """vibe-rag — Memory and semantic code search for Mistral Vibe."""
    pass


def _embedding_dimensions() -> int:
    raw = os.environ.get("VIBE_RAG_EMBEDDING_DIMENSIONS", "1024").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise click.ClickException("VIBE_RAG_EMBEDDING_DIMENSIONS must be an integer") from exc
    if value <= 0:
        raise click.ClickException("VIBE_RAG_EMBEDDING_DIMENSIONS must be positive")
    return value


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

    vibe_rag_bin = shutil.which("vibe-rag") or "vibe-rag"
    bundle_mappings = {
        "vibe": ".vibe",
        "codex": ".codex",
        "claude": ".claude",
        "gemini": ".gemini",
    }

    for bundle_name, target_name in bundle_mappings.items():
        source_dir = bundle_dir / bundle_name
        if source_dir.exists():
            shutil.copytree(source_dir, target / target_name, dirs_exist_ok=True)

    mcp_json_template = bundle_dir / "mcp.json"
    if mcp_json_template.exists():
        shutil.copy2(mcp_json_template, target / ".mcp.json")

    _replace_placeholder_in_files(_generated_config_files(target), "__VIBE_RAG_BIN__", vibe_rag_bin)
    _initialize_git_repo(target)

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
    click.echo(f"    .vibe/config.toml  — Vibe MCP + hooks")
    click.echo(f"    .codex/            — Codex MCP + session-start hook")
    click.echo(f"    .claude/           — Claude Code session-start hook")
    click.echo(f"    .gemini/           — Gemini CLI MCP + session-start hook")
    click.echo(f"    .mcp.json          — Claude Code MCP server config")
    click.echo(f"\n  Next:")
    click.echo(f"    cd {target}")
    click.echo(f"    vibe")


def _generated_config_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_dir in (".vibe", ".codex", ".claude", ".gemini"):
        config_dir = root / relative_dir
        if not config_dir.exists():
            continue
        files.extend(path for path in config_dir.rglob("*") if path.is_file())

    mcp_json = root / ".mcp.json"
    if mcp_json.exists():
        files.append(mcp_json)

    return files


def _replace_placeholder_in_files(paths: list[Path], placeholder: str, value: str) -> None:
    for path in paths:
        text = path.read_text()
        if placeholder in text:
            path.write_text(text.replace(placeholder, value))


def _initialize_git_repo(target: Path) -> None:
    if (target / ".git").exists():
        return

    git_bin = shutil.which("git")
    if not git_bin:
        return

    subprocess.run(
        [git_bin, "init"],
        cwd=target,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

@main.command()
def status():
    """Check memory status for current project."""
    from vibe_rag.db.sqlite import SqliteVecDB

    db_path = Path(os.environ.get("VIBE_RAG_DB", Path.cwd() / ".vibe" / "index.db")).expanduser()
    user_db_path = Path(os.environ.get("VIBE_RAG_USER_DB", Path.home() / ".vibe" / "memory.db")).expanduser()
    embedding_dimensions = _embedding_dimensions()
    click.echo(f"\n  vibe-rag {__version__}")
    click.echo(f"  DB: {db_path}\n")

    if db_path.exists():
        db = SqliteVecDB(db_path, embedding_dimensions=embedding_dimensions)
        db.initialize()
        click.echo(f"  Code chunks: {db.code_chunk_count()}")
        click.echo(f"  Doc chunks:  {db.doc_count()}")
        click.echo(f"  Project:     {db.memory_count()} memories")
        db.close()
    else:
        click.echo("  No index yet. Run vibe and use index_project.")

    if user_db_path.exists():
        user_db = SqliteVecDB(user_db_path, embedding_dimensions=embedding_dimensions)
        user_db.initialize()
        click.echo(f"  User:        {user_db.memory_count()} memories ({user_db_path})")
        user_db.close()
    else:
        click.echo(f"  User:        0 memories ({user_db_path})")
    click.echo()


@main.command()
@click.option("--fix", is_flag=True, help="Run provider-specific setup helpers when possible.")
def doctor(fix: bool):
    """Check local vibe-rag setup and embedding provider health."""
    from vibe_rag.indexing.embedder import embedding_provider_status

    project_db_path = Path.cwd() / ".vibe" / "index.db"
    user_db_path = Path(os.environ.get("VIBE_RAG_USER_DB", Path.home() / ".vibe" / "memory.db")).expanduser()
    provider = embedding_provider_status()

    click.echo(f"\n  vibe-rag {__version__}")
    click.echo(f"  Project DB:  {project_db_path}")
    click.echo(f"  User DB:     {user_db_path}")
    click.echo(f"  Provider:    {provider['provider']}")
    click.echo(f"  Model:       {provider['model'] or 'unset'}")
    click.echo(f"  Status:      {provider['detail']}")

    if provider["provider"] == "ollama" and not provider["ok"]:
        click.echo("\n  Ollama fast path:")
        click.echo("    ollama serve")
        if provider["model"]:
            click.echo(f"    ollama pull {provider['model']}")
        if fix:
            ctx = click.get_current_context()
            ctx.invoke(setup_ollama, model=provider["model"] or "")
    click.echo()


def _wait_for_ollama(host: str, timeout_seconds: float = 10.0) -> bool:
    import httpx

    deadline = time.time() + timeout_seconds
    url = f"{host.rstrip('/')}/api/version"
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def _start_ollama_if_needed() -> str:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise click.ClickException("Ollama is not installed or not on PATH.")

    host = (
        os.environ.get("VIBE_RAG_OLLAMA_HOST", "").strip()
        or os.environ.get("OLLAMA_HOST", "").strip()
        or "http://localhost:11434"
    )
    if _wait_for_ollama(host, timeout_seconds=1.0):
        return host

    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if _wait_for_ollama(host):
        return host
    raise click.ClickException(f"Ollama did not become ready at {host}.")


@main.command("setup-ollama")
@click.option("--model", default="qwen3-embedding:0.6b", show_default=True, help="Embedding model to pull.")
def setup_ollama(model: str):
    """Start Ollama if needed and pull the default embedding model."""
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise click.ClickException("Ollama is not installed or not on PATH.")

    host = _start_ollama_if_needed()
    click.echo(f"\n  Ollama ready at {host}")
    click.echo(f"  Pulling {model} ...")

    result = subprocess.run([ollama_bin, "pull", model], check=False)
    if result.returncode != 0:
        raise click.ClickException(f"ollama pull {model} failed with exit code {result.returncode}")

    click.echo(f"  Pulled {model}")
    click.echo("\n  MCP env:")
    click.echo(f'    VIBE_RAG_EMBEDDING_PROVIDER = "ollama"')
    click.echo(f'    VIBE_RAG_EMBEDDING_MODEL = "{model}"')
    click.echo('    VIBE_RAG_EMBEDDING_DIMENSIONS = "1024"')
    click.echo()


@main.command("hook-session-start")
@click.option("--format", "target_format", type=click.Choice(["codex", "claude", "gemini"]), required=True)
def hook_session_start(target_format: str):
    """Render SessionStart hook output for supported agent CLIs."""
    from vibe_rag.hook_bridge import render_session_start_hook_json

    raw_input = sys.stdin.read()
    click.echo(render_session_start_hook_json(target_format, raw_input))


@main.command()
def serve():
    """Start the MCP server (called by Vibe, not the user)."""
    from vibe_rag.server import run_server
    run_server()


if __name__ == "__main__":
    main()
