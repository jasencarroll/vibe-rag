from __future__ import annotations
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
import tomllib

import click
from vibe_rag import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """vibe-rag — Semantic repo search and coding memory over MCP."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
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


def _status_label(ok: bool, warning: bool = False) -> str:
    if ok:
        return "ok"
    if warning:
        return "warn"
    return "fail"


def _read_toml_state(path: Path) -> tuple[dict | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        return tomllib.loads(path.read_text()), "ok"
    except OSError:
        return None, "unreadable"
    except tomllib.TOMLDecodeError:
        return None, "corrupt"


def _read_toml(path: Path) -> dict | None:
    parsed, _ = _read_toml_state(path)
    return parsed


def _resolve_command(command: str) -> tuple[bool, str]:
    if not command:
        return False, "missing command"
    if os.path.sep in command:
        path = Path(command).expanduser()
        return path.exists(), str(path)
    resolved = shutil.which(command)
    if resolved:
        return True, resolved
    return False, command


def _project_mcp_command_status(project_root: Path) -> dict:
    vibe_config, state = _read_toml_state(project_root / ".vibe" / "config.toml")
    if vibe_config is None:
        if state == "corrupt":
            return {"ok": False, "detail": "invalid TOML in .vibe/config.toml"}
        if state == "unreadable":
            return {"ok": False, "detail": "unreadable .vibe/config.toml"}
        return {"ok": False, "detail": "missing .vibe/config.toml"}

    servers = vibe_config.get("mcp_servers")
    if not isinstance(servers, list):
        return {"ok": False, "detail": "no [[mcp_servers]] entries in .vibe/config.toml"}

    memory_server = next((item for item in servers if isinstance(item, dict) and item.get("name") == "memory"), None)
    if not memory_server:
        return {"ok": False, "detail": "memory MCP server missing from .vibe/config.toml"}

    command = str(memory_server.get("command") or "")
    ok, resolved = _resolve_command(command)
    if not ok:
        return {"ok": False, "detail": f"MCP command not found: {resolved}"}
    return {"ok": True, "detail": f"{command} -> {resolved}"}


def _vibe_cli_status() -> dict:
    vibe_bin = shutil.which("vibe")
    if not vibe_bin:
        return {
            "ok": False,
            "warning": True,
            "detail": "Vibe CLI not found. Install the required mistral-vibe fork for first-class session bootstrap.",
        }

    try:
        result = subprocess.run(
            [vibe_bin, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError as exc:
        return {"ok": False, "warning": True, "detail": f"Vibe CLI failed to start: {exc}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "warning": True, "detail": "Vibe CLI version check timed out"}

    version_output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return {
            "ok": False,
            "warning": True,
            "detail": f"Vibe CLI found at {vibe_bin} but `vibe --version` failed",
        }
    return {"ok": True, "warning": False, "detail": f"{vibe_bin} ({version_output or 'version unknown'})"}


def _project_vibe_hook_status(project_root: Path) -> dict:
    vibe_config, state = _read_toml_state(project_root / ".vibe" / "config.toml")
    if vibe_config is None:
        if state == "corrupt":
            return {"ok": False, "warning": True, "detail": "invalid TOML in .vibe/config.toml"}
        if state == "unreadable":
            return {"ok": False, "warning": True, "detail": "unreadable .vibe/config.toml"}
        return {"ok": False, "warning": True, "detail": "missing .vibe/config.toml"}

    background = vibe_config.get("background_mcp_hook")
    session_memory = vibe_config.get("session_memory_hook")
    if not isinstance(background, dict) or not background.get("enabled"):
        return {
            "ok": False,
            "warning": True,
            "detail": "background_mcp_hook is not enabled in .vibe/config.toml",
        }
    if background.get("tool_name") != "memory_load_session_context":
        return {
            "ok": False,
            "warning": True,
            "detail": "background_mcp_hook.tool_name should be memory_load_session_context",
        }
    if not isinstance(session_memory, dict) or not session_memory.get("enabled"):
        return {
            "ok": False,
            "warning": True,
            "detail": "session_memory_hook is not enabled in .vibe/config.toml",
        }
    return {
        "ok": True,
        "warning": False,
        "detail": "background and session memory hooks are enabled",
    }


def _codex_hook_status(project_root: Path) -> dict:
    hooks_path = project_root / ".codex" / "hooks.json"
    if not hooks_path.exists():
        return {"ok": False, "detail": "missing .codex/hooks.json"}
    try:
        hooks = json.loads(hooks_path.read_text())
    except json.JSONDecodeError:
        return {"ok": False, "detail": "invalid JSON in .codex/hooks.json"}

    command = (
        ((((hooks.get("hooks") or {}).get("SessionStart") or [{}])[0].get("hooks") or [{}])[0].get("command"))
        or ""
    )
    if not command:
        return {"ok": False, "detail": "no SessionStart command in .codex/hooks.json"}

    argv = shlex.split(command)
    if not argv:
        return {"ok": False, "detail": "empty SessionStart command"}
    ok, resolved = _resolve_command(argv[0])
    if not ok:
        return {"ok": False, "detail": f"hook command not found: {argv[0]}"}

    try:
        result = subprocess.run(
            argv,
            cwd=project_root,
            input='{"source":"startup"}',
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError as exc:
        return {"ok": False, "detail": f"hook failed to start: {exc}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "hook timed out after 30s"}

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return {"ok": False, "detail": f"hook exited {result.returncode}: {stderr or 'no stderr'}"}

    stdout = (result.stdout or "").strip()
    if not stdout:
        return {"ok": False, "detail": "hook returned no output"}

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {"ok": False, "detail": "hook returned invalid JSON"}

    context = ((payload.get("hookSpecificOutput") or {}).get("additionalContext")) or ""
    system_message = payload.get("systemMessage")
    if system_message:
        return {"ok": False, "detail": system_message}
    if not context:
        return {"ok": False, "detail": "hook output missing additionalContext"}
    return {"ok": True, "detail": "hook returned session context"}


def _db_readable_status(db_path: Path, *, label: str) -> dict:
    from vibe_rag.db.sqlite import SqliteVecDB

    if not db_path.exists():
        return {"ok": False, "warning": True, "detail": f"{label} DB missing at {db_path}"}

    try:
        db = SqliteVecDB(db_path, embedding_dimensions=_embedding_dimensions())
        db.initialize()
        summary = f"{label} DB readable ({db_path})"
        db.close()
        return {"ok": True, "warning": False, "detail": summary}
    except Exception as exc:
        return {"ok": False, "warning": False, "detail": f"{label} DB unreadable: {exc}"}


def _provider_candidates() -> list[dict[str, str | bool]]:
    ollama_installed = bool(shutil.which("ollama"))
    mistral_key = bool(os.environ.get("MISTRAL_API_KEY", "").strip())
    openai_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    voyage_key = bool(os.environ.get("VOYAGE_API_KEY", "").strip())
    return [
        {
            "provider": "ollama",
            "available": ollama_installed,
            "detail": "local embeddings via Ollama" if ollama_installed else "install Ollama to use local embeddings",
        },
        {
            "provider": "mistral",
            "available": mistral_key,
            "detail": "MISTRAL_API_KEY is set" if mistral_key else "set MISTRAL_API_KEY",
        },
        {
            "provider": "openai",
            "available": openai_key,
            "detail": "OPENAI_API_KEY is set" if openai_key else "set OPENAI_API_KEY",
        },
        {
            "provider": "voyage",
            "available": voyage_key,
            "detail": "VOYAGE_API_KEY is set" if voyage_key else "set VOYAGE_API_KEY",
        },
    ]


def _recommended_provider() -> dict[str, str]:
    explicit = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "").strip().lower()
    if explicit:
        return {
            "provider": explicit,
            "reason": "explicitly configured by VIBE_RAG_EMBEDDING_PROVIDER",
        }

    for candidate in _provider_candidates():
        if candidate["provider"] == "ollama" and candidate["available"]:
            return {"provider": "ollama", "reason": "local embeddings are available"}
    for candidate in _provider_candidates():
        if candidate["available"]:
            return {
                "provider": str(candidate["provider"]),
                "reason": f"{candidate['provider']} credentials are already available",
            }
    return {
        "provider": "ollama",
        "reason": "default local-first provider; install Ollama or configure a hosted provider",
    }


def _provider_setup_hint(provider: str) -> str:
    if provider == "ollama":
        return 'run `vibe-rag setup-ollama` or set `VIBE_RAG_EMBEDDING_PROVIDER` to a hosted provider'
    if provider == "mistral":
        return 'export `MISTRAL_API_KEY=...` and optionally `VIBE_RAG_EMBEDDING_PROVIDER=mistral`'
    if provider == "openai":
        return 'export `OPENAI_API_KEY=...` and optionally `VIBE_RAG_EMBEDDING_PROVIDER=openai`'
    if provider == "voyage":
        return 'export `VOYAGE_API_KEY=...` and optionally `VIBE_RAG_EMBEDDING_PROVIDER=voyage`'
    return "configure a supported embedding provider"


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

    _initialize_git_repo(target)

    # .gitignore
    gitignore = target / ".gitignore"
    ignore_lines = [
        ".vibe/index.db",
        ".vibe/index.db-shm",
        ".vibe/index.db-wal",
        ".vibe/backups/",
    ]
    if gitignore.exists():
        text = gitignore.read_text()
        additions = [line for line in ignore_lines if line not in text]
        if additions:
            gitignore.write_text(text.rstrip() + "\n" + "\n".join(additions) + "\n")
    else:
        gitignore.write_text("\n".join(ignore_lines) + "\n")

    click.echo(f"\n  ✓ {name} created at {target}\n")
    click.echo("    AGENTS.md          — project coding rules")
    click.echo("    .vibe/config.toml  — Vibe MCP + hooks")
    click.echo("    .codex/            — Codex MCP + session-start hook")
    click.echo("    .claude/           — Claude Code session-start hook")
    click.echo("    .gemini/           — Gemini CLI MCP + session-start hook")
    click.echo("    .mcp.json          — Claude Code MCP server config")
    click.echo("\n  Client support:")
    click.echo("    Vibe is first-class and expects the mistral-vibe fork for session bootstrap.")
    click.echo("    Codex, Claude Code, and Gemini CLI scaffolding are experimental.")
    recommended = _recommended_provider()
    click.echo("\n  Embeddings:")
    click.echo(f"    Recommended provider: {recommended['provider']} ({recommended['reason']})")
    click.echo(f"    Next step: {_provider_setup_hint(recommended['provider'])}")
    click.echo("\n  Next:")
    click.echo(f"    cd {target}")
    click.echo("    vibe")

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
        click.echo("  No index yet. Run `vibe-rag reindex` or use `index_project` through your client.")

    if user_db_path.exists():
        user_db = SqliteVecDB(user_db_path, embedding_dimensions=embedding_dimensions)
        user_db.initialize()
        click.echo(f"  User:        {user_db.memory_count()} memories ({user_db_path})")
        user_db.close()
    else:
        click.echo(f"  User:        0 memories ({user_db_path})")
    click.echo()


@main.command("reindex")
@click.argument("paths", nargs=-1)
def reindex(paths: tuple[str, ...]):
    """Refresh the current project's code and docs index."""
    from vibe_rag.tools import index_project

    target_paths: list[str] | str = list(paths) if paths else "."
    click.echo()
    result = index_project(target_paths)
    if result.get("ok"):
        click.echo(result.get("summary") or "Index complete")
    else:
        error = result.get("error") or {}
        click.echo(f"Error: {error.get('message') or 'indexing failed'}")
    click.echo()


@main.command()
@click.option("--fix", is_flag=True, help="Run provider-specific setup helpers when possible.")
def doctor(fix: bool):
    """Check local vibe-rag setup and embedding provider health."""
    from vibe_rag.indexing.embedder import embedding_provider_status
    from vibe_rag.server import _ensure_project_id, _get_db, _get_embedder
    from vibe_rag.tools import _codex_trust_status, _stale_state, _vibe_trust_status

    project_root = Path.cwd().resolve()
    project_db_path = Path(os.environ.get("VIBE_RAG_DB", project_root / ".vibe" / "index.db")).expanduser()
    user_db_path = Path(os.environ.get("VIBE_RAG_USER_DB", Path.home() / ".vibe" / "memory.db")).expanduser()
    provider = embedding_provider_status()
    project_id = _ensure_project_id()
    mcp_status = _project_mcp_command_status(project_root)
    vibe_cli_status = _vibe_cli_status()
    vibe_hook_status = _project_vibe_hook_status(project_root)
    hook_status = _codex_hook_status(project_root)
    project_db_status = _db_readable_status(project_db_path, label="Project")
    user_db_status = _db_readable_status(user_db_path, label="User")
    vibe_trust = _vibe_trust_status(project_root)
    codex_trust = _codex_trust_status(project_root)
    recommended = _recommended_provider()
    candidates = _provider_candidates()

    provider_detail = str(provider["detail"])
    provider_ok = bool(provider["ok"])
    if provider_ok:
        try:
            _get_embedder().embed_text_sync(["doctor healthcheck"])
        except Exception as exc:
            provider_ok = False
            provider_detail = f"Embedding failed: {exc}"

    stale_state = {"is_stale": False, "warnings": []}
    if project_db_status["ok"]:
        try:
            stale_state = _stale_state(_get_db(), project_root, project_id)
        except Exception as exc:
            stale_state = {
                "is_stale": True,
                "warnings": [{"kind": "stale_check_failed", "detail": f"stale-state check failed: {exc}"}],
            }

    click.echo(f"\n  vibe-rag {__version__}")
    click.echo(f"  Project root: {project_root}")
    click.echo(f"  Project id:   {project_id}")
    click.echo(f"  Project DB:   {project_db_path}")
    click.echo(f"  User DB:      {user_db_path}")
    click.echo(f"  Provider:     {provider['provider']}")
    click.echo(f"  Model:        {provider['model'] or 'unset'}")
    click.echo(f"  Recommended:  {recommended['provider']} ({recommended['reason']})")
    click.echo()
    click.echo(f"  [{_status_label(mcp_status['ok'])}] MCP command     {mcp_status['detail']}")
    click.echo(
        f"  [{_status_label(vibe_cli_status['ok'], vibe_cli_status.get('warning', False))}] Vibe CLI        {vibe_cli_status['detail']}"
    )
    click.echo(
        f"  [{_status_label(vibe_hook_status['ok'], vibe_hook_status.get('warning', False))}] Vibe hooks      {vibe_hook_status['detail']}"
    )
    click.echo(f"  [{_status_label(hook_status['ok'])}] SessionStart   {hook_status['detail']}")
    click.echo(
        f"  [{_status_label(project_db_status['ok'], project_db_status.get('warning', False))}] Project DB      {project_db_status['detail']}"
    )
    click.echo(
        f"  [{_status_label(user_db_status['ok'], user_db_status.get('warning', False))}] User DB         {user_db_status['detail']}"
    )
    click.echo(f"  [{_status_label(provider_ok)}] Embedding       {provider_detail}")
    click.echo(f"  [{_status_label(vibe_trust['status'] == 'ok', vibe_trust['status'] == 'warn')}] Vibe trust      {vibe_trust['detail']}")
    click.echo(f"  [{_status_label(codex_trust['status'] == 'ok', codex_trust['status'] == 'warn')}] Codex trust     {codex_trust['detail']}")

    if stale_state.get("warnings"):
        click.echo("  [warn] Stale state     " + stale_state["warnings"][0]["detail"])
        for warning in stale_state["warnings"][1:]:
            click.echo(f"                    {warning['detail']}")
        click.echo("  Suggested stale fix: vibe-rag reindex")
    else:
        click.echo("  [ok] Stale state     no stale index warnings")

    click.echo("  Provider options:")
    for candidate in candidates:
        available = bool(candidate["available"])
        click.echo(f"  [{_status_label(available, not available)}] {candidate['provider']:<15} {candidate['detail']}")

    if not provider_ok or provider["provider"] != recommended["provider"]:
        click.echo(f"  Suggested next step: {_provider_setup_hint(recommended['provider'])}")
    if not vibe_cli_status["ok"]:
        click.echo(
            "  Vibe first-class path: install the required fork with `uv tool install git+https://github.com/jasencarroll/mistral-vibe.git`"
        )

    if provider["provider"] == "ollama" and not provider_ok:
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
    click.echo('    VIBE_RAG_EMBEDDING_PROVIDER = "ollama"')
    click.echo(f'    VIBE_RAG_EMBEDDING_MODEL = "{model}"')
    click.echo('    VIBE_RAG_EMBEDDING_DIMENSIONS = "1024"')
    click.echo()


@main.command("hook-session-start")
@click.option("--format", "target_format", type=click.Choice(["codex", "claude", "gemini", "vibe"]), required=True)
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
