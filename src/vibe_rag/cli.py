"""Click CLI for vibe-rag.

Commands:
    init               Scaffold a new project with MCP config for all supported agent CLIs.
    status             Show index, memory, and health summary for the current project.
    reindex            Re-index code and docs (incremental by default, --full for rebuild).
    reset-index        Alias for ``reindex --full``.
    reset-user-memory  Delete and recreate the user memory DB with the active embedding profile.
    doctor             Run diagnostic checks on setup, embedding provider, index, and memory.
    hook-session-start Emit SessionStart hook JSON for a given agent CLI format.
    serve              Launch the MCP stdio server (called by agent CLIs, not directly).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import click
from vibe_rag import __version__
from vibe_rag.paths import project_index_db_path, user_config_path, user_memory_db_path

_VIBE_RAG_BIN_PLACEHOLDER = "__VIBE_RAG_BIN__"
_VIBE_RAG_SHELL_BIN_PLACEHOLDER = "__VIBE_RAG_BIN_SHELL__"
_GENERATED_SCAFFOLD_FILES = (
    "AGENTS.md",
    ".vibe/config.toml",
    ".codex/config.toml",
    ".codex/hooks.json",
    ".claude/settings.json",
    ".gemini/settings.json",
    ".mcp.json",
)


@click.group()
@click.version_option(version=__version__)
def main():
    """vibe-rag — Semantic repo search and coding memory over MCP."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _embedding_dimensions() -> int:
    """Resolve the configured embedding dimension count, or abort with a ClickException."""
    try:
        from vibe_rag.indexing.embedder import resolve_embedding_dimensions

        return resolve_embedding_dimensions()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _embedding_model() -> str:
    """Resolve the configured embedding model name, or abort with a ClickException."""
    try:
        from vibe_rag.indexing.embedder import resolve_embedding_model

        return resolve_embedding_model()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _status_label(ok: bool, warning: bool = False) -> str:
    """Return a colored pass/warn/FAIL label for doctor output."""
    if warning:
        return click.style("warn", fg="yellow")
    if ok:
        return click.style("pass", fg="green")
    return click.style("FAIL", fg="red")


def _plain_status_label(ok: bool, warning: bool = False) -> str:
    """Return uncolored status label for assertions in tests."""
    if warning:
        return "warn"
    if ok:
        return "pass"
    return "FAIL"


def _read_toml_state(path: Path) -> tuple[dict | None, str]:
    """Parse a TOML file and return (data, state) where state is 'ok', 'missing', 'unreadable', or 'corrupt'."""
    if not path.exists():
        return None, "missing"
    try:
        return tomllib.loads(path.read_text()), "ok"
    except OSError:
        return None, "unreadable"
    except tomllib.TOMLDecodeError:
        return None, "corrupt"


def _read_toml(path: Path) -> dict | None:
    """Parse a TOML file, returning the dict or None on any failure."""
    parsed, _ = _read_toml_state(path)
    return parsed


def _resolve_command(command: str) -> tuple[bool, str]:
    """Resolve a command string to (found, absolute_path). Handles both bare names and paths."""
    if not command:
        return False, "missing command"
    if os.path.sep in command:
        path = Path(command).expanduser()
        return path.exists(), str(path)
    resolved = shutil.which(command)
    if resolved:
        return True, resolved
    return False, command


def _current_vibe_rag_binary() -> str:
    """Return the absolute path to the running vibe-rag binary, for template rewriting."""
    argv0 = str(Path(sys.argv[0]).expanduser())
    if Path(argv0).name == "vibe-rag":
        ok, resolved = _resolve_command(argv0)
        if ok:
            return resolved

    ok, resolved = _resolve_command("vibe-rag")
    if ok:
        return resolved
    return "vibe-rag"


def _rewrite_generated_client_files(target: Path) -> None:
    """Replace __VIBE_RAG_BIN__ placeholders in scaffolded config files with the real binary path.

    Only touches known generated files (.vibe/config.toml, .codex/*, .claude/*,
    .gemini/*, .mcp.json) -- user files in the project are never modified.
    """
    vibe_rag_bin = _current_vibe_rag_binary()
    shell_vibe_rag_bin = shlex.quote(vibe_rag_bin)
    replacements = {
        _VIBE_RAG_BIN_PLACEHOLDER: vibe_rag_bin,
        _VIBE_RAG_SHELL_BIN_PLACEHOLDER: shell_vibe_rag_bin,
    }
    for relative_path in _GENERATED_SCAFFOLD_FILES:
        if relative_path == "AGENTS.md":
            continue
        path = target / relative_path
        if not path.exists():
            continue

        text = path.read_text()
        updated = text
        for placeholder, value in replacements.items():
            updated = updated.replace(placeholder, value)
        if updated != text:
            path.write_text(updated)


def _generated_scaffold_exists(target: Path) -> bool:
    """Return True when *target* already contains vibe-rag scaffold files."""
    return any((target / relative_path).exists() for relative_path in _GENERATED_SCAFFOLD_FILES)


def _clear_generated_scaffold(target: Path) -> None:
    """Remove only the known generated scaffold files, preserving repo data like .vibe-rag/index.db."""
    for relative_path in _GENERATED_SCAFFOLD_FILES:
        (target / relative_path).unlink(missing_ok=True)


def _project_mcp_command_status(project_root: Path) -> dict:
    """Check that .vibe/config.toml declares a 'memory' MCP server whose command resolves on disk."""
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


def _client_cli_status(binary_name: str, display_name: str, version_flag: str = "--version") -> dict:
    """Check whether a client CLI binary is installed and responds to a version check."""
    cli_bin = shutil.which(binary_name)
    if not cli_bin:
        return {
            "ok": False,
            "warning": True,
            "detail": f"{display_name} not found. Install it to use vibe-rag with {display_name}.",
        }

    try:
        result = subprocess.run(
            [cli_bin, version_flag],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError as exc:
        return {"ok": False, "warning": True, "detail": f"{display_name} failed to start: {exc}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "warning": True, "detail": f"{display_name} version check timed out"}

    version_output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return {
            "ok": False,
            "warning": True,
            "detail": f"{display_name} found at {cli_bin} but `{binary_name} {version_flag}` failed",
        }
    return {"ok": True, "warning": False, "detail": f"{cli_bin} ({version_output or 'version unknown'})"}


def _vibe_cli_status() -> dict:
    """Check whether the Vibe CLI binary is installed."""
    return _client_cli_status("vibe", "Vibe CLI")


def _claude_cli_status() -> dict:
    """Check whether the Claude Code CLI binary is installed."""
    return _client_cli_status("claude", "Claude Code")


def _codex_cli_status() -> dict:
    """Check whether the Codex CLI binary is installed."""
    return _client_cli_status("codex", "Codex CLI")


def _gemini_cli_status() -> dict:
    """Check whether the Gemini CLI binary is installed."""
    return _client_cli_status("gemini", "Gemini CLI")


def _project_vibe_hook_status(project_root: Path) -> dict:
    """Check that .vibe/config.toml has a SessionStart hook whose command resolves on disk.

    Also detects and warns about the legacy background_mcp_hook format.
    Does NOT execute the hook command -- only verifies it exists.
    """
    vibe_config, state = _read_toml_state(project_root / ".vibe" / "config.toml")
    if vibe_config is None:
        if state == "corrupt":
            return {"ok": False, "warning": True, "detail": "invalid TOML in .vibe/config.toml"}
        if state == "unreadable":
            return {"ok": False, "warning": True, "detail": "unreadable .vibe/config.toml"}
        return {"ok": False, "warning": True, "detail": "missing .vibe/config.toml"}

    hooks = vibe_config.get("hooks")
    if isinstance(hooks, dict):
        session_start = hooks.get("SessionStart")
        if isinstance(session_start, list):
            for item in session_start:
                if not isinstance(item, dict):
                    continue
                command = str(item.get("command") or "").strip()
                if not command:
                    continue
                command_token = command.split()[0]
                ok, resolved = _resolve_command(command_token)
                if not ok:
                    return {"ok": False, "warning": True, "detail": f"hook command not found: {command_token}"}
                return {
                    "ok": True,
                    "warning": False,
                    "detail": f"SessionStart hook configured (not executed): {resolved}",
                }

    background = vibe_config.get("background_mcp_hook")
    if isinstance(background, dict) and background.get("enabled"):
        if background.get("tool_name") == "memory_load_session_context":
            return {
                "ok": True,
                "warning": True,
                "detail": "legacy background_mcp_hook enabled; migrate to [[hooks.SessionStart]]",
            }
        return {
            "ok": False,
            "warning": True,
            "detail": "background_mcp_hook.tool_name should be memory_load_session_context",
        }
    return {
        "ok": False,
        "warning": True,
        "detail": "hooks.SessionStart.command is not configured in .vibe/config.toml",
    }


def _codex_hook_status(project_root: Path) -> dict:
    """Check that .codex/hooks.json has a SessionStart hook whose command resolves on disk.

    Does NOT execute the hook command -- only verifies it exists.
    """
    hooks_path = project_root / ".codex" / "hooks.json"
    if not hooks_path.exists():
        return {"ok": False, "detail": "missing .codex/hooks.json"}
    try:
        hooks = json.loads(hooks_path.read_text())
    except json.JSONDecodeError:
        return {"ok": False, "detail": "invalid JSON in .codex/hooks.json"}

    session_starts = (hooks.get("hooks") or {}).get("SessionStart")
    if not isinstance(session_starts, list) or not session_starts:
        return {"ok": False, "detail": "no SessionStart command in .codex/hooks.json"}

    command = ""
    for item in session_starts:
        if not isinstance(item, dict):
            continue
        hooks_value = item.get("hooks")
        if not isinstance(hooks_value, list) or not hooks_value:
            continue
        first_hook = hooks_value[0]
        if not isinstance(first_hook, dict):
            continue
        command = first_hook.get("command") or ""
        if command:
            break

    if not command:
        return {"ok": False, "detail": "no SessionStart command in .codex/hooks.json"}

    try:
        argv = shlex.split(str(command))
    except ValueError:
        return {"ok": False, "detail": "invalid SessionStart command"}
    if not argv:
        return {"ok": False, "detail": "empty SessionStart command"}

    ok, resolved = _resolve_command(argv[0])
    if not ok:
        return {"ok": False, "detail": f"hook command not found: {argv[0]}"}

    return {"ok": True, "detail": f"SessionStart hook configured (not executed): {resolved}"}


def _db_readable_status(db_path: Path, *, label: str) -> dict:
    """Open a sqlite-vec DB to verify it exists and is readable."""
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


def _openrouter_setup_hint() -> str:
    """Return a one-line hint for configuring OpenRouter embedding credentials."""
    return (
        "create `~/.vibe-rag/config.toml` with "
        "`[embedding] api_key = \"...\"` "
        "(env overrides: `RAG_OR_API_KEY`, `RAG_OR_EMBED_MOD`, `RAG_OR_EMBED_DIM`)"
    )


@main.command()
@click.argument("name", required=False)
@click.option("--here", is_flag=True, help="Write vibe-rag scaffold into the current directory.")
def init(name: str | None, here: bool):
    """Scaffold a new project or stamp the current repo with MCP config for all supported agent CLIs."""
    from importlib.resources import files as pkg_files

    templates_dir = Path(str(pkg_files("vibe_rag") / "templates"))
    bundle_dir = Path(str(pkg_files("vibe_rag") / "template_bundle"))

    if here and name:
        raise click.UsageError("pass either a project name or `--here`, not both")

    in_place = here or name in {None, ".", "./"}

    target = Path.cwd() if in_place else Path.cwd() / name

    if target.exists():
        if not target.is_dir():
            raise click.ClickException(f"{target} exists and is not a directory")
        if in_place:
            _clear_generated_scaffold(target)
        else:
            click.echo(f"\n  {target} already exists.")
            if _generated_scaffold_exists(target):
                prompt = "  Re-stamp vibe-rag scaffold in the existing directory?"
            else:
                prompt = "  Write vibe-rag scaffold into the existing directory?"
            if click.confirm(prompt, default=False):
                _clear_generated_scaffold(target)
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

    _rewrite_generated_client_files(target)
    _initialize_git_repo(target)

    # .gitignore
    gitignore = target / ".gitignore"
    ignore_lines = [
        ".vibe-rag/index.db",
        ".vibe-rag/index.db-shm",
        ".vibe-rag/index.db-wal",
        ".vibe-rag/backups/",
    ]
    if gitignore.exists():
        text = gitignore.read_text()
        additions = [line for line in ignore_lines if line not in text]
        if additions:
            gitignore.write_text(text.rstrip() + "\n" + "\n".join(additions) + "\n")
    else:
        gitignore.write_text("\n".join(ignore_lines) + "\n")

    created_label = target.name if in_place else name
    click.echo(f"\n  ✓ {created_label} created at {target}\n")
    click.echo("    AGENTS.md          — project coding rules")
    click.echo("    .codex/            — Codex MCP + session-start hook")
    click.echo("    .claude/           — Claude Code session-start hook")
    click.echo("    .gemini/           — Gemini CLI MCP + session-start hook")
    click.echo("    .mcp.json          — Claude Code MCP server config")
    click.echo("    .vibe/config.toml  — Vibe MCP + hooks")
    click.echo("    .vibe-rag/         — project-local vibe-rag runtime state")
    click.echo("\n  Supported clients:")
    click.echo("    All four agent CLIs are supported: Claude Code, Codex, Gemini CLI, and Vibe.")
    profile = {
        "model": _embedding_model(),
        "dimensions": str(_embedding_dimensions()),
    }
    click.echo("\n  Embeddings:")
    click.echo("    Recommended path: ~/.vibe-rag/config.toml")
    click.echo(f"    Default profile: {profile['model']} @ {profile['dimensions']} dims")
    click.echo(f"    Setup hint: {_openrouter_setup_hint()}")
    click.echo("\n  Golden path:")
    click.echo(f"    cd {target}" if not in_place else "    already in the repo root")
    click.echo("    write ~/.vibe-rag/config.toml once with your OpenRouter key")
    click.echo("    or use RAG_OR_API_KEY=... as a shell/session override")
    click.echo("    start Claude Code, Codex, Gemini CLI, or Vibe in this repo")
    click.echo('    "load session context for understanding this repo"')
    click.echo('    "index this project"')
    click.echo("    if tools or hooks are missing, run vibe-rag doctor")
    click.echo("\n  New repo path:")
    click.echo("    to create a new subdirectory instead: vibe-rag init NAME")
    click.echo("\n  Tool naming:")
    click.echo("    The MCP server exposes bare tools like load_session_context, index_project, search, remember, and project_status.")
    click.echo("    Depending on client config the server may be named memory, so tools appear as memory_load_session_context, memory_index_project, memory_search, memory_search_memory, memory_remember, and memory_project_status.")

def _initialize_git_repo(target: Path) -> None:
    """Run ``git init`` in *target* if it is not already a git repo and git is available."""
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

def _index_freshness(db) -> str:
    """Return a human-readable freshness label based on the last index timestamp."""
    import json as _json
    from datetime import datetime, timezone

    raw = db.get_setting("project_index_metadata")
    if not raw:
        return "unknown"
    try:
        metadata = _json.loads(raw)
    except Exception:
        return "unknown"
    indexed_at = metadata.get("indexed_at")
    if not indexed_at:
        return "unknown"
    try:
        ts = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - ts
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"fresh (indexed {int(delta.total_seconds() / 60)}m ago)"
        if hours < 24:
            return f"fresh (indexed {int(hours)}h ago)"
        days = int(hours / 24)
        return f"stale (indexed {days}d ago)"
    except Exception:
        return "unknown"


def _distinct_file_count(db) -> int:
    """Count distinct files across code_chunks and docs tables."""
    conn = db._get_conn()
    code_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM code_chunks").fetchone()[0]
    doc_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM docs").fetchone()[0]
    return code_files + doc_files


def _format_language_stats(lang_stats: dict[str, int], top_n: int = 5) -> str:
    """Format language stats as 'python (180), javascript (80), ...'."""
    sorted_langs = sorted(lang_stats.items(), key=lambda x: -x[1])
    parts = []
    for lang, count in sorted_langs[:top_n]:
        label = lang if lang else "unknown"
        parts.append(f"{label} ({count})")
    if len(sorted_langs) > top_n:
        rest = sum(c for _, c in sorted_langs[top_n:])
        parts.append(f"+{len(sorted_langs) - top_n} more ({rest})")
    return ", ".join(parts) if parts else "none"


def _remove_sqlite_files(db_path: Path) -> None:
    """Delete a SQLite DB file plus its WAL/SHM sidecars."""
    for suffix in ("", "-shm", "-wal"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


@main.command()
def status():
    """Show index, memory, and health summary for the current project."""
    from vibe_rag.db.sqlite import SqliteVecDB
    from vibe_rag.server import _ensure_project_id
    from vibe_rag.tools import _stale_state

    db_path = Path(os.environ.get("RAG_DB", project_index_db_path())).expanduser()
    user_db_path = Path(os.environ.get("RAG_USER_DB", user_memory_db_path())).expanduser()
    embedding_dimensions = _embedding_dimensions()
    click.echo(f"\n  vibe-rag {__version__}")
    click.echo(f"  DB: {db_path}\n")

    def _user_memory_summary() -> None:
        if not user_db_path.exists():
            click.echo(f"  User:      0 memories ({user_db_path})")
            return
        try:
            user_db = SqliteVecDB(user_db_path, embedding_dimensions=embedding_dimensions)
            user_db.initialize()
            click.echo(f"  User:      {user_db.memory_count()} memories ({user_db_path})")
            user_db.close()
        except Exception as exc:
            click.echo(f"  User:      unreadable ({user_db_path})")
            click.echo(f"  Detail:    {exc}")
            click.echo("  Action:    vibe-rag reset-user-memory")

    if db_path.exists():
        try:
            db = SqliteVecDB(db_path, embedding_dimensions=embedding_dimensions)
            db.initialize()
            code_chunks = db.code_chunk_count()
            doc_chunks = db.doc_count()
            file_count = _distinct_file_count(db)
            project_memories = db.memory_count()
            lang_stats = db.language_stats()
            freshness = _index_freshness(db)
            stale_state = _stale_state(db, Path.cwd(), _ensure_project_id())
            db.close()

            click.echo(f"  Index:     {code_chunks} code chunks, {doc_chunks} doc chunks ({file_count} files)")

            user_memories = 0
            user_error = None
            if user_db_path.exists():
                try:
                    user_db = SqliteVecDB(user_db_path, embedding_dimensions=embedding_dimensions)
                    user_db.initialize()
                    user_memories = user_db.memory_count()
                    user_db.close()
                except Exception as exc:
                    user_error = exc
            click.echo(f"  Memory:    {project_memories} project, {user_memories} user")
            if user_error is not None:
                click.echo(f"  User:      unreadable ({user_db_path})")
                click.echo(f"  Detail:    {user_error}")
                click.echo("  Action:    vibe-rag reset-user-memory")
            if stale_state.get("is_incompatible"):
                click.echo("  Health:    incompatible index")
                first_warning = (stale_state.get("warnings") or [{}])[0]
                if first_warning.get("detail"):
                    click.echo(f"  Detail:    {first_warning['detail']}")
                click.echo("  Action:    vibe-rag reindex --full")
            elif stale_state.get("warnings"):
                click.echo(f"  Health:    {freshness}")
                click.echo(f"  Detail:    {stale_state['warnings'][0]['detail']}")
            else:
                click.echo(f"  Health:    {freshness}")
            if lang_stats:
                click.echo(f"  Languages: {_format_language_stats(lang_stats)}")
        except Exception as exc:
            click.echo(f"  Index:     unreadable ({db_path})")
            click.echo("  Health:    unreadable index")
            click.echo(f"  Detail:    {exc}")
            click.echo("  Action:    vibe-rag reindex --full")
            _user_memory_summary()
    else:
        click.echo("  No index yet. Run `vibe-rag reindex` or use `index_project` through your client.")
        _user_memory_summary()
    click.echo()


@main.command("reindex")
@click.option(
    "--full",
    is_flag=True,
    help="Clear incremental index state and rebuild the entire project. Use after embedding profile changes.",
)
@click.argument("paths", nargs=-1)
def reindex(paths: tuple[str, ...], full: bool):
    """Re-index code and docs for the current project.

    Incremental by default (only changed files). Pass --full to clear state
    and rebuild everything. Optional PATHS restrict indexing to specific
    directories or files (cannot be combined with --full).
    """
    from vibe_rag.tools import index_project
    from vibe_rag.tools.index import _index_project_impl

    if full and paths and tuple(paths) != (".",):
        raise click.ClickException("--full rebuilds the entire project; omit paths.")

    target_paths: list[str] | str = list(paths) if paths else "."
    click.echo()
    if full:
        click.echo("Full rebuild requested. Clearing incremental index state and rebuilding the full project.")
        result = _index_project_impl(
            ".",
            force_full_rebuild=True,
            rebuild_reason="explicit_cli_full_reindex",
        )
    else:
        result = index_project(target_paths)
    if result.get("ok"):
        click.echo(result.get("summary") or "Index complete")
    else:
        error = result.get("error") or {}
        click.echo(f"Error: {error.get('message') or 'indexing failed'}")
    click.echo()


@main.command("reset-index")
def reset_index():
    """Alias for ``reindex --full``. Clears incremental state and rebuilds the full project index."""
    ctx = click.get_current_context()
    ctx.invoke(reindex, paths=(), full=True)


@main.command("reset-user-memory")
def reset_user_memory():
    """Delete and recreate the user memory DB with the current embedding dimensions."""
    from vibe_rag.db.sqlite import SqliteVecDB

    user_db_path = Path(os.environ.get("RAG_USER_DB", user_memory_db_path())).expanduser()
    embedding_dimensions = _embedding_dimensions()

    click.echo()
    _remove_sqlite_files(user_db_path)
    user_db = SqliteVecDB(user_db_path, embedding_dimensions=embedding_dimensions)
    user_db.initialize()
    click.echo(f"Reset user memory DB at {user_db_path}")
    click.echo(f"User memory count: {user_db.memory_count()}")
    user_db.close()
    click.echo()


def _check_language_coverage(db) -> dict:
    """Check whether code chunks have language annotations."""
    lang_stats = db.language_stats()
    total = sum(lang_stats.values())
    none_count = lang_stats.get(None, 0)
    if total == 0:
        return {"ok": True, "warning": True, "detail": "no code chunks indexed"}
    if none_count == total:
        return {"ok": False, "warning": True, "detail": f"all {total} chunks have unknown language"}
    if none_count > 0:
        return {"ok": True, "warning": True, "detail": f"{none_count}/{total} chunks have unknown language"}
    return {"ok": True, "warning": False, "detail": f"{total} chunks across {len(lang_stats)} languages"}


def _check_memory_health(db, user_db) -> dict:
    """Check memory staleness and cleanup pressure."""
    total = db.memory_count(include_superseded=True)
    active = db.memory_count(include_superseded=False)
    superseded = total - active
    user_total = user_db.memory_count(include_superseded=True) if user_db else 0
    user_active = user_db.memory_count(include_superseded=False) if user_db else 0
    user_superseded = user_total - user_active

    stale_total = superseded + user_superseded
    parts = []
    if stale_total > 0:
        parts.append(f"{stale_total} superseded")
    parts.append(f"{active + user_active} active")
    detail = ", ".join(parts)
    if stale_total > 10:
        return {"ok": False, "warning": True, "detail": detail + " -- consider cleanup"}
    return {"ok": True, "warning": False, "detail": detail}


def _check_tool_count() -> dict:
    """Verify expected MCP tools are registered."""
    try:
        from vibe_rag.server import mcp as _mcp
        tool_names = sorted(_mcp._tool_manager._tools.keys())
        count = len(tool_names)
        if count == 0:
            return {"ok": False, "warning": False, "detail": "no MCP tools registered"}
        return {"ok": True, "warning": False, "detail": f"{count} tools registered", "tools": tool_names}
    except Exception as exc:
        return {"ok": False, "warning": True, "detail": f"tool check failed: {exc}"}


@main.command()
@click.option("--fix", is_flag=True, help="Retained for compatibility; doctor no longer performs provider setup.")
def doctor(fix: bool):
    """Run diagnostic checks on MCP config, agent CLIs, hooks, DBs, embedding, trust, index, and memory health.

    Checks: MCP command, Claude/Codex/Gemini/Vibe CLI presence, Vibe and
    Codex session-start hooks, project and user DB readability, embedding
    provider connectivity, Vibe and Codex repo trust, language coverage,
    memory staleness, registered MCP tool count, and index state freshness.
    """
    from vibe_rag.db.sqlite import SqliteVecDB
    from vibe_rag.indexing.embedder import embedding_provider_status
    from vibe_rag.server import _ensure_project_id, _get_db, _get_embedder
    from vibe_rag.tools import _codex_trust_status, _stale_state, _vibe_trust_status

    project_root = Path.cwd().resolve()
    project_db_path = Path(os.environ.get("RAG_DB", project_index_db_path(project_root))).expanduser()
    user_db_path = Path(os.environ.get("RAG_USER_DB", user_memory_db_path())).expanduser()
    provider = embedding_provider_status()
    project_id = _ensure_project_id()
    mcp_status = _project_mcp_command_status(project_root)
    claude_cli_st = _claude_cli_status()
    codex_cli_st = _codex_cli_status()
    gemini_cli_st = _gemini_cli_status()
    vibe_cli_status = _vibe_cli_status()
    vibe_hook_status = _project_vibe_hook_status(project_root)
    hook_status = _codex_hook_status(project_root)
    project_db_status = _db_readable_status(project_db_path, label="Project")
    user_db_status = _db_readable_status(user_db_path, label="User")
    vibe_trust = _vibe_trust_status(project_root)
    codex_trust = _codex_trust_status(project_root)

    provider_detail = str(provider["detail"])
    provider_ok = bool(provider["ok"])
    if provider_ok:
        try:
            _get_embedder().embed_text_sync(["doctor healthcheck"])
        except Exception as exc:
            provider_ok = False
            provider_detail = f"embedding failed: {exc}"

    stale_state = {"is_stale": False, "is_incompatible": False, "warnings": []}
    if project_db_status["ok"]:
        try:
            stale_state = _stale_state(_get_db(), project_root, project_id)
        except Exception as exc:
            stale_state = {
                "is_stale": True,
                "is_incompatible": True,
                "warnings": [{"kind": "stale_check_failed", "detail": f"stale-state check failed: {exc}"}],
            }

    lang_coverage = {"ok": True, "warning": True, "detail": "no project DB"}
    if project_db_status["ok"]:
        try:
            embedding_dimensions = _embedding_dimensions()
            proj_db = SqliteVecDB(project_db_path, embedding_dimensions=embedding_dimensions)
            proj_db.initialize()
            lang_coverage = _check_language_coverage(proj_db)
            proj_db.close()
        except Exception as exc:
            lang_coverage = {"ok": False, "warning": True, "detail": f"language check failed: {exc}"}

    memory_health = {"ok": True, "warning": True, "detail": "no project DB"}
    if project_db_status["ok"]:
        try:
            embedding_dimensions = _embedding_dimensions()
            proj_db = SqliteVecDB(project_db_path, embedding_dimensions=embedding_dimensions)
            proj_db.initialize()
            user_db_obj = None
            if user_db_path.exists():
                user_db_obj = SqliteVecDB(user_db_path, embedding_dimensions=embedding_dimensions)
                user_db_obj.initialize()
            memory_health = _check_memory_health(proj_db, user_db_obj)
            proj_db.close()
            if user_db_obj:
                user_db_obj.close()
        except Exception as exc:
            memory_health = {"ok": False, "warning": True, "detail": f"memory health check failed: {exc}"}

    tool_status = _check_tool_count()

    click.echo(f"\n  vibe-rag {__version__}")
    click.echo(f"  Project root: {project_root}")
    click.echo(f"  Project id:   {project_id}")
    click.echo(f"  Project DB:   {project_db_path}")
    click.echo(f"  User DB:      {user_db_path}")
    click.echo(f"  Provider:     {provider['provider']}")
    click.echo(f"  Config path:  {provider.get('config_path') or user_config_path()}")
    click.echo(f"  Config source: {provider.get('source') or 'unknown'}")
    click.echo(f"  Model:        {provider['model'] or 'unset'}")
    click.echo(f"  Dimensions:   {provider.get('dimensions') or 'unset'}")
    click.echo("  Golden path:  OpenRouter embeddings with one API key")
    click.echo()

    vibe_hook_ok = vibe_hook_status["ok"]
    vibe_hook_warning = vibe_hook_status.get("warning", False)
    vibe_hook_detail = str(vibe_hook_status["detail"])
    if vibe_hook_ok and vibe_trust["status"] != "ok":
        vibe_hook_warning = True
        vibe_hook_detail += "; repo not trusted"

    codex_hook_ok = hook_status["ok"]
    codex_hook_warning = False
    codex_hook_detail = str(hook_status["detail"])
    if codex_hook_ok and codex_trust["status"] != "ok":
        codex_hook_warning = True
        codex_hook_detail += "; repo not trusted"

    def _emit(label: str, ok: bool, warning: bool, detail: str) -> None:
        click.echo(f"  [{_status_label(ok, warning)}] {label:<16s} {detail}")

    _emit("MCP command", mcp_status["ok"], False, mcp_status["detail"])
    _emit("Claude Code", claude_cli_st["ok"], claude_cli_st.get("warning", False), claude_cli_st["detail"])
    _emit("Codex CLI", codex_cli_st["ok"], codex_cli_st.get("warning", False), codex_cli_st["detail"])
    _emit("Gemini CLI", gemini_cli_st["ok"], gemini_cli_st.get("warning", False), gemini_cli_st["detail"])
    _emit("Vibe CLI", vibe_cli_status["ok"], vibe_cli_status.get("warning", False), vibe_cli_status["detail"])
    _emit("Vibe hooks", vibe_hook_ok, vibe_hook_warning, vibe_hook_detail)
    _emit("SessionStart", codex_hook_ok, codex_hook_warning, codex_hook_detail)
    _emit("Project DB", project_db_status["ok"], project_db_status.get("warning", False), project_db_status["detail"])
    _emit("User DB", user_db_status["ok"], user_db_status.get("warning", False), user_db_status["detail"])
    _emit("Embedding", provider_ok, bool(provider.get("warning", False)), provider_detail)
    _emit("Vibe trust", vibe_trust["status"] == "ok", vibe_trust["status"] == "warn", vibe_trust["detail"])
    _emit("Codex trust", codex_trust["status"] == "ok", codex_trust["status"] == "warn", codex_trust["detail"])
    _emit("Languages", lang_coverage["ok"], lang_coverage.get("warning", False), lang_coverage["detail"])
    _emit("Memory health", memory_health["ok"], memory_health.get("warning", False), memory_health["detail"])
    _emit("MCP tools", tool_status["ok"], tool_status.get("warning", False), tool_status["detail"])

    if not project_db_status["ok"]:
        _emit(
            "Index state",
            False,
            project_db_status.get("warning", False),
            "skipped because project DB is not readable",
        )
        if not project_db_status.get("warning", False):
            click.echo("  Suggested fix: vibe-rag reindex --full")
            click.echo("  Alias:         vibe-rag reset-index")
    elif stale_state.get("warnings"):
        stale_ok = False
        stale_warning = not stale_state.get("is_incompatible", False)
        label = "Index state"
        first_warning = stale_state["warnings"][0]
        _emit(label, stale_ok, stale_warning, first_warning["detail"])
        for warning in stale_state["warnings"][1:]:
            click.echo(f"                         {warning['detail']}")
        if stale_state.get("is_incompatible", False):
            click.echo("  Suggested fix: vibe-rag reindex --full")
            click.echo("  Alias:         vibe-rag reset-index")
        else:
            click.echo("  Suggested fix: vibe-rag reindex")
    else:
        _emit("Index state", True, False, "no stale or incompatible index warnings")

    if not user_db_status["ok"] and not user_db_status.get("warning", False):
        click.echo("  Suggested fix: vibe-rag reset-user-memory")
    if not provider_ok:
        click.echo(f"  Suggested next step: {_openrouter_setup_hint()}")
    if fix:
        click.echo("  Automatic provider setup was removed. Configure OpenRouter credentials explicitly.")
    click.echo()


@main.command("hook-session-start")
@click.option("--format", "target_format", type=click.Choice(["codex", "claude", "gemini", "vibe"]), required=True)
def hook_session_start(target_format: str):
    """Emit SessionStart hook JSON for a given agent CLI format.

    Reads hook input from stdin and writes the formatted JSON response
    to stdout. Called by agent CLIs at session start, not directly by users.
    """
    from vibe_rag.hook_bridge import render_session_start_hook_json

    raw_input = sys.stdin.read()
    click.echo(render_session_start_hook_json(target_format, raw_input))


@main.command()
def serve():
    """Launch the MCP stdio server (called by agent CLIs, not directly)."""
    from vibe_rag.server import run_server
    run_server()


if __name__ == "__main__":
    main()
