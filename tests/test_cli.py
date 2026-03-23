from pathlib import Path
import os
import subprocess
import sys
import json

from click.testing import CliRunner
import vibe_rag.cli as cli
from vibe_rag.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.30" in result.output


def test_cli_help_uses_broader_product_framing():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Semantic repo search and coding memory over MCP." in result.output
    assert "for Mistral Vibe" not in result.output


def test_cli_status():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            ["status"],
            env={
                "RAG_DB": str(Path.cwd() / "project.db"),
                "RAG_USER_DB": str(Path.cwd() / "user.db"),
            },
        )
        assert result.exit_code == 0
        assert "vibe-rag" in result.output


def test_cli_status_uses_env_db_paths_and_dimensions(monkeypatch, tmp_path):
    runner = CliRunner()
    project_db = tmp_path / "project.db"
    user_db = tmp_path / "user.db"
    monkeypatch.setenv("RAG_DB", str(project_db))
    monkeypatch.setenv("RAG_USER_DB", str(user_db))
    monkeypatch.setenv("RAG_OR_EMBED_DIM", "2560")

    result = runner.invoke(main, ["status"])

    assert result.exit_code == 0
    assert str(project_db) in result.output
    assert str(user_db) in result.output


def test_cli_reindex_uses_index_project(monkeypatch):
    runner = CliRunner()
    calls = {}

    def fake_index_project(paths):
        calls["paths"] = paths
        return {"ok": True, "summary": "Indexed 2 files"}

    monkeypatch.setattr("vibe_rag.tools.index_project", fake_index_project)

    result = runner.invoke(main, ["reindex", "src", "docs"])

    assert result.exit_code == 0
    assert calls["paths"] == ["src", "docs"]
    assert "Indexed 2 files" in result.output


def test_cli_reindex_defaults_to_current_project(monkeypatch):
    runner = CliRunner()
    calls = {}

    def fake_index_project(paths):
        calls["paths"] = paths
        return {"ok": True, "summary": "Indexed current project"}

    monkeypatch.setattr("vibe_rag.tools.index_project", fake_index_project)

    result = runner.invoke(main, ["reindex"])

    assert result.exit_code == 0
    assert calls["paths"] == "."
    assert "Indexed current project" in result.output


def test_cli_reindex_full_uses_explicit_full_rebuild(monkeypatch):
    runner = CliRunner()
    calls = {}

    def fake_index_project_impl(paths=None, *, progress_callback=None, force_full_rebuild=False, rebuild_reason=None):
        calls["paths"] = paths
        calls["force_full_rebuild"] = force_full_rebuild
        calls["rebuild_reason"] = rebuild_reason
        return {"ok": True, "summary": "Rebuilt index"}

    monkeypatch.setattr("vibe_rag.tools.index._index_project_impl", fake_index_project_impl)

    result = runner.invoke(main, ["reindex", "--full"])

    assert result.exit_code == 0
    assert calls == {
        "paths": ".",
        "force_full_rebuild": True,
        "rebuild_reason": "explicit_cli_full_reindex",
    }
    assert "Full rebuild requested." in result.output
    assert "Rebuilt index" in result.output


def test_cli_reindex_full_rejects_partial_paths():
    runner = CliRunner()

    result = runner.invoke(main, ["reindex", "--full", "src"])

    assert result.exit_code != 0
    assert "--full rebuilds the entire project; omit paths." in result.output


def test_cli_reset_index_alias_uses_full_rebuild(monkeypatch):
    runner = CliRunner()
    calls = {}

    def fake_index_project_impl(paths=None, *, progress_callback=None, force_full_rebuild=False, rebuild_reason=None):
        calls["paths"] = paths
        calls["force_full_rebuild"] = force_full_rebuild
        calls["rebuild_reason"] = rebuild_reason
        return {"ok": True, "summary": "Rebuilt index"}

    monkeypatch.setattr("vibe_rag.tools.index._index_project_impl", fake_index_project_impl)

    result = runner.invoke(main, ["reset-index"])

    assert result.exit_code == 0
    assert calls == {
        "paths": ".",
        "force_full_rebuild": True,
        "rebuild_reason": "explicit_cli_full_reindex",
    }
    assert "Full rebuild requested." in result.output


def _patch_doctor_new_checks(monkeypatch):
    """Patch the new doctor checks (language coverage, memory health, tool count)."""
    class FakeSqliteVecDB:
        def __init__(self, path, embedding_dimensions=None):
            self.path = path
            self.embedding_dimensions = embedding_dimensions

        def initialize(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("vibe_rag.db.sqlite.SqliteVecDB", FakeSqliteVecDB)
    monkeypatch.setattr("vibe_rag.cli._check_language_coverage", lambda db: {"ok": True, "warning": False, "detail": "100 chunks across 3 languages"})
    monkeypatch.setattr("vibe_rag.cli._check_memory_health", lambda db, user_db: {"ok": True, "warning": False, "detail": "5 active"})
    monkeypatch.setattr("vibe_rag.cli._check_tool_count", lambda: {"ok": True, "warning": False, "detail": "15 tools registered"})


def test_cli_doctor_defaults_to_openrouter(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": True,
            "detail": "ready",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "SessionStart hook returned context via /usr/local/bin/vibe-rag"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 2560]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted in ~/.vibe/trusted_folders.toml"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted in ~/.codex/config.toml"})
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {"is_stale": False, "is_incompatible": False, "warnings": []},
    )
    _patch_doctor_new_checks(monkeypatch)
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "Project id:   demo-project" in result.output
    assert "[pass] MCP command" in result.output
    assert "[pass] Vibe CLI" in result.output
    assert "[pass] Vibe hooks" in result.output
    assert "[pass] SessionStart" in result.output
    assert "[pass] Embedding" in result.output
    assert "[pass] Languages" in result.output
    assert "[pass] Memory health" in result.output
    assert "[pass] MCP tools" in result.output
    assert "Model:        perplexity/pplx-embed-v1-4b" in result.output
    assert "Dimensions:   2560" in result.output
    assert "Golden path:  OpenRouter embeddings with one API key" in result.output
    assert "openrouter" in result.output


def test_cli_doctor_for_openrouter_without_api_key(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": False,
            "detail": "RAG_OR_API_KEY not set",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": False, "detail": "MCP command not found: vibe-rag"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": False, "detail": "missing .codex/hooks.json"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": False, "warning": True, "detail": f"{label} DB missing at {path}"},
    )
    monkeypatch.setattr(
        "vibe_rag.cli._vibe_cli_status",
        lambda: {
            "ok": False,
            "warning": True,
            "detail": "Vibe CLI not found. Vibe stays bootstrapped, but Claude Code and Codex are the strongest validated clients today.",
        },
    )
    monkeypatch.setattr(
        "vibe_rag.cli._project_vibe_hook_status",
        lambda root: {"ok": False, "warning": True, "detail": "hooks.SessionStart.command is not configured in .vibe/config.toml"},
    )
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "warn", "detail": "repo not trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "warn", "detail": "repo not trusted"})
    _patch_doctor_new_checks(monkeypatch)
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "[FAIL] MCP command" in result.output
    assert "[warn] Vibe CLI" in result.output
    assert "[warn] Vibe hooks" in result.output
    assert "[warn] Project DB" in result.output
    assert "Suggested next step:" in result.output
    assert "export `RAG_OR_API_KEY=...`" in result.output


def test_cli_doctor_warns_for_embedding_provider_warning(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": True,
            "warning": True,
            "detail": "provider temporarily unavailable",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.server._get_embedder",
        lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 2560]})(),
    )
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "SessionStart hook returned context via /usr/local/bin/vibe-rag"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {"is_stale": False, "is_incompatible": False, "warnings": []},
    )
    _patch_doctor_new_checks(monkeypatch)
    result = runner.invoke(main, ["doctor", "--fix"])
    assert result.exit_code == 0
    assert "[warn] Embedding" in result.output
    assert "provider temporarily unavailable" in result.output
    assert "Automatic provider setup was removed." in result.output


def test_cli_doctor_does_not_show_provider_fast_path(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": True,
            "detail": "ready",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": False, "detail": "missing .codex/hooks.json"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (vibe 0.1.0)"})
    monkeypatch.setattr(
        "vibe_rag.cli._project_vibe_hook_status",
        lambda root: {"ok": False, "warning": True, "detail": "hooks.SessionStart.command is not configured in .vibe/config.toml"},
    )
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 2560]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {"is_stale": False, "is_incompatible": False, "warnings": []},
    )
    _patch_doctor_new_checks(monkeypatch)

    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[warn] Embedding" not in result.output
    assert "Ollama fast path:" not in result.output


def test_project_mcp_command_status_reports_invalid_toml(tmp_path: Path):
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / "config.toml").write_text("[mcp_servers\nname = 'memory'\n")

    status = cli._project_mcp_command_status(tmp_path)

    assert status["ok"] is False
    assert status["detail"] == "invalid TOML in .vibe/config.toml"


def test_project_vibe_hook_status_distinguishes_missing_and_invalid_toml(tmp_path: Path):
    missing = cli._project_vibe_hook_status(tmp_path)
    assert missing["ok"] is False
    assert missing["detail"] == "missing .vibe/config.toml"

    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / "config.toml").write_text("[hooks\nSessionStart = true\n")

    invalid = cli._project_vibe_hook_status(tmp_path)
    assert invalid["ok"] is False
    assert invalid["detail"] == "invalid TOML in .vibe/config.toml"


def test_project_vibe_hook_status_does_not_execute_session_start(tmp_path: Path, monkeypatch):
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hook = fake_bin / "session-start-hook"
    fake_hook.write_text("#!/bin/sh\necho pwned > /tmp/vibe_rag_poc\n")
    fake_hook.chmod(0o755)

    (vibe_dir / "config.toml").write_text(
        "[[hooks.SessionStart]]\n"
        f"command = \"{fake_hook}\"\n"
    )

    monkeypatch.setattr("vibe_rag.cli.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hook command executed")))

    status = cli._project_vibe_hook_status(tmp_path)

    assert status["ok"] is True
    assert "SessionStart hook configured (not executed)" in status["detail"]


def test_codex_hook_status_does_not_execute_session_start(tmp_path: Path, monkeypatch):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hook = fake_bin / "session-start-hook"
    fake_hook.write_text("#!/bin/sh\necho pwned > /tmp/vibe_rag_poc\n")
    fake_hook.chmod(0o755)

    (codex_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"command": str(fake_hook)}]}
                    ]
                }
            }
        )
    )

    monkeypatch.setattr("vibe_rag.cli.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hook command executed")))

    status = cli._codex_hook_status(tmp_path)

    assert status["ok"] is True
    assert "SessionStart hook configured (not executed)" in status["detail"]


def test_cli_doctor_warns_for_configured_hooks_in_untrusted_repo(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": True,
            "detail": "ready",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr(
        "vibe_rag.cli._codex_hook_status",
        lambda root: {"ok": True, "detail": "SessionStart hook configured (not executed): /usr/local/bin/vibe-rag"},
    )
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (vibe 0.1.0)"})
    monkeypatch.setattr(
        "vibe_rag.cli._project_vibe_hook_status",
        lambda root: {"ok": True, "warning": False, "detail": "SessionStart hook configured (not executed): /usr/local/bin/vibe-rag"},
    )
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 2560]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "warn", "detail": "repo not trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "warn", "detail": "repo not trusted"})
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {"is_stale": False, "is_incompatible": False, "warnings": []},
    )
    _patch_doctor_new_checks(monkeypatch)

    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[warn] Vibe hooks" in result.output
    assert "[warn] SessionStart" in result.output
    assert "not executed" in result.output
    assert "repo not trusted" in result.output


def test_cli_doctor_reports_stale_state(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": True,
            "detail": "ready",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "SessionStart hook returned context via /usr/local/bin/vibe-rag"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 2560]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {
            "is_stale": True,
            "is_incompatible": False,
            "warnings": [
                {"kind": "git_head_changed", "detail": "git HEAD changed since index (abc -> def)"},
                {"kind": "indexed_files_missing", "detail": "2 indexed files no longer exist"},
            ],
        },
    )
    _patch_doctor_new_checks(monkeypatch)

    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[warn] Index state" in result.output
    assert "git HEAD changed since index" in result.output
    assert "Suggested fix: vibe-rag reindex" in result.output


def test_cli_doctor_reports_incompatible_index_with_full_rebuild_fix(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "openrouter",
            "ok": True,
            "detail": "ready",
            "model": "perplexity/pplx-embed-v1-4b",
            "dimensions": 2560,
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "SessionStart hook returned context via /usr/local/bin/vibe-rag"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 2560]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {
            "is_stale": True,
            "is_incompatible": True,
            "warnings": [
                {"kind": "embedding_profile_changed", "detail": "embedding profile changed since last index (old -> new)"},
            ],
        },
    )
    _patch_doctor_new_checks(monkeypatch)

    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[FAIL] Index state" in result.output
    assert "Suggested fix: vibe-rag reindex --full" in result.output
    assert "Alias:         vibe-rag reset-index" in result.output


def test_cli_init_prints_openrouter_golden_path(monkeypatch):
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

    assert result.exit_code == 0
    assert "One obvious way: OpenRouter embeddings." in result.output
    assert "Default profile: perplexity/pplx-embed-v1-4b @ 2560 dims" in result.output
    assert "Next step: export `RAG_OR_API_KEY=...`" in result.output
    assert "Golden path:" in result.output
    assert "start Claude Code, Codex, or Vibe in this repo" in result.output
    assert '"load session context for understanding this repo"' in result.output
    assert '"index this project"' in result.output
    assert "memory_load_session_context" in result.output
    assert "memory_project_status" in result.output


def test_cli_module_entrypoint():
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path if not pythonpath else f"{src_path}:{pythonpath}"

    result = subprocess.run(
        [sys.executable, "-m", "vibe_rag.cli", "--version"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "0.0.30" in result.stdout


def test_cli_init_does_not_persist_secrets(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vibe_rag.cli._current_vibe_rag_binary", lambda: "/tmp/fake-bin/vibe-rag")
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            ["init", "demo"],
            env={
                "RAG_OR_API_KEY": "top-secret-key",
                "RAG_USER_DB": "/tmp/vibe-user.db",
            },
        )

        assert result.exit_code == 0

        config_text = Path("demo/.vibe/config.toml").read_text()
        assert "top-secret-key" not in config_text
        assert "/tmp/vibe-user.db" not in config_text
        assert 'skill_paths = [".vibe/skills"]' in config_text
        assert 'command = "/tmp/fake-bin/vibe-rag"' in config_text
        assert "source ~/.zprofile" not in config_text
        assert "source ~/.zshrc" not in config_text
        assert "__VIBE_RAG_BIN__" not in config_text
        assert "__VIBE_RAG_BIN_SHELL__" not in config_text
        assert "[[hooks.SessionStart]]" in config_text
        assert 'hook-session-start --format vibe' in config_text


def test_cli_init_does_not_install_agent_profiles():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0
        assert "Installed agents to ~/.vibe/agents/" not in result.output
        assert "vibe --agent builder" not in result.output
        assert "start Claude Code, Codex, or Vibe in this repo" in result.output


def test_cli_init_writes_memory_first_agents_guide():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0

        agents_text = Path("demo/AGENTS.md").read_text()
        assert "memory_load_session_context" in agents_text
        assert "memory_index_project" in agents_text
        assert "memory_search" in agents_text
        assert "memory_search_memory" in agents_text
        assert "memory_project_status" in agents_text
        assert "Use the memory MCP tools first" in agents_text


def test_cli_init_installs_semantic_repo_search_skill():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0

        skill_text = Path("demo/.vibe/skills/semantic-repo-search/SKILL.md").read_text()
        assert "name: semantic-repo-search" in skill_text
        assert "memory_load_session_context" in skill_text
        assert "memory_search" in skill_text
        assert 'scope: "code"' in skill_text
        assert "memory_project_status" in skill_text
        assert "Prefer memory tools over `grep`" in skill_text


def test_cli_init_installs_codex_and_claude_scaffolding(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vibe_rag.cli._current_vibe_rag_binary", lambda: "/tmp/fake-bin/vibe-rag")
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0

        codex_config = Path("demo/.codex/config.toml").read_text()
        codex_hooks = Path("demo/.codex/hooks.json").read_text()
        claude_settings = Path("demo/.claude/settings.json").read_text()
        gemini_settings = Path("demo/.gemini/settings.json").read_text()
        claude_mcp = Path("demo/.mcp.json").read_text()

        assert "__VIBE_RAG_BIN__" not in codex_config
        assert "__VIBE_RAG_BIN_SHELL__" not in codex_hooks
        assert "__VIBE_RAG_BIN__" not in codex_hooks
        assert "__VIBE_RAG_BIN__" not in claude_settings
        assert "__VIBE_RAG_BIN__" not in gemini_settings
        assert "__VIBE_RAG_BIN__" not in claude_mcp

        assert "suppress_unstable_features_warning = true" in codex_config
        assert "codex_hooks = true" in codex_config
        assert 'command = "/tmp/fake-bin/vibe-rag"' in codex_config
        assert 'args = ["serve"]' in codex_config
        assert "/tmp/fake-bin/vibe-rag hook-session-start --format codex" in codex_hooks
        assert "/tmp/fake-bin/vibe-rag hook-session-start --format claude" in claude_settings
        assert '"/tmp/fake-bin/vibe-rag"' in claude_mcp
        assert "/tmp/fake-bin/vibe-rag hook-session-start --format gemini" in gemini_settings
        assert '"mcpServers"' in gemini_settings
        assert '"mcpServers"' in claude_mcp
        assert '"vibe-rag"' in claude_mcp


def test_cli_init_only_rewrites_generated_files(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr("vibe_rag.cli._current_vibe_rag_binary", lambda: "/tmp/fake bin/vibe-rag")
    monkeypatch.setattr("vibe_rag.cli.shutil.which", lambda name: "/usr/bin/git" if name == "git" else None)
    monkeypatch.setattr("vibe_rag.cli.subprocess.run", lambda *args, **kwargs: None)

    with runner.isolated_filesystem():
        Path("demo").mkdir()
        Path("demo/.vibe").mkdir()
        Path("demo/notes.txt").write_text("leave __VIBE_RAG_BIN__ untouched")

        result = runner.invoke(main, ["init", "demo"], input="y\n")

        assert result.exit_code == 0
        assert Path("demo/notes.txt").read_text() == "leave __VIBE_RAG_BIN__ untouched"
        assert 'command = "/tmp/fake bin/vibe-rag"' in Path("demo/.codex/config.toml").read_text()
        assert "'/tmp/fake bin/vibe-rag' hook-session-start --format codex" in Path("demo/.codex/hooks.json").read_text()


def test_cli_init_runs_git_init_when_missing(monkeypatch):
    runner = CliRunner()
    calls = []

    def fake_which(name):
        if name == "vibe-rag":
            return "/tmp/fake-bin/vibe-rag"
        if name == "git":
            return "/usr/bin/git"
        return None

    def fake_run(args, cwd=None, check=False, stdout=None, stderr=None):
        calls.append(
            {
                "args": args,
                "cwd": cwd,
                "check": check,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        return None

    monkeypatch.setattr("vibe_rag.cli.shutil.which", fake_which)
    monkeypatch.setattr("vibe_rag.cli.subprocess.run", fake_run)

    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0
        assert calls == [
            {
                "args": ["/usr/bin/git", "init"],
                "cwd": Path.cwd() / "demo",
                "check": False,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
        ]


def test_cli_hook_session_start_renders_codex_output(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": True,
            "project_id": "demo-project",
            "stale": {"warnings": [{"kind": "git_head_changed", "detail": "git HEAD changed since index"}]},
            "memories": [{"id": "1", "summary": "Use memory tools first"}],
            "code": [{"file_path": "src/app.py", "start_line": 12, "content": "def run(): pass", "indexed_at": "2026-03-22T00:00:00Z"}],
            "docs": [{"file_path": "README.md", "preview": "Quick start for the project", "indexed_at": "2026-03-22T00:00:00Z"}],
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "codex"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["suppressOutput"] is True
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert payload["hookSpecificOutput"]["additionalContext"].startswith(
        "vibe-rag context for project `demo-project`"
    )
    assert "Index warnings:" in payload["hookSpecificOutput"]["additionalContext"]


def test_cli_hook_session_start_uses_briefing_format(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": True,
            "project_id": "test-project",
            "briefing": "vibe-rag | test-project | main | clean\n\n! No code index",
            "memories": [],
            "code": [],
            "docs": [],
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "claude"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "vibe-rag | test-project | main | clean" in context
    assert "No code index" in context


def test_cli_hook_session_start_renders_claude_output(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": True,
            "project_id": "demo-project",
            "memories": [],
            "code": [],
            "docs": [],
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "claude"],
        input='{"source":"resume"}',
    )

    assert result.exit_code == 0
    assert '"hookEventName": "SessionStart"' in result.output
    assert "demo-project" in result.output


def test_cli_hook_session_start_renders_gemini_output(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": True,
            "project_id": "demo-project",
            "memories": [],
            "code": [],
            "docs": [],
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "gemini"],
        input='{"source":"clear"}',
    )

    assert result.exit_code == 0
    assert '"additionalContext": "vibe-rag context for project `demo-project`' in result.output


def test_cli_hook_session_start_renders_vibe_output(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": True,
            "project_id": "demo-project",
            "memories": [],
            "code": [],
            "docs": [],
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "vibe"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    assert '"hookEventName": "SessionStart"' in result.output
    assert '"additionalContext": "vibe-rag context for project `demo-project`' in result.output


def test_cli_hook_session_start_categorizes_failures(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": False,
            "error": "Embedding failed: OpenRouter not reachable",
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "codex"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    assert "embedding failure" in result.output


def test_cli_hook_session_start_extracts_structured_error_message(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": False,
            "error": {
                "code": "no_memories",
                "message": "no memories stored yet",
                "details": {},
            },
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "codex"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    assert "empty retrieval" in result.output
    assert "no memories stored yet" in result.output
    assert "'code': 'no_memories'" not in result.output


def test_cli_hook_session_start_handles_bootstrap_exceptions(monkeypatch):
    runner = CliRunner()

    def raise_failure(**kwargs):
        raise RuntimeError("sqlite is locked")

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        raise_failure,
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "codex"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    assert "sqlite is locked" in result.output


def test_cli_hook_session_start_handles_invalid_json_input():
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "codex"],
        input='{"source"',
    )

    assert result.exit_code == 0
    assert "invalid hook input" in result.output
