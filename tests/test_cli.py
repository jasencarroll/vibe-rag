from pathlib import Path
import os
import subprocess
import sys

from click.testing import CliRunner
from vibe_rag.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.20" in result.output


def test_cli_help_uses_broader_product_framing():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Semantic repo search and coding memory over MCP." in result.output
    assert "for Mistral Vibe" not in result.output


def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "vibe-rag" in result.output


def test_cli_status_uses_env_db_paths_and_dimensions(monkeypatch, tmp_path):
    runner = CliRunner()
    project_db = tmp_path / "project.db"
    user_db = tmp_path / "user.db"
    monkeypatch.setenv("VIBE_RAG_DB", str(project_db))
    monkeypatch.setenv("VIBE_RAG_USER_DB", str(user_db))
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_DIMENSIONS", "1024")

    result = runner.invoke(main, ["status"])

    assert result.exit_code == 0
    assert str(project_db) in result.output
    assert str(user_db) in result.output


def test_cli_reindex_uses_index_project(monkeypatch):
    runner = CliRunner()
    calls = {}

    def fake_index_project(paths):
        calls["paths"] = paths
        return "Indexed 2 files"

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
        return "Indexed current project"

    monkeypatch.setattr("vibe_rag.tools.index_project", fake_index_project)

    result = runner.invoke(main, ["reindex"])

    assert result.exit_code == 0
    assert calls["paths"] == "."
    assert "Indexed current project" in result.output


def test_cli_doctor_defaults_to_ollama(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "ollama",
            "ok": True,
            "detail": "ready (http://localhost:11434)",
            "model": "qwen3-embedding:0.6b",
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (mistral-vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "background and session memory hooks are enabled"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 1024]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted in ~/.vibe/trusted_folders.toml"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted in ~/.codex/config.toml"})
    monkeypatch.setattr("vibe_rag.tools._stale_state", lambda db, root, project_id: {"is_stale": False, "warnings": []})
    monkeypatch.setattr("vibe_rag.cli._recommended_provider", lambda: {"provider": "ollama", "reason": "local embeddings are available"})
    monkeypatch.setattr(
        "vibe_rag.cli._provider_candidates",
        lambda: [
            {"provider": "ollama", "available": True, "detail": "local embeddings via Ollama"},
            {"provider": "mistral", "available": False, "detail": "set MISTRAL_API_KEY"},
        ],
    )
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "Project id:   demo-project" in result.output
    assert "[ok] MCP command" in result.output
    assert "[ok] Vibe CLI" in result.output
    assert "[ok] Vibe hooks" in result.output
    assert "[ok] SessionStart" in result.output
    assert "[ok] Embedding" in result.output
    assert "Recommended:  ollama (local embeddings are available)" in result.output
    assert "ready (http://localhost:11434)" in result.output


def test_cli_doctor_for_ollama_missing_host(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "ollama",
            "ok": False,
            "detail": "Ollama not reachable. Set VIBE_RAG_OLLAMA_HOST or OLLAMA_HOST.",
            "model": "qwen3-embedding:0.6b",
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
            "detail": "Vibe CLI not found. Install the required mistral-vibe fork for first-class session bootstrap.",
        },
    )
    monkeypatch.setattr(
        "vibe_rag.cli._project_vibe_hook_status",
        lambda root: {"ok": False, "warning": True, "detail": "background_mcp_hook is not enabled in .vibe/config.toml"},
    )
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "warn", "detail": "repo not trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "warn", "detail": "repo not trusted"})
    monkeypatch.setattr("vibe_rag.cli._recommended_provider", lambda: {"provider": "mistral", "reason": "mistral credentials are already available"})
    monkeypatch.setattr(
        "vibe_rag.cli._provider_candidates",
        lambda: [
            {"provider": "ollama", "available": False, "detail": "install Ollama to use local embeddings"},
            {"provider": "mistral", "available": True, "detail": "MISTRAL_API_KEY is set"},
        ],
    )
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "[fail] MCP command" in result.output
    assert "[warn] Vibe CLI" in result.output
    assert "[warn] Vibe hooks" in result.output
    assert "[warn] Project DB" in result.output
    assert "Suggested next step:" in result.output
    assert "Ollama fast path:" in result.output
    assert "ollama pull qwen3-embedding:0.6b" in result.output


def test_cli_doctor_fix_invokes_setup_ollama(monkeypatch):
    runner = CliRunner()
    invoked = {}

    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "ollama",
            "ok": False,
            "detail": "Ollama not reachable.",
            "model": "qwen3-embedding:0.6b",
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (mistral-vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "background and session memory hooks are enabled"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._stale_state", lambda db, root, project_id: {"is_stale": False, "warnings": []})
    monkeypatch.setattr("vibe_rag.cli._recommended_provider", lambda: {"provider": "ollama", "reason": "local embeddings are available"})
    monkeypatch.setattr(
        "vibe_rag.cli._provider_candidates",
        lambda: [{"provider": "ollama", "available": True, "detail": "local embeddings via Ollama"}],
    )

    def fake_invoke(command, *args, **kwargs):
        invoked["command"] = command.name
        invoked["kwargs"] = kwargs

    class FakeCtx:
        def invoke(self, command, *args, **kwargs):
            return fake_invoke(command, *args, **kwargs)

    monkeypatch.setattr("vibe_rag.cli.click.get_current_context", lambda *args, **kwargs: FakeCtx())
    result = runner.invoke(main, ["doctor", "--fix"])

    assert result.exit_code == 0
    assert invoked["command"] == "setup-ollama"
    assert invoked["kwargs"]["model"] == "qwen3-embedding:0.6b"


def test_cli_doctor_reports_stale_state(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder.embedding_provider_status",
        lambda: {
            "provider": "ollama",
            "ok": True,
            "detail": "ready (http://localhost:11434)",
            "model": "qwen3-embedding:0.6b",
        },
    )
    monkeypatch.setattr("vibe_rag.cli._project_mcp_command_status", lambda root: {"ok": True, "detail": "uv -> /usr/bin/uv"})
    monkeypatch.setattr("vibe_rag.cli._codex_hook_status", lambda root: {"ok": True, "detail": "hook returned session context"})
    monkeypatch.setattr(
        "vibe_rag.cli._db_readable_status",
        lambda path, label: {"ok": True, "warning": False, "detail": f"{label} DB readable ({path})"},
    )
    monkeypatch.setattr("vibe_rag.cli._vibe_cli_status", lambda: {"ok": True, "warning": False, "detail": "/usr/local/bin/vibe (mistral-vibe 0.1.0)"})
    monkeypatch.setattr("vibe_rag.cli._project_vibe_hook_status", lambda root: {"ok": True, "warning": False, "detail": "background and session memory hooks are enabled"})
    monkeypatch.setattr("vibe_rag.server._ensure_project_id", lambda: "demo-project")
    monkeypatch.setattr("vibe_rag.server._get_embedder", lambda: type("Embedder", (), {"embed_text_sync": lambda self, texts: [[0.0] * 1024]})())
    monkeypatch.setattr("vibe_rag.server._get_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._vibe_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.tools._codex_trust_status", lambda root: {"status": "ok", "detail": "trusted"})
    monkeypatch.setattr("vibe_rag.cli._recommended_provider", lambda: {"provider": "ollama", "reason": "local embeddings are available"})
    monkeypatch.setattr(
        "vibe_rag.cli._provider_candidates",
        lambda: [{"provider": "ollama", "available": True, "detail": "local embeddings via Ollama"}],
    )
    monkeypatch.setattr(
        "vibe_rag.tools._stale_state",
        lambda db, root, project_id: {
            "is_stale": True,
            "warnings": [
                {"kind": "git_head_changed", "detail": "git HEAD changed since index (abc -> def)"},
                {"kind": "indexed_files_missing", "detail": "2 indexed files no longer exist"},
            ],
        },
    )

    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 0
    assert "[warn] Stale state" in result.output
    assert "git HEAD changed since index" in result.output
    assert "Suggested stale fix: vibe-rag reindex" in result.output


def test_cli_init_prints_provider_recommendation(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("vibe_rag.cli._recommended_provider", lambda: {"provider": "mistral", "reason": "mistral credentials are already available"})
    monkeypatch.setattr("vibe_rag.cli._provider_setup_hint", lambda provider: "export MISTRAL_API_KEY=...")

    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

    assert result.exit_code == 0
    assert "Vibe is first-class and expects the mistral-vibe fork" in result.output
    assert "Recommended provider: mistral" in result.output
    assert "Next step: export MISTRAL_API_KEY=..." in result.output


def test_cli_setup_ollama(monkeypatch):
    runner = CliRunner()
    calls = []

    monkeypatch.setattr("vibe_rag.cli.shutil.which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr("vibe_rag.cli._start_ollama_if_needed", lambda: "http://localhost:11434")

    class Result:
        returncode = 0

    def fake_run(args, check=False):
        calls.append(args)
        return Result()

    monkeypatch.setattr("vibe_rag.cli.subprocess.run", fake_run)

    result = runner.invoke(main, ["setup-ollama"])

    assert result.exit_code == 0
    assert calls == [["/usr/local/bin/ollama", "pull", "qwen3-embedding:0.6b"]]
    assert "Pulled qwen3-embedding:0.6b" in result.output


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
    assert "0.0.20" in result.stdout


def test_cli_init_does_not_persist_secrets():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            ["init", "demo"],
            env={
                "MISTRAL_API_KEY": "top-secret-key",
                "VIBE_RAG_USER_DB": "/tmp/vibe-user.db",
            },
        )

        assert result.exit_code == 0

        config_text = Path("demo/.vibe/config.toml").read_text()
        assert "top-secret-key" not in config_text
        assert "/tmp/vibe-user.db" not in config_text
        assert 'skill_paths = [".vibe/skills"]' in config_text
        assert 'command = "/bin/zsh"' in config_text
        assert "source ~/.zprofile" in config_text
        assert "source ~/.zshrc" in config_text
        assert "__VIBE_RAG_BIN__" not in config_text
        assert "[background_mcp_hook]" in config_text
        assert 'tool_name = "memory_load_session_context"' in config_text
        assert "[session_memory_hook]" in config_text


def test_cli_init_does_not_install_agent_profiles():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0
        assert "Installed agents to ~/.vibe/agents/" not in result.output
        assert "vibe --agent builder" not in result.output
        assert "    vibe\n" in result.output


def test_cli_init_writes_memory_first_agents_guide():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0

        agents_text = Path("demo/AGENTS.md").read_text()
        assert "memory_load_session_context" in agents_text
        assert "memory_index_project" in agents_text
        assert "memory_search_code" in agents_text
        assert "memory_search_memory" in agents_text
        assert "Use the memory MCP tools first" in agents_text


def test_cli_init_installs_semantic_repo_search_skill():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0

        skill_text = Path("demo/.vibe/skills/semantic-repo-search/SKILL.md").read_text()
        assert "name: semantic-repo-search" in skill_text
        assert "memory_load_session_context" in skill_text
        assert "memory_search_code" in skill_text
        assert "Prefer memory tools over `grep`" in skill_text


def test_cli_init_installs_codex_and_claude_scaffolding():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init", "demo"])

        assert result.exit_code == 0

        codex_config = Path("demo/.codex/config.toml").read_text()
        codex_hooks = Path("demo/.codex/hooks.json").read_text()
        claude_settings = Path("demo/.claude/settings.json").read_text()
        gemini_settings = Path("demo/.gemini/settings.json").read_text()
        claude_mcp = Path("demo/.mcp.json").read_text()

        assert "__VIBE_RAG_BIN__" not in codex_config
        assert "__VIBE_RAG_BIN__" not in codex_hooks
        assert "__VIBE_RAG_BIN__" not in claude_settings
        assert "__VIBE_RAG_BIN__" not in gemini_settings
        assert "__VIBE_RAG_BIN__" not in claude_mcp

        assert "suppress_unstable_features_warning = true" in codex_config
        assert "codex_hooks = true" in codex_config
        assert 'args = ["serve"]' in codex_config
        assert "hook-session-start --format codex" in codex_hooks
        assert "hook-session-start --format claude" in claude_settings
        assert "hook-session-start --format gemini" in gemini_settings
        assert '"mcpServers"' in gemini_settings
        assert '"mcpServers"' in claude_mcp
        assert '"vibe-rag"' in claude_mcp


def test_cli_init_only_rewrites_generated_files(monkeypatch):
    runner = CliRunner()

    def fake_which(name):
        if name == "vibe-rag":
            return "/tmp/fake-bin/vibe-rag"
        if name == "git":
            return "/usr/bin/git"
        return None

    monkeypatch.setattr("vibe_rag.cli.shutil.which", fake_which)
    monkeypatch.setattr("vibe_rag.cli.subprocess.run", lambda *args, **kwargs: None)

    with runner.isolated_filesystem():
        Path("demo").mkdir()
        Path("demo/.vibe").mkdir()
        Path("demo/notes.txt").write_text("leave __VIBE_RAG_BIN__ untouched")

        result = runner.invoke(main, ["init", "demo"], input="y\n")

        assert result.exit_code == 0
        assert Path("demo/notes.txt").read_text() == "leave __VIBE_RAG_BIN__ untouched"
        assert "/tmp/fake-bin/vibe-rag" in Path("demo/.codex/config.toml").read_text()


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
    assert '"hookEventName": "SessionStart"' in result.output
    assert '"additionalContext": "vibe-rag context for project `demo-project`' in result.output
    assert "Index warnings:" in result.output


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


def test_cli_hook_session_start_categorizes_failures(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "vibe_rag.hook_bridge.load_session_context",
        lambda **kwargs: {
            "ok": False,
            "error": "Embedding failed: Ollama not reachable",
        },
    )

    result = runner.invoke(
        main,
        ["hook-session-start", "--format", "codex"],
        input='{"source":"startup"}',
    )

    assert result.exit_code == 0
    assert "embedding failure" in result.output
