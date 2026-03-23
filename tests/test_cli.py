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
    assert "0.0.18" in result.output


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
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "Provider:    ollama" in result.output
    assert "Model:       qwen3-embedding:0.6b" in result.output
    assert "Status:      ready (http://localhost:11434)" in result.output


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
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "Provider:    ollama" in result.output
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
    assert "0.0.18" in result.stdout


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
        assert "env =" not in config_text
        assert 'skill_paths = [".vibe/skills"]' in config_text


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
            "memories": [{"id": "1", "summary": "Use memory tools first"}],
            "code": [{"file_path": "src/app.py", "start_line": 12, "content": "def run(): pass"}],
            "docs": [{"file_path": "README.md", "preview": "Quick start for the project"}],
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
