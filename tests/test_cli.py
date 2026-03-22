from pathlib import Path

from click.testing import CliRunner
from vibe_rag.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.8" in result.output


def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "vibe-rag" in result.output


def test_cli_init_does_not_persist_secrets():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main,
            ["init", "demo"],
            env={
                "MISTRAL_API_KEY": "top-secret-key",
                "DATABASE_URL": "postgresql://user:pass@example.com:5432/vibe_rag",
            },
        )

        assert result.exit_code == 0

        config_text = Path("demo/.vibe/config.toml").read_text()
        assert "top-secret-key" not in config_text
        assert "postgresql://user:pass@example.com:5432/vibe_rag" not in config_text
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
        assert "memory_search_code" in skill_text
        assert "Prefer memory tools over `grep`" in skill_text
