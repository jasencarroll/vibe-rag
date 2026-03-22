from click.testing import CliRunner
from vibe_memory.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.output


def test_cli_status_no_db(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert "no DATABASE_URL" in result.output
