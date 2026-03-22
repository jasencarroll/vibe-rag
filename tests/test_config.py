import os
from pathlib import Path

from vibe_memory.config import resolve_project_id, load_config


def test_resolve_project_id_from_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-app"')
    assert resolve_project_id(tmp_path) == "my-app"


def test_resolve_project_id_from_package_json(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"name": "frontend-app"}')
    assert resolve_project_id(tmp_path) == "frontend-app"


def test_resolve_project_id_fallback_dirname(tmp_path: Path):
    assert resolve_project_id(tmp_path) == tmp_path.name


def test_load_config_reads_env_vars(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-test")
    monkeypatch.setenv("CODESTRAL_API_KEY", "ck-test")
    cfg = load_config()
    assert cfg.database_url == "postgresql://localhost/test"
    assert cfg.mistral_api_key == "mk-test"
    assert cfg.codestral_api_key == "ck-test"


def test_load_config_missing_keys_returns_none(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("CODESTRAL_API_KEY", raising=False)
    cfg = load_config()
    assert cfg.database_url is None
    assert cfg.mistral_api_key is None
    assert cfg.codestral_api_key is None
