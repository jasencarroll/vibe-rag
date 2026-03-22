from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class Config:
    database_url: str | None
    mistral_api_key: str | None
    codestral_api_key: str | None
    session_log_dir: Path
    project_id: str | None = None


def resolve_project_id(cwd: Path) -> str:
    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        name = data.get("project", {}).get("name")
        if name:
            return name

    pkg_json = cwd / "package.json"
    if pkg_json.exists():
        data = json.loads(pkg_json.read_text())
        name = data.get("name")
        if name:
            return name

    cargo = cwd / "Cargo.toml"
    if cargo.exists():
        with open(cargo, "rb") as f:
            data = tomllib.load(f)
        name = data.get("package", {}).get("name")
        if name:
            return name

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).name
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return cwd.name


def load_config(cwd: Path | None = None) -> Config:
    return Config(
        database_url=os.environ.get("DATABASE_URL"),
        mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
        codestral_api_key=os.environ.get("CODESTRAL_API_KEY"),
        session_log_dir=Path(os.environ.get(
            "VIBE_SESSION_LOG_DIR",
            Path.home() / ".vibe" / "logs" / "session",
        )),
        project_id=resolve_project_id(cwd or Path.cwd()),
    )
