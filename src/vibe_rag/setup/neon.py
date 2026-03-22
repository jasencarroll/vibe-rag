from __future__ import annotations
import json
import shutil
import subprocess


def neonctl_available() -> bool:
    return shutil.which("neonctl") is not None


def neon_auth() -> bool:
    result = subprocess.run(["neonctl", "auth"], capture_output=False)
    return result.returncode == 0


def neon_create_project(name: str = "vibe-rag") -> dict:
    result = subprocess.run(
        ["neonctl", "projects", "create", "--name", name, "--output", "json"],
        capture_output=True, text=True,
    )
    result.check_returncode()
    return json.loads(result.stdout)


def neon_get_connection_string(project_id: str) -> str:
    result = subprocess.run(
        ["neonctl", "connection-string", project_id, "--output", "json"],
        capture_output=True, text=True,
    )
    result.check_returncode()
    data = json.loads(result.stdout)
    return data.get("connection_string", data.get("uri", ""))


def neon_enable_pgvector(connection_string: str) -> None:
    subprocess.run(
        ["psql", connection_string, "-c", "CREATE EXTENSION IF NOT EXISTS vector;"],
        capture_output=True,
    )
