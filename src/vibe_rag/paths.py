"""Shared filesystem paths for vibe-rag-owned state.

Client-specific config files may still live in client-required locations such
as ``.vibe/config.toml`` for Vibe, but vibe-rag's own runtime state lives
under ``.vibe-rag`` in the repo and ``~/.vibe-rag`` at user scope.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_STATE_DIRNAME = ".vibe-rag"
USER_STATE_DIRNAME = ".vibe-rag"


def project_state_dir(project_root: Path | None = None) -> Path:
    """Return the project-local directory for vibe-rag runtime state."""
    return (project_root or Path.cwd()) / PROJECT_STATE_DIRNAME


def project_index_db_path(project_root: Path | None = None) -> Path:
    """Return the default project index DB path."""
    return project_state_dir(project_root) / "index.db"


def user_state_dir(home: Path | None = None) -> Path:
    """Return the user-scoped directory for vibe-rag runtime state."""
    return (home or Path.home()) / USER_STATE_DIRNAME


def user_memory_db_path(home: Path | None = None) -> Path:
    """Return the default user memory DB path."""
    return user_state_dir(home) / "memory.db"


def user_config_path(home: Path | None = None) -> Path:
    """Return the default user-scoped vibe-rag config path."""
    return user_state_dir(home) / "config.toml"
