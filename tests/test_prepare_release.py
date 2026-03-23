from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_release_updates_versioned_files_and_changelog(tmp_path):
    _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "vibe-rag"\nversion = "0.0.19"\n',
    )
    _write(tmp_path / "src/vibe_rag/__init__.py", '__version__ = "0.0.19"\n')
    _write(
        tmp_path / "tests/test_cli.py",
        'def test_cli_version():\n    assert "0.0.19" in "0.0.19"\n',
    )
    _write(tmp_path / "README.md", "uv tool install vibe-rag@0.0.19\n")
    _write(
        tmp_path / "uv.lock",
        '[[package]]\nname = "vibe-rag"\nversion = "0.0.19"\nsource = { editable = "." }\n',
    )
    _write(
        tmp_path / "CHANGELOG.md",
        "# Changelog\n\n## [Unreleased]\n\n### Added\n- New release pipeline\n\n## [0.0.19] - 2026-03-22\n\n### Fixed\n- Something\n",
    )

    notes_path = tmp_path / "release-notes.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_release.py",
            "--root",
            str(tmp_path),
            "--version",
            "0.0.20",
            "--date",
            "2026-03-23",
            "--notes-out",
            str(notes_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert 'version = "0.0.20"' in (tmp_path / "pyproject.toml").read_text()
    assert '__version__ = "0.0.20"' in (tmp_path / "src/vibe_rag/__init__.py").read_text()
    assert '"0.0.20"' in (tmp_path / "tests/test_cli.py").read_text()
    assert "vibe-rag@0.0.20" in (tmp_path / "README.md").read_text()
    assert 'version = "0.0.20"' in (tmp_path / "uv.lock").read_text()

    changelog = (tmp_path / "CHANGELOG.md").read_text()
    assert "## [Unreleased]\n\n## [0.0.20] - 2026-03-23" in changelog
    assert "### Added\n- New release pipeline" in changelog
    assert notes_path.read_text() == "### Added\n- New release pipeline\n"


def test_prepare_release_requires_unreleased_content(tmp_path):
    _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "vibe-rag"\nversion = "0.0.19"\n',
    )
    _write(tmp_path / "src/vibe_rag/__init__.py", '__version__ = "0.0.19"\n')
    _write(tmp_path / "tests/test_cli.py", 'assert "0.0.19"\n')
    _write(tmp_path / "README.md", "uv tool install vibe-rag@0.0.19\n")
    _write(
        tmp_path / "uv.lock",
        '[[package]]\nname = "vibe-rag"\nversion = "0.0.19"\nsource = { editable = "." }\n',
    )
    _write(
        tmp_path / "CHANGELOG.md",
        "# Changelog\n\n## [Unreleased]\n\n## [0.0.19] - 2026-03-22\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_release.py",
            "--root",
            str(tmp_path),
            "--version",
            "0.0.20",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "CHANGELOG [Unreleased] section is empty" in result.stderr
