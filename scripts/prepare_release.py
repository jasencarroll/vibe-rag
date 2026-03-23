#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys


VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _replace_once(content: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, content, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected to update {label} exactly once")
    return updated


def _current_version(pyproject_text: str) -> str:
    match = re.search(r'(?m)^version = "([^"]+)"$', pyproject_text)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return match.group(1)


def _update_uv_lock(uv_lock_text: str, version: str) -> str:
    pattern = r'(\[\[package\]\]\nname = "vibe-rag"\nversion = ")([^"]+)(")'
    updated, count = re.subn(pattern, rf"\g<1>{version}\3", uv_lock_text, count=1)
    if count != 1:
        raise RuntimeError("Expected to update vibe-rag package version in uv.lock exactly once")
    return updated


def _release_section(changelog_text: str, version: str, release_date: str) -> tuple[str, str]:
    if f"## [{version}]" in changelog_text:
        raise RuntimeError(f"CHANGELOG already contains version {version}")

    marker = "## [Unreleased]"
    if marker not in changelog_text:
        raise RuntimeError("CHANGELOG is missing an [Unreleased] section")

    after_unreleased = changelog_text.split(marker, 1)[1]
    next_header_index = after_unreleased.find("\n## [")
    if next_header_index == -1:
        unreleased_body = after_unreleased.strip()
        remainder = ""
    else:
        unreleased_body = after_unreleased[:next_header_index].strip()
        remainder = after_unreleased[next_header_index + 1 :]

    if not unreleased_body:
        raise RuntimeError("CHANGELOG [Unreleased] section is empty")

    release_header = f"## [{version}] - {release_date}"
    new_changelog = changelog_text.replace(
        f"{marker}\n\n{unreleased_body}",
        f"{marker}\n\n{release_header}\n\n{unreleased_body}",
        1,
    )
    if remainder and not new_changelog.endswith(remainder):
        # Defensive check in case the simple replacement stops matching the existing shape.
        raise RuntimeError("Unexpected CHANGELOG shape while promoting [Unreleased]")
    return new_changelog, unreleased_body.rstrip() + "\n"


def prepare_release(root: pathlib.Path, version: str, release_date: str) -> str:
    if not VERSION_RE.fullmatch(version):
        raise RuntimeError(f"Invalid version '{version}'. Expected X.Y.Z")

    pyproject_path = root / "pyproject.toml"
    version_paths = {
        "pyproject.toml": pyproject_path,
        "src/vibe_rag/__init__.py": root / "src/vibe_rag/__init__.py",
        "tests/test_cli.py": root / "tests/test_cli.py",
        "README.md": root / "README.md",
        "uv.lock": root / "uv.lock",
        "CHANGELOG.md": root / "CHANGELOG.md",
    }

    pyproject_text = _read(pyproject_path)
    old_version = _current_version(pyproject_text)
    if old_version == version:
        raise RuntimeError(f"Version {version} is already current")

    updated_pyproject = _replace_once(
        pyproject_text,
        r'(?m)^version = "[^"]+"$',
        f'version = "{version}"',
        "pyproject.toml version",
    )
    _write(pyproject_path, updated_pyproject)

    init_path = version_paths["src/vibe_rag/__init__.py"]
    _write(
        init_path,
        _replace_once(
            _read(init_path),
            r'(?m)^__version__ = "[^"]+"$',
            f'__version__ = "{version}"',
            "__init__.py version",
        ),
    )

    test_cli_path = version_paths["tests/test_cli.py"]
    test_cli_text = _read(test_cli_path).replace(old_version, version)
    if test_cli_text == _read(test_cli_path):
        raise RuntimeError("Expected to update tests/test_cli.py version references")
    _write(test_cli_path, test_cli_text)

    readme_path = version_paths["README.md"]
    _write(
        readme_path,
        _replace_once(
            _read(readme_path),
            r"vibe-rag@\d+\.\d+\.\d+",
            f"vibe-rag@{version}",
            "README install version",
        ),
    )

    uv_lock_path = version_paths["uv.lock"]
    _write(uv_lock_path, _update_uv_lock(_read(uv_lock_path), version))

    changelog_path = version_paths["CHANGELOG.md"]
    new_changelog, release_notes = _release_section(_read(changelog_path), version, release_date)
    _write(changelog_path, new_changelog)

    return release_notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a vibe-rag release")
    parser.add_argument("--version", required=True, help="Semantic version to release, e.g. 0.0.20")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Release date in YYYY-MM-DD format")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--notes-out", help="Optional path to write release notes for this version")
    args = parser.parse_args(argv)

    try:
        release_notes = prepare_release(pathlib.Path(args.root).resolve(), args.version, args.date)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.notes_out:
        notes_path = pathlib.Path(args.notes_out)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(release_notes, encoding="utf-8")

    print(f"Prepared release {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
