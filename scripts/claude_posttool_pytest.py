#!/usr/bin/env python3
"""Claude Code PostToolUse hook that runs the test suite after file edits.

Triggered after Write/Edit/MultiEdit tool calls, this hook runs
``uv run pytest tests/ -x -q --tb=short`` from the repo root and reports
whether the suite passed or failed.  Output is trimmed to the last 40
non-empty lines to keep the feedback concise.  A non-zero exit (code 2)
signals to Claude Code that the edit may have introduced a regression.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMAND = ["uv", "run", "pytest", "tests/", "-x", "-q", "--tb=short"]


def _trimmed_output(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= 40:
        return "\n".join(lines)
    return "\n".join(lines[-40:])


def main() -> int:
    result = subprocess.run(
        COMMAND,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = _trimmed_output("\n".join(part for part in (result.stdout, result.stderr) if part))
    if result.returncode == 0:
        if combined:
            print(f"PostToolUse auto-test passed:\n{combined}")
        else:
            print("PostToolUse auto-test passed.")
        return 0

    message = combined or "pytest failed with no output"
    print(f"PostToolUse auto-test failed:\n{message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
