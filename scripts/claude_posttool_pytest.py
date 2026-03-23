#!/usr/bin/env python3
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
