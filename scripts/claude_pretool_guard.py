#!/usr/bin/env python3
"""Claude Code PreToolUse hook that guards protected files from direct edits.

Reads the hook payload from stdin (JSON with tool_input containing file paths)
and denies Write/Edit/MultiEdit operations targeting files in PROTECTED_FILES
(e.g. .env, uv.lock, CHANGELOG.md).  Returns a JSON response with either an
"allow" or "deny" permission decision.  This prevents accidental overwrites of
files that should only be modified by dedicated scripts or workflows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROTECTED_FILES = {
    ".env",
    ".env.local",
    "uv.lock",
    "CHANGELOG.md",
}


def _tool_paths(payload: dict) -> list[Path]:
    tool_input = payload.get("tool_input") or {}
    candidates: list[str] = []
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    if isinstance(tool_input.get("files"), list):
        for value in tool_input["files"]:
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    return [Path(value) for value in candidates]


def _is_protected(path: Path) -> bool:
    normalized = path.as_posix()
    return path.name in PROTECTED_FILES or normalized in PROTECTED_FILES


def _response(decision: str, *, message: str | None = None) -> dict:
    payload = {"hookSpecificOutput": {"permissionDecision": decision}}
    if message:
        payload["systemMessage"] = message
    return payload


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print(json.dumps(_response("allow")), end="")
        return 0

    for path in _tool_paths(payload):
        if _is_protected(path):
            print(
                json.dumps(
                    _response(
                        "deny",
                        message=f"Direct edits to {path.as_posix()} are blocked by project policy.",
                    )
                ),
                end="",
            )
            return 0

    print(json.dumps(_response("allow")), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
