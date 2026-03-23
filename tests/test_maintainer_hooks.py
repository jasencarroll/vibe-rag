from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claude_settings_include_pre_and_post_tool_hooks():
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
    hooks = settings["hooks"]

    assert "PreToolUse" in hooks
    assert "PostToolUse" in hooks
    assert hooks["PreToolUse"][0]["matcher"] == "Write|Edit|MultiEdit"
    assert hooks["PostToolUse"][0]["matcher"] == "Write|Edit|MultiEdit"


def test_claude_pretool_guard_blocks_protected_files():
    payload = json.dumps({"tool_input": {"file_path": "uv.lock"}})

    result = subprocess.run(
        ["python3", "scripts/claude_pretool_guard.py"],
        cwd=REPO_ROOT,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "uv.lock" in response["systemMessage"]


def test_claude_pretool_guard_allows_normal_files():
    payload = json.dumps({"tool_input": {"file_path": "src/vibe_rag/tools.py"}})

    result = subprocess.run(
        ["python3", "scripts/claude_pretool_guard.py"],
        cwd=REPO_ROOT,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_claude_posttool_pytest_script_reports_success(monkeypatch):
    module = _load_script_module(REPO_ROOT / "scripts" / "claude_posttool_pytest.py", "claude_posttool_pytest")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="240 passed in 2.25s\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() == 0


def test_claude_posttool_pytest_script_reports_failure(monkeypatch, capsys):
    module = _load_script_module(REPO_ROOT / "scripts" / "claude_posttool_pytest.py", "claude_posttool_pytest_fail")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="FAILED tests/test_tools.py::test_example\n",
            stderr="AssertionError\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main() == 2
    captured = capsys.readouterr()
    assert "PostToolUse auto-test failed" in captured.err
    assert "FAILED tests/test_tools.py::test_example" in captured.err
