from __future__ import annotations

import json
import logging
from typing import Any

from vibe_rag.tools import load_session_context


def _session_task(source: str) -> str:
    if source == "resume":
        return "Resume prior work in this repo and surface the most relevant memory, code, and docs."
    return "Bootstrap likely context for the current work in this repo."


def _trim_block(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def _format_context(payload: dict[str, Any]) -> str:
    lines: list[str] = []

    project_id = payload.get("project_id")
    if project_id:
        lines.append(f"vibe-rag context for project `{project_id}`")

    memories = payload.get("memories") or []
    if memories:
        lines.append("Relevant memories:")
        for memory in memories[:2]:
            summary = memory.get("summary") or memory.get("content") or ""
            memory_id = memory.get("id")
            lines.append(f"- [id={memory_id}] {_trim_block(summary, 220)}")

    code_results = payload.get("code") or []
    if code_results:
        lines.append("Relevant code:")
        for item in code_results[:2]:
            file_path = item.get("file_path") or "unknown"
            start_line = item.get("start_line") or 1
            content = _trim_block(item.get("content") or "", 240)
            lines.append(f"- {file_path}:{start_line} {content}")

    docs_results = payload.get("docs") or []
    if docs_results:
        lines.append("Relevant docs:")
        for item in docs_results[:2]:
            file_path = item.get("file_path") or "unknown"
            preview = _trim_block(item.get("preview") or item.get("content") or "", 220)
            lines.append(f"- {file_path} {preview}")

    if len(lines) == 1 and project_id:
        lines.append("No matching memory, code, or docs were returned.")

    return "\n".join(lines)


def _response_for_format(target_format: str, additional_context: str, system_message: str | None) -> dict[str, Any]:
    response: dict[str, Any] = {}
    if system_message:
        response["systemMessage"] = system_message

    if target_format in {"codex", "claude"}:
        response["hookSpecificOutput"] = {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
        return response

    if target_format == "gemini":
        response["hookSpecificOutput"] = {
            "additionalContext": additional_context,
        }
        return response

    raise ValueError(f"unsupported hook format: {target_format}")


def render_session_start_hook(target_format: str, hook_input: dict[str, Any]) -> dict[str, Any]:
    logging.getLogger("httpx").setLevel(logging.WARNING)

    payload = load_session_context(
        task=_session_task(str(hook_input.get("source") or "startup")),
        refresh_index=False,
        memory_limit=2,
        code_limit=2,
        docs_limit=2,
    )

    if not isinstance(payload, dict) or not payload.get("ok"):
        error = "unknown error"
        if isinstance(payload, dict):
            error = str(payload.get("error") or error)
        return _response_for_format(
            target_format,
            "vibe-rag session bootstrap did not return usable context.",
            f"vibe-rag session hook failed: {error}",
        )

    return _response_for_format(target_format, _format_context(payload), None)


def render_session_start_hook_json(target_format: str, raw_input: str) -> str:
    hook_input = json.loads(raw_input)
    return json.dumps(render_session_start_hook(target_format, hook_input))
