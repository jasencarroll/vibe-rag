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


def _error_category(error: str) -> str:
    lowered = error.lower()
    if "not found" in lowered or "no such file" in lowered:
        return "command not found"
    if "no code index" in lowered or "no docs indexed" in lowered or "no memories stored" in lowered:
        return "empty retrieval"
    if "embedding failed" in lowered or "ollama" in lowered or "api key" in lowered:
        return "embedding failure"
    if "trust" in lowered:
        return "trust/config issue"
    if "db" in lowered or "sqlite" in lowered:
        return "db missing"
    return "unknown error"


def _truncate_context(text: str, limit: int = 1800) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_context(payload: dict[str, Any]) -> str:
    lines: list[str] = []

    project_id = payload.get("project_id")
    if project_id:
        lines.append(f"vibe-rag context for project `{project_id}`")

    stale = payload.get("stale") or {}
    stale_warnings = stale.get("warnings") or []
    if stale_warnings:
        lines.append("Index warnings:")
        for warning in stale_warnings[:2]:
            lines.append(f"- {warning.get('detail')}")

    memories = payload.get("memories") or []
    if memories:
        lines.append("Relevant memories:")
        for memory in memories[:2]:
            summary = memory.get("summary") or memory.get("content") or ""
            memory_id = memory.get("id")
            kind = memory.get("memory_kind") or "note"
            updated_at = ((memory.get("provenance") or {}).get("updated_at")) or memory.get("updated_at")
            suffix = f" ({updated_at})" if updated_at else ""
            lines.append(f"- [{kind} id={memory_id}]{suffix} {_trim_block(summary, 200)}")

    code_results = payload.get("code") or []
    if code_results:
        lines.append("Relevant code:")
        for item in code_results[:2]:
            file_path = item.get("file_path") or "unknown"
            start_line = item.get("start_line") or 1
            content = _trim_block(item.get("content") or "", 240)
            indexed_at = ((item.get("provenance") or {}).get("indexed_at")) or item.get("indexed_at")
            suffix = f" [{indexed_at}]" if indexed_at else ""
            lines.append(f"- {file_path}:{start_line}{suffix} {content}")

    docs_results = payload.get("docs") or []
    if docs_results:
        lines.append("Relevant docs:")
        for item in docs_results[:2]:
            file_path = item.get("file_path") or "unknown"
            preview = _trim_block(item.get("preview") or item.get("content") or "", 220)
            indexed_at = ((item.get("provenance") or {}).get("indexed_at")) or item.get("indexed_at")
            suffix = f" [{indexed_at}]" if indexed_at else ""
            lines.append(f"- {file_path}{suffix} {preview}")

    if len(lines) == 1 and project_id:
        lines.append("No matching memory, code, or docs were returned.")

    return _truncate_context("\n".join(lines))


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
            f"vibe-rag session hook failed ({_error_category(error)}): {error}",
        )

    return _response_for_format(target_format, _format_context(payload), None)


def render_session_start_hook_json(target_format: str, raw_input: str) -> str:
    hook_input = json.loads(raw_input)
    return json.dumps(render_session_start_hook(target_format, hook_input))
