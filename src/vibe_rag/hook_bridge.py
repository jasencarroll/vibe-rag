"""Session-start hook renderer for agent CLI integrations.

This module bridges vibe-rag's ``load_session_context`` retrieval with the
JSON response shapes expected by four agent CLI hook protocols:

* **codex** -- OpenAI Codex CLI ``SessionStart`` hook.  Includes
  ``suppressOutput`` (``True`` when no system message is needed) and
  ``hookSpecificOutput.hookEventName``.
* **claude** -- Anthropic Claude Code ``SessionStart`` hook.  Same shape
  as *codex* minus ``suppressOutput``.
* **vibe** -- vibe-tools ``SessionStart`` hook.  Identical shape to
  *claude*.
* **gemini** -- Google Gemini CLI hook.  ``hookSpecificOutput`` carries
  ``additionalContext`` but omits ``hookEventName``.

Entry points
------------
``render_session_start_hook(target_format, hook_input)``
    Accepts a parsed dict of hook input, runs retrieval, returns the
    response dict.
``render_session_start_hook_json(target_format, raw_input)``
    Convenience wrapper that deserializes JSON stdin from the CLI,
    delegates to ``render_session_start_hook``, and returns a JSON string.

Called by
---------
``vibe_rag.cli.hook_session_start`` -- the ``vibe-rag hook-session-start``
CLI command, which reads ``--format`` from the command line and raw JSON
from stdin.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vibe_rag.tools import load_session_context


def _session_task(hook_input: dict[str, Any]) -> str:
    """Derive a natural-language task string from hook input.

    Agent CLIs pass a JSON object on stdin whose shape varies by client.
    This function extracts a usable task description by checking, in order:

    1. An explicit ``"task"`` key (used when the user provides a prompt).
    2. The ``"source"`` key -- ``"resume"`` yields a resume-oriented
       query; any other value (including the default ``"startup"``)
       yields a generic bootstrap query.

    The returned string is forwarded to ``load_session_context`` as the
    semantic search query for memories, code, and docs.
    """
    task = hook_input.get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()
    source = str(hook_input.get("source") or "startup")
    if source == "resume":
        return "Resume prior work in this repo and surface the most relevant memory, code, and docs."
    return "Bootstrap likely context for the current work in this repo."


def _trim_block(text: str, limit: int = 500) -> str:
    """Collapse whitespace and truncate *text* to *limit* characters.

    Used to keep individual memory/code/doc snippets within a reasonable
    size when building the fallback briefing in ``_format_context``.
    """
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def _error_category(error: str) -> str:
    """Classify an error message into a human-readable category label.

    The returned label is embedded in the ``systemMessage`` field so the
    agent can present a concise failure reason to the user.  Categories
    are matched by simple substring checks on the lowercased error text:

    * ``"command not found"`` -- missing binary or file.
    * ``"empty retrieval"`` -- index or memory store is empty.
    * ``"embedding failure"`` -- embedding provider unreachable or
      misconfigured (ollama down, missing API key, etc.).
    * ``"trust/config issue"`` -- trust-related configuration problem.
    * ``"db missing"`` -- SQLite or database-layer error.
    * ``"unknown error"`` -- fallback when no keyword matches.
    """
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
    """Hard-cap the assembled briefing text to *limit* characters.

    Prevents oversized ``additionalContext`` payloads when multiple
    sections contribute content.  Truncates with a trailing ellipsis.
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _payload_error_message(payload: dict[str, Any]) -> str:
    """Extract a human-readable error string from a failed payload.

    ``load_session_context`` may return errors as either a plain string
    or a structured dict (``{"code": ..., "message": ...}``).  This
    helper normalises both shapes into a single message string, falling
    back to ``"unknown error"`` when the payload is missing or empty.
    """
    raw_error = payload.get("error")
    if isinstance(raw_error, dict):
        message = raw_error.get("message")
        if isinstance(message, str) and message.strip():
            return message
    if isinstance(raw_error, str) and raw_error.strip():
        return raw_error
    return "unknown error"


def _format_context(payload: dict[str, Any]) -> str:
    """Build a plain-text briefing from a successful ``load_session_context`` payload.

    This is the **fallback** formatter used when the payload does not
    include a pre-built ``briefing`` key (older server versions or edge
    cases).  It assembles sections in this order:

    1. Project ID header.
    2. Index staleness warnings (up to 2).
    3. Relevant memories (up to 2) -- shows kind, id, timestamp, and a
       trimmed summary.
    4. Relevant code chunks (up to 2) -- shows file path, start line,
       index timestamp, and a trimmed snippet.
    5. Relevant doc chunks (up to 2) -- shows file path, index
       timestamp, and a trimmed preview.

    If only the project ID header was emitted (no retrieval results), a
    "No matching memory, code, or docs" line is appended.

    The assembled text is hard-capped to 1800 characters via
    ``_truncate_context``.
    """
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
    """Wrap retrieval output into the JSON shape expected by *target_format*.

    Each agent CLI defines its own hook response contract.  This function
    adapts a common (additional_context, system_message) pair into the
    correct envelope:

    **codex**::

        {
            "systemMessage": "...",          # present only on error
            "suppressOutput": true|false,    # True when no systemMessage
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "..."
            }
        }

    **claude** / **vibe**::

        {
            "systemMessage": "...",          # present only on error
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "..."
            }
        }

    **gemini**::

        {
            "systemMessage": "...",          # present only on error
            "hookSpecificOutput": {
                "additionalContext": "..."
            }
        }

    Raises ``ValueError`` for unrecognised formats (the CLI constrains
    input via ``click.Choice``, so this is a defensive guard).
    """
    response: dict[str, Any] = {}
    if system_message:
        response["systemMessage"] = system_message

    if target_format == "codex":
        response["suppressOutput"] = system_message is None
        response["hookSpecificOutput"] = {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
        return response

    if target_format in {"claude", "vibe"}:
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
    """Main entry point: produce a hook response dict for *target_format*.

    Orchestration flow:

    1. Silence noisy ``httpx`` debug logs.
    2. Derive a task description from *hook_input* via ``_session_task``.
    3. Call ``load_session_context`` with tight limits (2 each for
       memories, code, and docs) to keep the payload compact.
    4. On success, prefer the server-generated ``briefing`` field; fall
       back to ``_format_context`` if absent.
    5. On failure (exception or ``ok=False``), produce an error response
       with a categorised system message.

    Args:
        target_format: One of ``"codex"``, ``"claude"``, ``"gemini"``,
            or ``"vibe"``.
        hook_input: Parsed JSON from stdin.  Expected keys:
            ``"task"`` (optional str), ``"source"`` (optional str --
            ``"startup"``, ``"resume"``, etc.).

    Returns:
        dict suitable for JSON-serialisation and printing to stdout.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        payload = load_session_context(
            task=_session_task(hook_input),
            refresh_index=False,
            memory_limit=2,
            code_limit=2,
            docs_limit=2,
        )
    except Exception as exc:
        error = str(exc) or "unknown error"
        return _response_for_format(
            target_format,
            "vibe-rag session bootstrap did not return usable context.",
            f"vibe-rag session hook failed ({_error_category(error)}): {error}",
        )

    if not isinstance(payload, dict) or not payload.get("ok"):
        error = _payload_error_message(payload) if isinstance(payload, dict) else "unknown error"
        return _response_for_format(
            target_format,
            "vibe-rag session bootstrap did not return usable context.",
            f"vibe-rag session hook failed ({_error_category(error)}): {error}",
        )

    briefing = payload.get("briefing") or _format_context(payload)
    return _response_for_format(target_format, briefing, None)


def render_session_start_hook_json(target_format: str, raw_input: str) -> str:
    """JSON convenience wrapper around ``render_session_start_hook``.

    Deserialises *raw_input* (the raw stdin string from the CLI), delegates
    to ``render_session_start_hook``, and returns the result as a JSON
    string.  If *raw_input* is not valid JSON, returns an error response
    without raising.

    This is the function called by ``vibe_rag.cli.hook_session_start``.
    """
    try:
        hook_input = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        return json.dumps(
            _response_for_format(
                target_format,
                "vibe-rag session bootstrap did not return usable context.",
                f"vibe-rag session hook failed (unknown error): invalid hook input: {exc.msg}",
            )
        )
    return json.dumps(render_session_start_hook(target_format, hook_input))
