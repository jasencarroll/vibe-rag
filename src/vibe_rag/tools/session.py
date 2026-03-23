from __future__ import annotations

from pathlib import Path

from vibe_rag.server import _ensure_project_id, mcp
from vibe_rag.tools._helpers import (
    _code_result_payload,
    _doc_result_payload,
    _failure_from_error,
    _memory_payload,
    _validate_query,
)


def _pkg():
    """Late-bound package lookup so monkeypatching ``vibe_rag.tools.X`` works."""
    import vibe_rag.tools as _t
    return _t


@mcp.tool()
def load_session_context(
    task: str,
    refresh_index: bool = False,
    memory_limit: int = 5,
    code_limit: int = 5,
    docs_limit: int = 3,
) -> dict:
    """Bootstrap likely context for a new task by retrieving related memories, code, and docs in one call."""
    error = _validate_query(task)
    if error:
        return _failure_from_error(error, task=task)

    pkg = _pkg()

    payload: dict = {
        "ok": True,
        "task": task,
        "project_id": _ensure_project_id(),
        "index": None,
        "stale": None,
        "pulse": None,
        "narrative": None,
        "hazards": [],
        "live_decisions": [],
        "briefing": "",
        "memories": [],
        "code": [],
        "docs": [],
        "errors": {},
    }

    if refresh_index:
        payload["index"] = pkg.index_project(paths=".")

    payload["pulse"] = pkg._project_pulse(Path.cwd())

    project_db = pkg._get_db()
    stale_state = pkg._stale_state(project_db, Path.cwd(), payload["project_id"])
    payload["stale"] = stale_state

    user_db = pkg._get_user_db()
    payload["narrative"] = pkg._session_narrative(user_db, payload["project_id"])
    payload["hazards"] = pkg._hazard_scan(project_db, Path.cwd(), payload["project_id"], payload["pulse"])
    payload["live_decisions"] = pkg._live_decisions(project_db, user_db, payload["project_id"])

    memory_error, memory_results = pkg._search_memory_results(task, limit=memory_limit)
    if memory_error:
        payload["errors"]["memory"] = memory_error
    else:
        payload["memories"] = [
            _memory_payload(result, current_project_id=payload["project_id"], query=task)
            for result in memory_results
        ]

    code_error, code_results = pkg._search_code_results(task, limit=code_limit)
    if code_error:
        payload["errors"]["code"] = code_error
    else:
        payload["code"] = [_code_result_payload(result, query=task) for result in code_results]

    docs_error, docs_results = pkg._search_docs_results(task, limit=docs_limit)
    if docs_error:
        payload["errors"]["docs"] = docs_error
    else:
        payload["docs"] = [_doc_result_payload(result, query=task) for result in docs_results]

    payload["briefing"] = pkg._format_briefing(
        payload["pulse"],
        payload["narrative"],
        payload["hazards"],
        payload["live_decisions"],
        {
            "memories": payload.get("memories", []),
            "code": payload.get("code", []),
            "docs": payload.get("docs", []),
        },
        payload["project_id"],
    )

    return payload
