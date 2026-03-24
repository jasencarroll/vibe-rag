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
    """Return the ``vibe_rag.tools`` module via late import.

    This indirection lets tests monkeypatch names on the package (e.g.
    ``vibe_rag.tools._project_pulse``) and have session.py pick up the
    patched version at call time.
    """
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
    """Bootstrap context for a new task.

    Call at session start or when switching tasks.  Retrieves related memories,
    code, and docs in one call, plus project health, activity pulse, hazards,
    live decisions, and a human-readable briefing.

    Triggers lazy initialization of project DB, user DB, and embedding provider
    on first call.

    Args:
        task: Natural-language description of the current task (used as the
            semantic search query for memories, code, and docs).
        refresh_index: If ``True``, re-indexes the project before searching.
            Default ``False``.
        memory_limit: Max memory results to return.  Default 5.
        code_limit: Max code chunk results to return.  Default 5.
        docs_limit: Max doc chunk results to return.  Default 3.

    Returns:
        dict -- On validation failure (empty or too-long *task*) returns
        ``{"ok": False, "error": {"code": str, "message": str, ...}}``.
        On success returns a dict with every key listed below.

        ok (bool):
            Always ``True`` on success.
        task (str):
            Echo of the *task* argument.
        project_id (str):
            Deterministic project identifier derived from the working
            directory.
        index (dict | None):
            Result of ``index_project()`` when *refresh_index* is
            ``True``; ``None`` otherwise.  The success dict contains
            ``ok`` (bool), ``summary`` (str), ``full_rebuild`` (bool),
            ``project_id`` (str), ``project_root`` (str),
            ``elapsed_seconds`` (float), ``counts`` (dict with
            ``code_files``, ``doc_files``, ``code_chunks``,
            ``doc_chunks``, ``code_unchanged``, ``doc_unchanged``,
            ``code_skipped``, ``doc_skipped``, ``indexed_code_chunks``,
            ``indexed_doc_chunks``), and ``warnings`` (list[dict]).
        stale (dict):
            Index-staleness report.  Keys: ``is_stale`` (bool),
            ``is_incompatible`` (bool), ``warnings`` (list[dict] each
            with ``kind`` and ``detail``), ``metadata`` (dict | None),
            ``current_profile`` (dict | None), ``indexed_profile``
            (dict | None).
        pulse (dict):
            Git activity snapshot.  Keys: ``branch`` (str | None),
            ``is_default_branch`` (bool | None), ``default_branch``
            (str | None), ``workspace`` (dict with ``modified``
            (list[str]), ``staged`` (list[str]), ``untracked``
            (list[str]), ``is_clean`` (bool)), ``recent_commits``
            (list[dict] each with ``sha`` (str) and ``message`` (str)).
            When on a non-default branch, ``ahead`` (int) and
            ``behind`` (int) may also be present.
        narrative (str | None):
            Human-readable prose recap of recent session memories for
            the project, or ``None`` when no session summaries exist.
        hazards (list[dict]):
            Project hazards sorted errors-first.  Each dict has
            ``level`` (``"error"`` | ``"warning"``), ``category``
            (str, e.g. ``"no_index"``, ``"stale_index"``,
            ``"uncommitted_work"``, ``"provider_unavailable"``,
            ``"incompatible_index"``, ``"cleanup_pressure"``), and
            ``message`` (str).
        live_decisions (list[dict]):
            Recent decision/constraint memories still in effect.  Each
            entry follows the ``MemoryPayload`` shape (see *memories*)
            but with the ``score`` key removed.
        briefing (str):
            Pre-formatted plain-text briefing (max 6 000 chars)
            assembled from pulse, narrative, hazards, live decisions,
            and task-related search hits.  Produced by
            ``_format_briefing()``.
        memories (list[dict]):
            Semantic search hits from user + project memory stores.
            Each dict is a ``MemoryPayload``: ``id`` (int),
            ``source_db`` (``"project"`` | ``"user"`` | None),
            ``summary`` (str), ``content`` (str), ``score`` (float),
            ``project_id`` (str | None), ``memory_kind`` (str -- one of
            ``"note"``, ``"decision"``, ``"constraint"``, ``"todo"``,
            ``"summary"``, ``"fact"``), ``tags`` (list[str]),
            ``created_at`` (str | None), ``updated_at`` (str | None),
            ``source_session_id`` (str | None), ``source_message_id``
            (str | None), ``supersedes`` (int | None),
            ``superseded_by`` (int | None), ``is_superseded`` (bool),
            ``is_stale`` (bool), ``stale_reasons`` (list[str]),
            ``metadata`` (dict), ``provenance`` (dict with
            ``capture_kind`` (str), ``source_type`` (str),
            ``is_current_project`` (bool)), ``match_reason`` (str).
            Optional: ``thread_id`` (str | None), ``thread_title``
            (str | None).
        code (list[dict]):
            Semantic search hits from the code index.  Each dict is a
            ``CodeSearchResult``: ``file_path`` (str), ``start_line``
            (int), ``end_line`` (int), ``content`` (str), ``language``
            (str | None), ``symbol`` (str | None), ``indexed_at``
            (str | None), ``rank_score`` (float), ``match_sources``
            (list[str]), ``provenance`` (dict with ``source``
            (``"project-index"``), ``indexed_at`` (str | None)),
            ``match_reason`` (str).
        docs (list[dict]):
            Semantic search hits from the docs index.  Each dict is a
            ``DocSearchResult``: ``file_path`` (str), ``chunk_index``
            (int), ``content`` (str), ``preview`` (str -- first 160
            chars, newlines collapsed), ``indexed_at`` (str | None),
            ``rank_score`` (float), ``match_sources`` (list[str]),
            ``provenance`` (dict with ``source``
            (``"project-index"``), ``indexed_at`` (str | None)),
            ``match_reason`` (str).
        errors (dict):
            Per-subsystem error details.  Possible keys: ``"memory"``,
            ``"code"``, ``"docs"``.  Each value is a ``ToolError`` dict
            with ``code`` (str) and ``message`` (str).  Empty dict when
            all retrievals succeed.
    """
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
