from __future__ import annotations

from pathlib import Path
from typing import Any

from vibe_rag.server import _ensure_project_id, _get_db, _get_user_db, mcp
from vibe_rag.tools._helpers import (
    _all_memory_payloads as _all_memory_payloads_impl,
    _count_by,
    _delete_memory_by_source_db as _delete_memory_by_source_db_impl,
    _duplicate_auto_memory_groups,
    _failure,
    _index_metadata,
    _memory_cleanup_candidates,
    _stale_state,
    _success,
)


def _resolve_all_memory_payloads():
    """Allow monkeypatching ``vibe_rag.tools._all_memory_payloads`` in tests."""
    import vibe_rag.tools as _pkg
    return getattr(_pkg, "_all_memory_payloads", _all_memory_payloads_impl)()


def _resolve_delete_memory_by_source_db(source_db, memory_id):
    """Allow monkeypatching ``vibe_rag.tools._delete_memory_by_source_db`` in tests."""
    import vibe_rag.tools as _pkg
    fn = getattr(_pkg, "_delete_memory_by_source_db", _delete_memory_by_source_db_impl)
    return fn(source_db, memory_id)


def _memory_health_summary() -> dict:
    """Build a concise memory-health dashboard section for project_status."""
    from vibe_rag.tools._helpers import _all_memory_payloads, _memory_cleanup_candidates, _duplicate_auto_memory_groups

    payloads = _all_memory_payloads()
    stale = [item for item in payloads if item.get("is_stale") is True]
    superseded = [item for item in payloads if item.get("is_superseded") is True]
    duplicate_groups = _duplicate_auto_memory_groups(payloads)
    # Use a wider pool for reason analysis, then trim for display.
    cleanup_candidates = _memory_cleanup_candidates(limit=10)

    capture_kind_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    cleanup_reason_counts: dict[str, int] = {}
    for item in payloads:
        provenance = item.get("provenance", {})
        capture_kind = str(provenance.get("capture_kind") or "unknown")
        source_type = str(provenance.get("source_type") or "unknown")
        capture_kind_counts[capture_kind] = capture_kind_counts.get(capture_kind, 0) + 1
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
    for candidate in cleanup_candidates:
        for reason in candidate.get("cleanup_reasons") or []:
            cleanup_reason_counts[reason] = cleanup_reason_counts.get(reason, 0) + 1

    recommended_actions: list[str] = []
    if stale:
        recommended_actions.append("Supersede or delete stale cross-project memories that no longer apply.")
    if superseded:
        recommended_actions.append("Prune superseded memories or confirm the replacement memory is still current.")
    if cleanup_reason_counts.get("freeform_note", 0) or cleanup_reason_counts.get("freeform_capture", 0):
        recommended_actions.append("Convert recurring freeform notes into structured memories with decision/constraint/todo kinds.")
    if cleanup_reason_counts.get("low_signal_auto_memory", 0):
        recommended_actions.append("Trim or suppress low-signal auto session summaries like greetings and one-turn rollups.")
    if cleanup_reason_counts.get("short_summary", 0):
        recommended_actions.append("Rewrite weak short summaries so future retrieval has better anchors.")
    if duplicate_groups:
        recommended_actions.append("Prune duplicate auto-captured session memories or keep only the newest copy per repeated probe.")
    if not recommended_actions:
        recommended_actions.append("Memory quality looks healthy; keep superseding stale decisions instead of appending duplicates.")

    return {
        "summary": {
            "total_memories": len(payloads),
            "stale_memories": len(stale),
            "superseded_memories": len(superseded),
            "duplicate_auto_memory_groups": len(duplicate_groups),
        },
        "top_cleanup_candidates": cleanup_candidates[:3],
        "recommended_actions": recommended_actions[:3],
        "by_capture_kind": capture_kind_counts,
        "by_source_type": source_type_counts,
    }


@mcp.tool()
def project_status(include_memory_health: bool = True) -> dict:
    """Dashboard view: index counts, staleness, language breakdown, and memory health including quality metrics, cleanup candidates, and recommendations. Use include_memory_health=False for a lighter response."""
    db = _get_db()
    metadata_state = _index_metadata(db)
    stale = _stale_state(db, Path.cwd(), _ensure_project_id())
    language_stats = db.language_stats()
    cleanup_candidates = _memory_cleanup_candidates(limit=3)

    status: dict[str, Any] = {
        "counts": {
            "code_chunks": db.code_chunk_count(),
            "doc_chunks": db.doc_count(),
            "project_memories": db.memory_count(),
            "user_memories": _get_user_db().memory_count(),
        },
        "metadata": metadata_state,
        "stale": stale,
        "language_stats": language_stats,
        "cleanup_candidates": cleanup_candidates,
    }

    if include_memory_health:
        status["memory_health"] = _memory_health_summary()

    return _success(
        project_id=_ensure_project_id(),
        status=status,
    )


@mcp.tool()
def cleanup_duplicate_auto_memories(limit: int = 20, apply: bool = False) -> dict:
    """Find and optionally prune duplicate auto-captured session memories. apply=False (default) to preview, apply=True to delete duplicates keeping newest."""
    if limit < 1:
        return _failure("invalid_limit", "limit must be at least 1", limit=limit)

    payloads = _resolve_all_memory_payloads()
    groups = _duplicate_auto_memory_groups(payloads)[:limit]
    source_db_by_id = {
        int(item.get("id")): str(item.get("source_db") or "user")
        for item in payloads
        if item.get("id") is not None
    }
    results: list[dict] = []
    deleted_total = 0
    for group in groups:
        memory_ids = list(group.get("memory_ids") or [])
        if len(memory_ids) < 2:
            continue
        keep_id = memory_ids[-1]
        delete_ids = memory_ids[:-1]
        deleted_ids: list[int] = []
        for memory_id in delete_ids:
            if not apply:
                continue
            source_db = source_db_by_id.get(int(memory_id), "user")
            if _resolve_delete_memory_by_source_db(source_db, memory_id):
                deleted_ids.append(memory_id)
        deleted_total += len(deleted_ids)
        results.append(
            {
                **group,
                "keep_id": keep_id,
                "delete_ids": delete_ids,
                "deleted_ids": deleted_ids,
            }
        )

    return {
        "ok": True,
        "project_id": _ensure_project_id(),
        "apply": apply,
        "group_total": len(results),
        "deleted_total": deleted_total,
        "groups": results,
    }
