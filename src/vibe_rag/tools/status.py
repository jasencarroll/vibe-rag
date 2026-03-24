"""MCP tools for project status dashboard and memory maintenance.

Provides two ``@mcp.tool()`` endpoints:

* **project_status** -- returns a comprehensive dashboard with index counts,
  staleness info, language breakdown, and optional memory-health diagnostics.
* **cleanup_duplicate_auto_memories** -- identifies (and optionally deletes)
  duplicate auto-captured session memories using a normalised-key deduplication
  strategy.

Internal helpers build the memory-health summary and resolve monkeypatched
references for testability.
"""

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


def _cleanup_candidate_summary(candidate: dict) -> dict:
    """Distil a raw cleanup-candidate dict into a compact display summary.

    Returns a dict with keys: ``id``, ``source_db``, ``summary``,
    ``memory_kind``, ``cleanup_reasons`` (list[str]),
    ``cleanup_priority`` (int), ``is_stale`` (bool), ``is_superseded`` (bool).
    """
    return {
        "id": candidate.get("id"),
        "source_db": candidate.get("source_db"),
        "summary": candidate.get("summary"),
        "memory_kind": candidate.get("memory_kind"),
        "cleanup_reasons": list(candidate.get("cleanup_reasons") or []),
        "cleanup_priority": int(candidate.get("cleanup_priority") or 0),
        "is_stale": bool(candidate.get("is_stale")),
        "is_superseded": bool(candidate.get("is_superseded")),
    }


def _resolve_all_memory_payloads():
    """Allow monkeypatching ``vibe_rag.tools._all_memory_payloads`` in tests."""
    import vibe_rag.tools as _pkg
    return getattr(_pkg, "_all_memory_payloads", _all_memory_payloads_impl)()


def _resolve_delete_memory_by_source_db(source_db, memory_id):
    """Allow monkeypatching ``vibe_rag.tools._delete_memory_by_source_db`` in tests."""
    import vibe_rag.tools as _pkg
    fn = getattr(_pkg, "_delete_memory_by_source_db", _delete_memory_by_source_db_impl)
    return fn(source_db, memory_id)


def _current_project_user_memory_count() -> int:
    """Return the number of non-superseded user-scoped memories for the current project."""
    current_project_id = _ensure_project_id()
    user_db = _get_user_db()
    return len(
        user_db.list_memories(
            limit=max(user_db.memory_count() + 10, 20),
            include_superseded=False,
            project_id=current_project_id,
        )
    )


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
        "top_cleanup_candidates": [_cleanup_candidate_summary(item) for item in cleanup_candidates[:3]],
        "recommended_actions": recommended_actions[:3],
        "by_capture_kind": capture_kind_counts,
        "by_source_type": source_type_counts,
    }


@mcp.tool()
def project_status(include_memory_health: bool = True) -> dict:
    """Dashboard view: index counts, staleness, language breakdown, and memory health including quality metrics, cleanup candidates, and recommendations. Use include_memory_health=False for a lighter response.

    Parameters
    ----------
    include_memory_health : bool, default True
        When *True* the response includes a ``memory_health`` section with
        stale/superseded/duplicate counts, top cleanup candidates, capture-kind
        and source-type breakdowns, and recommended maintenance actions.
        Set to *False* for a lighter, index-only response.

    Returns
    -------
    dict
        A ``_success()`` envelope (``ok=True``, ``project_id``) containing a
        ``status`` dict with the following top-level keys:

        * **counts** -- ``code_chunks``, ``doc_chunks``, ``project_memories``,
          ``user_memories`` (int counts for each store).
        * **metadata** -- index metadata (embedding provider, model, last
          index timestamp) via ``_index_metadata()``.
        * **stale** -- staleness indicators for the project index (files
          changed since last index, current git HEAD).
        * **language_stats** -- per-language chunk counts from the project DB.
        * **cleanup_candidates** -- up to 3 memory cleanup candidates with
          reasons, priority, and stale/superseded flags.
        * **memory_health** *(only when include_memory_health=True)* --
          nested dict with sub-keys:

          - ``summary`` -- ``total_memories``, ``stale_memories``,
            ``superseded_memories``, ``duplicate_auto_memory_groups``.
          - ``top_cleanup_candidates`` -- up to 3 candidate summaries.
          - ``recommended_actions`` -- up to 3 human-readable suggestions.
          - ``by_capture_kind`` -- dict mapping capture kind to count.
          - ``by_source_type`` -- dict mapping source type to count.
    """
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
            "user_memories": _current_project_user_memory_count(),
        },
        "metadata": metadata_state,
        "stale": stale,
        "language_stats": language_stats,
        "cleanup_candidates": [_cleanup_candidate_summary(item) for item in cleanup_candidates],
    }

    if include_memory_health:
        status["memory_health"] = _memory_health_summary()

    return _success(
        project_id=_ensure_project_id(),
        status=status,
    )


@mcp.tool()
def cleanup_duplicate_auto_memories(limit: int = 20, apply: bool = False) -> dict:
    """Find and optionally prune duplicate auto-captured session memories. apply=False (default) to preview, apply=True to delete duplicates keeping newest.

    Deduplication criteria
    ----------------------
    Two auto-captured memories are considered duplicates when they share the
    same normalised key -- a tuple of ``(project_id, capture_kind,
    lowercased single-line summary, lowercased single-line content)``.  Only
    memories whose ``capture_kind`` is ``session_rollup`` or
    ``session_distillation`` are eligible.  Within each duplicate group the
    newest memory (by ``updated_at`` then ``id``) is kept; older copies are
    candidates for deletion.

    Parameters
    ----------
    limit : int, default 20
        Maximum number of duplicate groups to process.  Must be >= 1.
    apply : bool, default False
        When *False* (default) the tool returns a **preview** of duplicate
        groups without deleting anything.  Set to *True* to actually delete
        the older duplicates, keeping only the newest memory per group.

    Returns
    -------
    dict
        Top-level keys:

        * **ok** (bool) -- always ``True`` on success.
        * **project_id** (str) -- current project identifier.
        * **apply** (bool) -- echo of the ``apply`` parameter.
        * **group_total** (int) -- number of duplicate groups found (up to
          *limit*).
        * **deleted_total** (int) -- total memories actually deleted (0 when
          ``apply=False``).
        * **groups** (list[dict]) -- one entry per duplicate group, each
          containing:

          - ``count`` (int) -- total memories in the group.
          - ``capture_kind`` (str) -- shared capture kind.
          - ``project_id`` (str) -- shared project id.
          - ``summary`` (str) -- shared summary text.
          - ``memory_ids`` (list[int]) -- all memory ids in the group,
            ordered by ``updated_at``.
          - ``keep_id`` (int) -- the newest memory id that is retained.
          - ``delete_ids`` (list[int]) -- ids that would be (or were) deleted.
          - ``deleted_ids`` (list[int]) -- ids that were actually deleted
            (empty when ``apply=False``).
    """
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
