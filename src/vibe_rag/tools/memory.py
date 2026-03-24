"""Memory tools -- MCP tool definitions for storing and editing memories.

This module registers 9 MCP tools via ``@mcp.tool()``:

  - **remember** -- store a durable memory (freeform or structured)
  - **update_memory** -- edit an existing memory in place
  - **summarize_thread** -- list and summarize memories attached to a thread
  - **ingest_daily_note** -- store a daily note (memory_event_v1 convention)
  - **ingest_pr_outcome** -- store a pull-request outcome (memory_event_v1)
  - **save_session_memory** -- hook-driven per-turn distillation
  - **save_session_summary** -- hook-driven rolling session summary
  - **supersede_memory** -- replace an outdated memory with a corrected version
  - **forget** -- permanently delete a memory by ID

It also provides **remember_structured**, a deprecated compat wrapper that
delegates to ``remember()`` and is *not* registered as an MCP tool.

Dual-DB design
~~~~~~~~~~~~~~
Memories live in one of two SQLite databases:

- **project** (``scope="project"``) -- stored in ``.vibe/index.db``,
  scoped to the current project.
- **user** (``scope="user"``) -- stored in ``~/.vibe/memory.db``,
  shared across all projects for cross-project knowledge.
"""

from __future__ import annotations

from datetime import date as calendar_date
from datetime import datetime, timezone
from typing import cast

from vibe_rag.server import _ensure_project_id, _get_db, _get_embedder, _get_user_db, mcp
from vibe_rag.tools._helpers import (
    _distill_session_summary,
    _distill_session_turn,
    _failure,
    _failure_from_error,
    _find_duplicate_auto_memory,
    _find_merge_candidate,
    _find_non_novel_auto_memory,
    _infer_auto_memory_kind,
    _infer_session_metadata,
    _is_low_signal_auto_capture,
    _is_transient_status_auto_capture,
    _list_thread_memory_results,
    _memory_payload,
    _merge_suggestion_payload,
    _metadata_dict,
    _parse_datetime_filter,
    _parse_memory_locator,
    _resolve_superseded_memory,
    _should_skip_session_capture,
    _single_line,
    _success,
    _truncate,
    _validate_memory_content,
    _validate_memory_kind,
    _validate_tags,
    _validate_thread_id,
    _with_source_db,
)
from vibe_rag.types import MemoryKind, SourceDB

MEMORY_EVENT_CONVENTION = "memory_event_v1"
DEFAULT_DAILY_NOTE_TAGS = "daily,note"
DEFAULT_PR_OUTCOME_TAGS = "pr,outcome"


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_event_at(event_at: str, field_name: str = "event_at") -> tuple[str | None, dict | None]:
    dt, error = _parse_datetime_filter(event_at, field_name)
    if error:
        return None, _failure_from_error(error)
    if dt is None:
        return None, None
    return _utc_iso(dt), None


def _normalize_note_date(note_date: str) -> tuple[str | None, dict | None]:
    normalized = note_date.strip()
    if not normalized:
        return None, _failure("missing_note_date", "note_date is required")
    try:
        return calendar_date.fromisoformat(normalized).isoformat(), None
    except ValueError:
        return None, _failure(
            "invalid_note_date",
            "note_date must be in YYYY-MM-DD format",
            note_date=note_date,
        )


def _normalize_thread(
    thread_id: str,
    default_thread_id: str,
    thread_title: str,
    default_thread_title: str,
) -> tuple[tuple[str, str] | None, dict | None]:
    resolved_thread_id = thread_id.strip() or default_thread_id
    error = _validate_thread_id(resolved_thread_id)
    if error:
        return None, _failure_from_error(error)
    resolved_thread_title = thread_title.strip() or default_thread_title
    return (resolved_thread_id, resolved_thread_title), None


@mcp.tool()
def remember(
    content: str,
    summary: str = "",
    details: str = "",
    memory_kind: MemoryKind = "",  # type: ignore[assignment]  # empty = auto-infer
    tags: str = "",
    scope: str = "project",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Store a durable memory. Pass just content for quick notes, or summary+details+memory_kind for structured memories. scope='user' for cross-project knowledge, scope='project' (default) for project-specific decisions. Memories are automatically retrieved in future sessions via load_session_context.

    Two storage paths
    -----------------
    **Freeform** (content only, summary left empty):
        A quick note.  The first 200 characters of *content* are used as the
        auto-generated summary.  ``memory_kind`` is inferred from content
        keywords when not explicitly provided (falls back to ``"note"``).
        Metadata is tagged ``capture_kind="freeform"``.

    **Structured** (summary provided, with optional details):
        A richer memory.  The stored body is ``summary`` alone when *details*
        is empty, or ``summary + "\\n\\n" + details`` otherwise.
        ``memory_kind`` defaults to ``"decision"`` when not provided.
        Metadata is tagged ``capture_kind="manual"``.

    Parameters
    ----------
    content : str
        The main text of the memory.  Required for freeform memories.
        Ignored when *summary* is provided (structured path), though it
        is accepted without error.  Max 10 000 characters.
    summary : str, optional
        Short headline.  When non-empty the structured path is used.
        Max 10 000 characters.
    details : str, optional
        Extended details appended after the summary in the stored body.
        Only meaningful on the structured path.  Max 10 000 characters.
    memory_kind : MemoryKind, optional
        One of ``"note"``, ``"decision"``, ``"constraint"``, ``"todo"``,
        ``"summary"``, ``"fact"``.  Defaults to ``"decision"`` on the
        structured path; auto-inferred on the freeform path (using
        keyword heuristics: todo > constraint > decision > fact, with
        ``"note"`` as the fallback when the inferred kind is
        ``"summary"``).
    tags : str, optional
        Comma-separated tag string, e.g. ``"api,auth,v2"``.
        Max 512 characters total.
    scope : str, optional
        ``"project"`` (default) writes to the project DB;
        ``"user"`` writes to the cross-project user DB.
    source_session_id : str, optional
        Opaque session identifier for provenance tracking.
    source_message_id : str, optional
        Opaque message identifier for provenance tracking.
    metadata : dict or None, optional
        Arbitrary key-value pairs merged into the stored metadata.
        ``capture_kind`` is set automatically (``"freeform"`` or
        ``"manual"``) and should not be overridden here.

    Returns
    -------
    dict
        On success: ``{"ok": True, "backend": "<scope>-sqlite",
        "memory": <MemoryPayload>}`` where ``<MemoryPayload>`` contains
        at minimum ``id`` (int), ``summary``, ``content``,
        ``memory_kind``, ``tags`` (list[str]), ``source_db``,
        ``created_at``, ``updated_at``, ``provenance``, and more.

        On failure: ``{"ok": False, "error": {"code": ..., "message": ...,
        "details": {...}}}``.  Common error codes:
        ``"invalid_scope"``, ``"empty_content"``, ``"content_too_large"``,
        ``"tags_too_long"``, ``"invalid_memory_kind"``,
        ``"embedding_failed"``.
    """
    if scope not in ("project", "user"):
        return _failure("invalid_scope", "scope must be 'project' or 'user'")

    current_project_id = _ensure_project_id()

    # --- Structured path: summary is provided ---
    if summary.strip():
        error = _validate_memory_content(summary)
        if error:
            return _failure_from_error(error)
        details_error = _validate_memory_content(details) if details else None
        if details_error:
            return _failure_from_error(details_error)
        tags_error = _validate_tags(tags)
        if tags_error:
            return _failure_from_error(tags_error)
        resolved_kind = memory_kind if memory_kind else "decision"
        kind_error = _validate_memory_kind(resolved_kind)
        if kind_error:
            return _failure_from_error(kind_error)

        body = summary if not details else f"{summary}\n\n{details}"
        try:
            embeddings = _get_embedder().embed_text_sync([body])
        except Exception as e:
            return _failure("embedding_failed", f"embedding failed: {e}")

        db = _get_db() if scope == "project" else _get_user_db()
        source_db_label: SourceDB = "project" if scope == "project" else "user"
        mid = db.remember_structured(
            summary=summary,
            content=body,
            embedding=embeddings[0],
            tags=tags,
            project_id=current_project_id,
            memory_kind=resolved_kind,
            metadata={"capture_kind": "manual", **(metadata or {})},
            source_session_id=source_session_id or None,
            source_message_id=source_message_id or None,
        )
        stored = db.get_memory(mid)
        return _success(
            backend=f"{source_db_label}-sqlite",
            memory=_memory_payload(
                _with_source_db(
                    stored or {"id": mid, "summary": summary, "content": body},
                    source_db_label,
                ),
                current_project_id=current_project_id,
            ),
        )

    # --- Freeform path: content only ---
    error = _validate_memory_content(content)
    if error:
        return _failure_from_error(error)
    tags_error = _validate_tags(tags)
    if tags_error:
        return _failure_from_error(tags_error)

    try:
        embeddings = _get_embedder().embed_text_sync([content])
    except Exception as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    inferred_kind = _infer_auto_memory_kind("", content, content)
    resolved_kind = inferred_kind if inferred_kind != "summary" else "note"
    if memory_kind:
        kind_error = _validate_memory_kind(memory_kind)
        if kind_error:
            return _failure_from_error(kind_error)
        resolved_kind = memory_kind

    db = _get_db() if scope == "project" else _get_user_db()
    source_db_label = "project" if scope == "project" else "user"
    mid = db.remember_structured(
        summary=_truncate(_single_line(content), 200),
        content=content,
        embedding=embeddings[0],
        tags=tags,
        project_id=current_project_id,
        memory_kind=resolved_kind,
        metadata={"capture_kind": "freeform", **(metadata or {})},
        source_session_id=source_session_id or None,
        source_message_id=source_message_id or None,
    )
    stored = db.get_memory(mid)
    return _success(
        backend=f"{source_db_label}-sqlite",
        memory=_memory_payload(
            _with_source_db(
                stored
                or {
                    "id": mid,
                    "summary": _truncate(_single_line(content), 200),
                    "content": content,
                    "memory_kind": resolved_kind,
                    "metadata": {"capture_kind": "freeform"},
                },
                source_db_label,
            ),
            current_project_id=current_project_id,
        ),
    )


def remember_structured(
    summary: str,
    details: str = "",
    memory_kind: MemoryKind = "decision",
    tags: str = "",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Deprecated: use ``remember(content='', summary=..., details=..., memory_kind=...)`` instead.

    Backward-compatible wrapper kept for existing tests and internal callers.
    """
    return remember(
        content="",
        summary=summary,
        details=details,
        memory_kind=memory_kind,
        tags=tags,
        scope="project",
        source_session_id=source_session_id,
        source_message_id=source_message_id,
        metadata=metadata,
    )


@mcp.tool()
def update_memory(
    memory_id: str,
    content: str = "",
    summary: str = "",
    details: str = "",
    memory_kind: MemoryKind = "",  # type: ignore[assignment]  # empty = unchanged
    tags: str = "",
    metadata: dict | None = None,
) -> dict:
    """Edit an existing memory in place. Only provided fields are changed. Use 'project:ID' or 'user:ID' prefix to target a specific database, or just the numeric ID (defaults to project).

    This performs a partial update: only non-empty parameters overwrite the
    corresponding stored field.  Empty-string parameters (the default) leave
    the existing value untouched.  When any text field (*content*, *summary*,
    or *details*) is changed, the embedding is automatically recomputed.

    Parameters
    ----------
    memory_id : str
        The memory to update.  Accepts three formats:

        - A bare integer string, e.g. ``"42"`` -- looks up the ID in the
          project DB first, then the user DB.  Returns an
          ``"ambiguous_memory_id"`` error if the same numeric ID exists
          in both databases.
        - ``"project:<int>"``, e.g. ``"project:42"`` -- targets only the
          project DB.
        - ``"user:<int>"``, e.g. ``"user:42"`` -- targets only the user DB.
    content : str, optional
        New freeform body text.  When provided, the summary is auto-generated
        from the first 200 characters unless *summary* is also provided.
        Max 10 000 characters.
    summary : str, optional
        New headline.  When provided alongside *details* (or existing
        details), the stored body is rebuilt as
        ``summary + "\\n\\n" + details``.  Max 10 000 characters.
    details : str, optional
        New extended details.  Combined with *summary* (new or existing)
        to form the stored body.  Max 10 000 characters.
    memory_kind : MemoryKind, optional
        New kind.  Must be one of ``"note"``, ``"decision"``,
        ``"constraint"``, ``"todo"``, ``"summary"``, ``"fact"``.
    tags : str, optional
        New comma-separated tag string.  Replaces all existing tags.
        Max 512 characters.
    metadata : dict or None, optional
        Key-value pairs **merged** into the existing metadata (existing
        keys not present in *metadata* are preserved; overlapping keys
        are overwritten by the new values).

    Returns
    -------
    dict
        On success: ``{"ok": True, "backend": "<scope>-sqlite",
        "memory": <MemoryPayload>}`` with the updated memory.

        On failure: ``{"ok": False, "error": {"code": ..., ...}}``.
        Common error codes: ``"invalid_memory_id"`` (not a valid
        locator string), ``"memory_not_found"`` (no memory with that
        ID in the resolved DB), ``"ambiguous_memory_id"`` (bare
        integer exists in both project and user DBs),
        ``"empty_content"``, ``"content_too_large"``,
        ``"tags_too_long"``, ``"invalid_memory_kind"``,
        ``"embedding_failed"``.
    """
    parsed = _parse_memory_locator(
        memory_id,
        error_code="invalid_memory_id",
        error_field="memory_id",
        error_message="memory_id must be an integer, optionally prefixed with 'project:' or 'user:'",
    )
    if isinstance(parsed, dict):
        return _failure_from_error(parsed)
    source_db, sqlite_id = parsed
    current_project_id = _ensure_project_id()

    # Resolve DB
    if source_db == "user":
        db = _get_user_db()
        source_db_label: SourceDB = "user"
    elif source_db == "project":
        db = _get_db()
        source_db_label = "project"
    else:
        project_db = _get_db()
        user_db = _get_user_db()
        project_memory = project_db.get_memory(sqlite_id)
        user_memory = user_db.get_memory(sqlite_id)
        if project_memory and user_memory:
            return _failure(
                "ambiguous_memory_id",
                f"memory {sqlite_id} exists in multiple memory stores",
                memory_id=sqlite_id,
                source_dbs=["project", "user"],
            )
        if project_memory:
            db = project_db
            source_db_label = "project"
        else:
            db = user_db
            source_db_label = "user"

    existing = db.get_memory(sqlite_id)
    if not existing:
        return _failure("memory_not_found", f"memory {sqlite_id} not found")

    # Validate non-empty fields
    new_content = content if content.strip() else None
    new_summary = summary if summary.strip() else None
    new_details = details if details.strip() else None
    new_tags = tags if tags else None
    new_kind = memory_kind if memory_kind else None
    new_metadata = metadata

    if new_content:
        error = _validate_memory_content(new_content)
        if error:
            return _failure_from_error(error)
    if new_summary:
        error = _validate_memory_content(new_summary)
        if error:
            return _failure_from_error(error)
    if new_details:
        error = _validate_memory_content(new_details)
        if error:
            return _failure_from_error(error)
    if new_tags:
        tags_error = _validate_tags(new_tags)
        if tags_error:
            return _failure_from_error(tags_error)
    if new_kind:
        kind_error = _validate_memory_kind(new_kind)
        if kind_error:
            return _failure_from_error(kind_error)

    # Build the final content for embedding when text fields change
    needs_reembed = bool(new_content or new_summary or new_details)
    embedding: list[float] | None = None
    # Determine effective content for DB update
    db_content: str | None = None
    db_summary: str | None = new_summary

    if new_content:
        # Freeform-style update: content drives the body
        db_content = new_content
        if not db_summary:
            db_summary = _truncate(_single_line(new_content), 200)
    elif new_summary or new_details:
        # Structured-style: rebuild content from summary + details
        eff_summary = new_summary or existing.get("summary") or ""
        eff_details = new_details or ""
        db_content = eff_summary if not eff_details else f"{eff_summary}\n\n{eff_details}"

    if needs_reembed:
        embed_text = db_content or existing.get("content", "")
        try:
            embeddings = _get_embedder().embed_text_sync([embed_text])
            embedding = embeddings[0]
        except Exception as e:
            return _failure("embedding_failed", f"embedding failed: {e}")

    # Merge metadata
    final_metadata: dict | None = None
    if new_metadata is not None:
        old_meta = existing.get("metadata") or {}
        final_metadata = {**old_meta, **new_metadata}

    db.update_memory(
        sqlite_id,
        embedding=embedding,
        content=db_content,
        summary=db_summary,
        tags=new_tags,
        memory_kind=new_kind,
        metadata=final_metadata,
    )

    updated = db.get_memory(sqlite_id)
    return _success(
        backend=f"{source_db_label}-sqlite",
        memory=_memory_payload(
            _with_source_db(updated or existing, source_db_label),
            current_project_id=current_project_id,
        ),
    )


@mcp.tool()
def summarize_thread(
    thread_id: str,
    limit: int = 20,
    scope: str = "all",
    since: str = "",
    until: str = "",
    include_superseded: bool = False,
) -> dict:
    """List and summarize memories attached to a thread. Memories can carry `metadata.thread_id` / `metadata.thread_title` or `metadata.thread = {id, title}`."""
    error, results = _list_thread_memory_results(
        thread_id,
        limit=limit,
        scope=scope,
        since=since,
        until=until,
        include_superseded=include_superseded,
    )
    if error:
        return _failure_from_error(
            error,
            thread_id=thread_id,
            limit=limit,
            scope=scope,
            since=since,
            until=until,
            include_superseded=include_superseded,
        )

    current_project_id = _ensure_project_id()
    payloads = [
        _memory_payload(result, current_project_id=current_project_id)
        for result in results
    ]
    normalized_thread_id = thread_id.strip()

    if not payloads:
        return _success(
            thread_id=normalized_thread_id,
            scope=scope,
            since=since,
            until=until,
            include_superseded=include_superseded,
            result_total=0,
            results=[],
            counts={"by_kind": {}, "by_source_db": {}},
            status={"started_at": None, "updated_at": None, "thread_title": None},
            summary=f"No memories found for thread '{normalized_thread_id}'.",
        )

    by_kind: dict[str, int] = {}
    by_source_db: dict[str, int] = {}
    thread_title: str | None = None
    for item in payloads:
        kind = str(item.get("memory_kind") or "note")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        source_db = str(item.get("source_db") or "unknown")
        by_source_db[source_db] = by_source_db.get(source_db, 0) + 1
        if not thread_title and item.get("thread_title"):
            thread_title = str(item["thread_title"])

    oldest = payloads[-1]
    newest = payloads[0]

    def _timeline_at(item: dict) -> str | None:
        metadata = _metadata_dict(item.get("metadata"))
        value = metadata.get("event_at") or item.get("updated_at") or item.get("created_at")
        return str(value) if value else None

    title = thread_title or normalized_thread_id
    latest_summary = str(newest.get("summary") or newest.get("content") or "").strip()
    dominant_kinds = ", ".join(
        f"{count} {kind}"
        for kind, count in sorted(by_kind.items(), key=lambda item: (-item[1], item[0]))[:3]
    )
    summary = f"Thread {title}: {len(payloads)} memories"
    if dominant_kinds:
        summary += f" ({dominant_kinds})"
    if latest_summary:
        summary += f". Latest: {latest_summary}"

    return _success(
        thread_id=normalized_thread_id,
        thread_title=thread_title,
        scope=scope,
        since=since,
        until=until,
        include_superseded=include_superseded,
        result_total=len(payloads),
        results=payloads,
        counts={"by_kind": by_kind, "by_source_db": by_source_db},
        status={
            "started_at": _timeline_at(oldest),
            "updated_at": _timeline_at(newest),
            "thread_title": thread_title,
        },
        summary=summary,
    )


@mcp.tool()
def ingest_daily_note(
    note_date: str,
    summary: str,
    details: str = "",
    event_at: str = "",
    tags: str = DEFAULT_DAILY_NOTE_TAGS,
    scope: str = "user",
    thread_id: str = "",
    thread_title: str = "",
    metadata: dict | None = None,
) -> dict:
    """Store a daily note using the memory_event_v1 convention. Defaults to user scope and a `daily:YYYY-MM-DD` thread."""
    normalized_note_date, date_error = _normalize_note_date(note_date)
    if date_error:
        return date_error

    resolved_thread, thread_error = _normalize_thread(
        thread_id,
        f"daily:{normalized_note_date}",
        thread_title,
        f"Daily Note {normalized_note_date}",
    )
    if thread_error:
        return thread_error
    resolved_thread_id, resolved_thread_title = resolved_thread

    normalized_event_at, event_error = _normalize_event_at(event_at)
    if event_error:
        return event_error
    if not normalized_event_at:
        normalized_event_at = f"{normalized_note_date}T00:00:00Z"

    result = remember(
        content="",
        summary=summary,
        details=details,
        memory_kind="summary",
        tags=tags,
        scope=scope,
        metadata={
            **(metadata or {}),
            "capture_kind": "adapter_daily_note",
            "convention": MEMORY_EVENT_CONVENTION,
            "adapter": "daily_note",
            "note_date": normalized_note_date,
            "event_at": normalized_event_at,
            "thread": {"id": resolved_thread_id, "title": resolved_thread_title},
        },
    )
    if result.get("ok"):
        result["adapter"] = "daily_note"
        result["convention"] = MEMORY_EVENT_CONVENTION
    return result


@mcp.tool()
def ingest_pr_outcome(
    pr_number: int,
    title: str,
    outcome: str,
    summary: str = "",
    details: str = "",
    event_at: str = "",
    tags: str = DEFAULT_PR_OUTCOME_TAGS,
    scope: str = "project",
    thread_id: str = "",
    thread_title: str = "",
    issue_id: str = "",
    branch: str = "",
    commit_sha: str = "",
    pr_url: str = "",
    metadata: dict | None = None,
) -> dict:
    """Store a pull-request outcome using the memory_event_v1 convention. Defaults to project scope and a `pr:<number>` thread."""
    try:
        normalized_pr_number = int(pr_number)
    except (TypeError, ValueError):
        return _failure("invalid_pr_number", "pr_number must be a positive integer", pr_number=pr_number)
    if normalized_pr_number <= 0:
        return _failure("invalid_pr_number", "pr_number must be a positive integer", pr_number=pr_number)

    title_error = _validate_memory_content(title)
    if title_error:
        return _failure_from_error(title_error, field="title")

    normalized_outcome = " ".join(outcome.split()).lower()
    if not normalized_outcome:
        return _failure("missing_outcome", "outcome is required")

    resolved_thread, thread_error = _normalize_thread(
        thread_id,
        f"pr:{normalized_pr_number}",
        thread_title,
        f"PR #{normalized_pr_number}: {title.strip()}",
    )
    if thread_error:
        return thread_error
    resolved_thread_id, resolved_thread_title = resolved_thread

    normalized_event_at, event_error = _normalize_event_at(event_at)
    if event_error:
        return event_error
    if not normalized_event_at:
        normalized_event_at = _utc_iso(datetime.now(timezone.utc))

    resolved_summary = summary.strip() or f"PR #{normalized_pr_number} {normalized_outcome}: {title.strip()}"
    detail_lines = [
        f"PR #{normalized_pr_number}: {title.strip()}",
        f"Outcome: {normalized_outcome}",
    ]
    if issue_id.strip():
        detail_lines.append(f"Issue: {issue_id.strip()}")
    if branch.strip():
        detail_lines.append(f"Branch: {branch.strip()}")
    if commit_sha.strip():
        detail_lines.append(f"Commit: {commit_sha.strip()}")
    if pr_url.strip():
        detail_lines.append(f"URL: {pr_url.strip()}")
    if details.strip():
        detail_lines.extend(("", details.strip()))

    result = remember(
        content="",
        summary=resolved_summary,
        details="\n".join(detail_lines),
        memory_kind="summary",
        tags=tags,
        scope=scope,
        metadata={
            **(metadata or {}),
            "capture_kind": "adapter_pr_outcome",
            "convention": MEMORY_EVENT_CONVENTION,
            "adapter": "pr_outcome",
            "event_at": normalized_event_at,
            "outcome": normalized_outcome,
            "pr_number": normalized_pr_number,
            "pr_title": title.strip(),
            "thread": {"id": resolved_thread_id, "title": resolved_thread_title},
            "issue_id": issue_id.strip() or None,
            "branch": branch.strip() or None,
            "commit_sha": commit_sha.strip() or None,
            "pr_url": pr_url.strip() or None,
        },
    )
    if result.get("ok"):
        result["adapter"] = "pr_outcome"
        result["convention"] = MEMORY_EVENT_CONVENTION
    return result


@mcp.tool()
def save_session_memory(
    task: str,
    response: str,
    source_session_id: str,
    source_message_id: str,
    user_message_id: str = "",
    tags: str = "session,auto",
    memory_kind: MemoryKind = "summary",
    metadata: dict | None = None,
) -> dict:
    """Hook-driven: persist a distilled memory from a completed chat turn. Filters low-signal content and deduplicates automatically. Typically invoked by session hooks, not called directly."""
    if _should_skip_session_capture(response):
        return _success(skipped=True, reason="low-signal response")
    task_error = _validate_memory_content(task)
    if task_error:
        return _failure_from_error(task_error)
    response_error = _validate_memory_content(response)
    if response_error:
        return _failure_from_error(response_error)
    if not source_session_id.strip():
        return _failure("missing_source_session_id", "source_session_id is required")
    if not source_message_id.strip():
        return _failure("missing_source_message_id", "source_message_id is required")
    tags_error = _validate_tags(tags)
    if tags_error:
        return _failure_from_error(tags_error)
    kind_error = _validate_memory_kind(memory_kind)
    if kind_error:
        return _failure_from_error(kind_error)

    enriched_metadata = {
        "capture_kind": "session_distillation",
        "task": _single_line(task.strip()),
        **(metadata or {}),
    }
    if user_message_id.strip():
        enriched_metadata["user_message_id"] = user_message_id.strip()

    summary, content = _distill_session_turn(task, response)
    inferred_memory_kind = memory_kind if memory_kind != "summary" else _infer_auto_memory_kind(task.strip(), summary, content)
    if _is_transient_status_auto_capture(task.strip(), summary, content):
        return _success(skipped=True, reason="non-durable auto memory")
    if _is_low_signal_auto_capture(
        task=task.strip(),
        summary=summary,
        content=content,
        capture_kind="session_distillation",
    ):
        return _success(skipped=True, reason="low-signal auto memory")

    enriched_metadata = _infer_session_metadata(task.strip(), response, enriched_metadata)

    current_project_id = _ensure_project_id()
    user_db = _get_user_db()
    existing = user_db.get_memory_by_source(
        source_session_id.strip(), source_message_id.strip()
    )
    if existing:
        return _success(
            backend="user-sqlite",
            deduplicated=True,
            memory=_memory_payload(
                _with_source_db(existing, "user"),
                current_project_id=current_project_id,
            ),
        )

    duplicate = _find_duplicate_auto_memory(
        user_db=user_db,
        project_id=current_project_id,
        summary=summary,
        content=content,
        capture_kind="session_distillation",
    )
    if duplicate:
        return _success(
            backend="user-sqlite",
            deduplicated=True,
            skipped=True,
            reason="duplicate auto memory",
            memory=_memory_payload(
                _with_source_db(duplicate, "user"),
                current_project_id=current_project_id,
            ),
        )

    non_novel = _find_non_novel_auto_memory(
        project_id=current_project_id,
        summary=summary,
        content=content,
    )
    if non_novel:
        return _success(
            backend="user-sqlite",
            deduplicated=True,
            skipped=True,
            reason="non-novel auto memory",
            memory_kind=inferred_memory_kind,
            memory=_memory_payload(non_novel, current_project_id=current_project_id),
            merge_suggestion=_merge_suggestion_payload(non_novel, current_project_id),
        )

    merge_candidate = _find_merge_candidate(
        project_id=current_project_id,
        summary=summary,
        content=content,
        memory_kind=inferred_memory_kind,
    )

    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except Exception as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    memory_id = user_db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=current_project_id,
        memory_kind=inferred_memory_kind,
        metadata=enriched_metadata,
        source_session_id=source_session_id.strip(),
        source_message_id=source_message_id.strip(),
    )
    stored = user_db.get_memory(memory_id)
    return _success(
        backend="user-sqlite",
        deduplicated=False,
        memory_kind=inferred_memory_kind,
        merge_suggestion=_merge_suggestion_payload(merge_candidate, current_project_id),
        memory=_memory_payload(
            _with_source_db(
                stored or {"id": memory_id, "summary": summary, "content": content},
                "user",
            ),
            current_project_id=current_project_id,
        ),
    )


@mcp.tool()
def save_session_summary(
    task: str,
    turns: list[dict],
    source_session_id: str,
    source_message_id: str,
    user_message_id: str = "",
    tags: str = "session,summary,auto",
    metadata: dict | None = None,
) -> dict:
    """Hook-driven: maintain a rolling summary of the current session. Each call supersedes the previous summary. Typically invoked by session hooks, not called directly."""
    task_error = _validate_memory_content(task)
    if task_error:
        return _failure_from_error(task_error)
    if not isinstance(turns, list):
        return _failure("invalid_turns", "turns must be a list")
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return _failure("invalid_turns", f"turns[{idx}] must be an object")
    if not source_session_id.strip():
        return _failure("missing_source_session_id", "source_session_id is required")
    if not source_message_id.strip():
        return _failure("missing_source_message_id", "source_message_id is required")
    tags_error = _validate_tags(tags)
    if tags_error:
        return _failure_from_error(tags_error)
    if turns:
        last_assistant = str(turns[-1].get("assistant", ""))
        if _should_skip_session_capture(last_assistant):
            return _success(skipped=True, reason="low-signal response")

    try:
        summary, content = _distill_session_summary(turns)
    except (ValueError, AttributeError, TypeError) as e:
        return _failure("invalid_turns", str(e))
    inferred_memory_kind = "summary"
    if _is_transient_status_auto_capture(task.strip(), summary, content):
        return _success(skipped=True, reason="non-durable auto memory")
    if _is_low_signal_auto_capture(
        task=task.strip(),
        summary=summary,
        content=content,
        capture_kind="session_rollup",
        turn_count=len(turns),
    ):
        return _success(skipped=True, reason="low-signal auto memory")

    summary_source_message_id = "__session_summary__"
    enriched_metadata = {
        "capture_kind": "session_rollup",
        "task": _single_line(task.strip()),
        "latest_message_id": source_message_id.strip(),
        "turn_count": len(turns),
        **(metadata or {}),
    }
    enriched_metadata = _infer_session_metadata(task.strip(), content, enriched_metadata)
    if user_message_id.strip():
        enriched_metadata["user_message_id"] = user_message_id.strip()

    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except Exception as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    current_project_id = _ensure_project_id()
    user_db = _get_user_db()
    existing = user_db.get_memory_by_source(
        source_session_id.strip(), summary_source_message_id
    )
    if existing and _metadata_dict(existing.get("metadata")).get("latest_message_id") == source_message_id.strip():
        return _success(
            backend="user-sqlite",
            deduplicated=True,
            memory=_memory_payload(
                _with_source_db(existing, "user"),
                current_project_id=current_project_id,
            ),
        )
    duplicate = _find_duplicate_auto_memory(
        user_db=user_db,
        project_id=current_project_id,
        summary=summary,
        content=content,
        capture_kind="session_rollup",
    )
    if duplicate:
        return _success(
            backend="user-sqlite",
            deduplicated=True,
            skipped=True,
            reason="duplicate auto memory",
            memory=_memory_payload(
                _with_source_db(duplicate, "user"),
                current_project_id=current_project_id,
            ),
        )
    merge_candidate = _find_merge_candidate(
        project_id=current_project_id,
        summary=summary,
        content=content,
        memory_kind=inferred_memory_kind,
    )
    memory_id = user_db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=current_project_id,
        memory_kind=inferred_memory_kind,
        metadata=enriched_metadata,
        source_session_id=source_session_id.strip(),
        source_message_id=summary_source_message_id,
        supersedes=int(existing["id"]) if existing else None,
    )
    stored = user_db.get_memory(memory_id)
    return _success(
        backend="user-sqlite",
        deduplicated=False,
        memory_kind=inferred_memory_kind,
        merge_suggestion=_merge_suggestion_payload(merge_candidate, current_project_id),
        memory=_memory_payload(
            _with_source_db(
                stored or {"id": memory_id, "summary": summary, "content": content},
                "user",
            ),
            current_project_id=current_project_id,
        ),
    )


@mcp.tool()
def supersede_memory(
    old_memory_id: str,
    summary: str,
    details: str = "",
    memory_kind: MemoryKind = "decision",
    tags: str = "",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Replace an outdated memory with a corrected version. Creates a new memory linked to the old one, marking it as superseded. Use for changed decisions; use update_memory for minor edits."""
    if not old_memory_id:
        return _failure("missing_old_memory_id", "old_memory_id is required")
    parsed_memory = _parse_memory_locator(
        old_memory_id,
        error_code="invalid_old_memory_id",
        error_field="old_memory_id",
        error_message="old_memory_id must be an integer or source-qualified integer like project:12",
    )
    if isinstance(parsed_memory, dict):
        return _failure_from_error(parsed_memory)
    source_db, old_memory_int = parsed_memory
    error = _validate_memory_content(summary)
    if error:
        return _failure_from_error(error)
    details_error = _validate_memory_content(details) if details else None
    if details_error:
        return _failure_from_error(details_error)
    tags_error = _validate_tags(tags)
    if tags_error:
        return _failure_from_error(tags_error)
    kind_error = _validate_memory_kind(memory_kind)
    if kind_error:
        return _failure_from_error(kind_error)

    current_project_id = _ensure_project_id()
    resolved = _resolve_superseded_memory(old_memory_int, current_project_id, source_db=source_db)
    if isinstance(resolved, dict):
        return _failure_from_error(resolved)
    owner_db, owner_source_db, _old_memory = resolved

    content = summary if not details else f"{summary}\n\n{details}"
    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except Exception as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    user_db = _get_user_db()
    new_memory_id = user_db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=current_project_id,
        memory_kind=memory_kind,
        metadata={"capture_kind": "manual", **(metadata or {})},
        source_session_id=source_session_id or None,
        source_message_id=source_message_id or None,
        supersedes=old_memory_int,
        update_superseded=False,
    )
    if not owner_db.set_memory_superseded_by(old_memory_int, new_memory_id):
        user_db.forget(new_memory_id)
        return _failure(
            "supersede_failed",
            f"failed to mark memory {old_memory_int} as superseded",
            old_memory_id=old_memory_int,
            source_db=owner_source_db,
        )

    stored = user_db.get_memory(new_memory_id)
    return _success(
        backend="user-sqlite",
        memory=_memory_payload(
            _with_source_db(
                stored
                or {
                    "id": new_memory_id,
                    "summary": summary,
                    "content": content,
                    "supersedes": old_memory_int,
                },
                "user",
            ),
            current_project_id=current_project_id,
        ),
    )


@mcp.tool()
def forget(memory_id: str) -> dict:
    """Permanently delete a memory by ID. Use 'project:ID' or 'user:ID' prefix to target a specific database."""
    parsed_memory = _parse_memory_locator(
        memory_id,
        error_code="invalid_memory_id",
        error_field="memory_id",
        error_message="memory_id must be an integer or source-qualified integer like project:12",
    )
    if isinstance(parsed_memory, dict):
        return _failure_from_error(parsed_memory)
    source_db, sqlite_id = parsed_memory

    candidates: list[tuple[object, SourceDB]] = []
    if source_db == "project":
        candidates = [(_get_db(), "project")]
    elif source_db == "user":
        candidates = [(_get_user_db(), "user")]
    else:
        for db, db_source in ((_get_db(), "project"), (_get_user_db(), "user")):
            if db.get_memory(sqlite_id):
                candidates.append((db, db_source))
        if len(candidates) > 1:
            return _failure(
                "ambiguous_memory_id",
                f"memory {sqlite_id} exists in multiple memory stores",
                memory_id=memory_id,
                source_dbs=[db_source for _, db_source in candidates],
            )

    for db, db_source in candidates:
        content = db.forget(sqlite_id)
        if not content:
            continue
        return _success(
            backend=f"{db_source}-sqlite",
            deleted=True,
            memory_id=sqlite_id,
            content_preview=content[:200],
        )
    return _failure("memory_not_found", f"memory {memory_id} not found", memory_id=sqlite_id)
