from __future__ import annotations

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
    _memory_payload,
    _merge_suggestion_payload,
    _metadata_dict,
    _parse_memory_locator,
    _resolve_superseded_memory,
    _should_skip_session_capture,
    _single_line,
    _success,
    _truncate,
    _validate_memory_content,
    _validate_memory_kind,
    _validate_tags,
    _with_source_db,
)
from vibe_rag.types import MemoryKind, SourceDB


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
    """Store a durable memory. Pass just content for quick notes, or summary+details+memory_kind for structured memories. scope='user' for cross-project knowledge, scope='project' (default) for project-specific decisions. Memories are automatically retrieved in future sessions via load_session_context."""
    if scope not in ("project", "user"):
        return _failure("invalid_scope", "scope must be 'project' or 'user'")

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
        except RuntimeError as e:
            return _failure("embedding_failed", f"embedding failed: {e}")

        db = _get_db() if scope == "project" else _get_user_db()
        source_db_label: SourceDB = "project" if scope == "project" else "user"
        mid = db.remember_structured(
            summary=summary,
            content=body,
            embedding=embeddings[0],
            tags=tags,
            project_id=_ensure_project_id(),
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
                current_project_id=_ensure_project_id(),
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
    except RuntimeError as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    inferred_kind = _infer_auto_memory_kind("", content, content)
    resolved_kind = inferred_kind if inferred_kind != "summary" else "note"
    capture_kind = "freeform"
    if memory_kind:
        kind_error = _validate_memory_kind(memory_kind)
        if kind_error:
            return _failure_from_error(kind_error)
        resolved_kind = memory_kind
        capture_kind = "manual"

    db = _get_db() if scope == "project" else _get_user_db()
    source_db_label = "project" if scope == "project" else "user"
    mid = db.remember_structured(
        summary=_truncate(_single_line(content), 200),
        content=content,
        embedding=embeddings[0],
        tags=tags,
        project_id=_ensure_project_id(),
        memory_kind=resolved_kind,
        metadata={"capture_kind": capture_kind, **(metadata or {})},
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
                    "metadata": {"capture_kind": capture_kind},
                },
                source_db_label,
            ),
            current_project_id=_ensure_project_id(),
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
    """Compatibility wrapper -- delegates to the unified ``remember`` tool."""
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
    """Edit an existing memory in place. Only provided fields are changed. Use 'project:ID' or 'user:ID' prefix to target a specific database, or just the numeric ID (defaults to project)."""
    parsed = _parse_memory_locator(
        memory_id,
        error_code="invalid_memory_id",
        error_field="memory_id",
        error_message="memory_id must be an integer, optionally prefixed with 'project:' or 'user:'",
    )
    if isinstance(parsed, dict):
        return _failure_from_error(parsed)
    source_db, sqlite_id = parsed

    # Resolve DB
    if source_db == "user":
        db = _get_user_db()
        source_db_label: SourceDB = "user"
    elif source_db == "project":
        db = _get_db()
        source_db_label = "project"
    else:
        # Default to project, fall back to user
        db = _get_db()
        source_db_label = "project"
        existing = db.get_memory(sqlite_id)
        if not existing:
            db = _get_user_db()
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
        except RuntimeError as e:
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
            current_project_id=_ensure_project_id(),
        ),
    )


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
                current_project_id=_ensure_project_id(),
            ),
        )

    duplicate = _find_duplicate_auto_memory(
        user_db=user_db,
        project_id=_ensure_project_id(),
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
                current_project_id=_ensure_project_id(),
            ),
        )

    non_novel = _find_non_novel_auto_memory(
        project_id=_ensure_project_id(),
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
            memory=_memory_payload(non_novel, current_project_id=_ensure_project_id()),
            merge_suggestion=_merge_suggestion_payload(non_novel, _ensure_project_id()),
        )

    merge_candidate = _find_merge_candidate(
        project_id=_ensure_project_id(),
        summary=summary,
        content=content,
        memory_kind=inferred_memory_kind,
    )

    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except RuntimeError as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    memory_id = user_db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=_ensure_project_id(),
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
        merge_suggestion=_merge_suggestion_payload(merge_candidate, _ensure_project_id()),
        memory=_memory_payload(
            _with_source_db(
                stored or {"id": memory_id, "summary": summary, "content": content},
                "user",
            ),
            current_project_id=_ensure_project_id(),
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
    if turns:
        last_assistant = str(turns[-1].get("assistant", ""))
        if _should_skip_session_capture(last_assistant):
            return _success(skipped=True, reason="low-signal response")
    task_error = _validate_memory_content(task)
    if task_error:
        return _failure_from_error(task_error)
    if not isinstance(turns, list):
        return _failure("invalid_turns", "turns must be a list")
    if not source_session_id.strip():
        return _failure("missing_source_session_id", "source_session_id is required")
    if not source_message_id.strip():
        return _failure("missing_source_message_id", "source_message_id is required")
    tags_error = _validate_tags(tags)
    if tags_error:
        return _failure_from_error(tags_error)

    try:
        summary, content = _distill_session_summary(turns)
    except ValueError as e:
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
    except RuntimeError as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

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
                current_project_id=_ensure_project_id(),
            ),
        )
    duplicate = _find_duplicate_auto_memory(
        user_db=user_db,
        project_id=_ensure_project_id(),
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
                current_project_id=_ensure_project_id(),
            ),
        )
    merge_candidate = _find_merge_candidate(
        project_id=_ensure_project_id(),
        summary=summary,
        content=content,
        memory_kind=inferred_memory_kind,
    )
    memory_id = user_db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=_ensure_project_id(),
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
        merge_suggestion=_merge_suggestion_payload(merge_candidate, _ensure_project_id()),
        memory=_memory_payload(
            _with_source_db(
                stored or {"id": memory_id, "summary": summary, "content": content},
                "user",
            ),
            current_project_id=_ensure_project_id(),
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
    except RuntimeError as e:
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
