"""Shared helper functions for MCP tool implementations.

Provides the building blocks that the tool submodules (search, memory, session,
status, index) import: result formatting, validation, search pipelines, ranking,
briefing assembly, and project-state introspection.  Constants for query/memory
limits and memory-classification term sets are also defined here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import sqlite3
import subprocess
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from vibe_rag.chunking import collect_files
from vibe_rag.constants import EXT_TO_LANG
from vibe_rag.server import (
    _ensure_project_id,
    _get_db,
    _get_embedder,
    _get_user_db,
)
from vibe_rag.indexing.embedder import (
    ProgressCallback,
    embedding_provider_status,
    resolve_embedding_profile,
)
from vibe_rag.types import (
    CodeSearchResult,
    DocSearchResult,
    MemoryKind,
    MemoryPayload,
    MemoryProvenance,
    MemoryRow,
    RankedCodeResult,
    RankedDocResult,
    SearchProvenance,
    SourceDB,
    ToolError,
)

MAX_QUERY_LENGTH = 10_000
MAX_MEMORY_LENGTH = 10_000
MAX_TAGS_LENGTH = 512
MAX_THREAD_ID_LENGTH = 256
BRIEFING_CHAR_BUDGET = 6000
ALLOWED_MEMORY_KINDS = {"note", "decision", "constraint", "todo", "summary", "fact"}
INDEX_METADATA_KEY = "project_index_metadata"
BOILERPLATE_TASK_PATTERNS = (
    "reply with only",
    "reply only",
    "reply with exactly",
    "respond with only",
    "respond only",
    "project id loaded in session context",
    "session context was loaded",
)
DURABLE_MEMORY_TERMS = {
    "decision",
    "constraint",
    "todo",
    "fact",
    "must",
    "should",
    "always",
    "never",
    "owns",
    "owner",
    "validate",
    "validation",
    "refresh",
    "deploy",
    "deployment",
    "pipeline",
    "auth",
    "role",
    "roles",
    "gateway",
}
DECISION_TERMS = {"decision", "decided", "choose", "chosen", "prefer", "policy", "owns", "owner"}
CONSTRAINT_TERMS = {"constraint", "must", "cannot", "required", "requires", "never", "always", "limit", "allowed", "accepted"}
TODO_TERMS = {"todo", "follow-up", "followup", "next", "still", "open", "pending", "finish", "add", "implement"}
FACT_TERMS = {"fact", "lives", "located", "uses", "version", "path", "project", "id"}
TRANSIENT_STATUS_PATTERNS = (
    "looks good",
    "working now",
    "works now",
    "passed",
    "tests passed",
    "all tests passed",
    "resolved",
    "fixed",
    "issue has been resolved",
    "completed successfully",
    "done",
    "complete",
)
_COMMON_TASK_VERBS = {"fix", "add", "update", "implement", "refactor", "debug", "check", "review", "test", "run", "build", "deploy"}
STRUCTURED_MEMORY_PRIORITY = {
    "decision": 0,
    "constraint": 1,
    "summary": 2,
    "todo": 3,
    "fact": 4,
    "note": 5,
}
AUTO_MEMORY_SCAN_LIMIT = 200
AUTO_MEMORY_RECENCY_DAYS = 30

PROCEDURAL_QUERY_TERMS = {
    "release",
    "publish",
    "tag",
    "workflow",
    "notes",
    "bootstrap",
    "hook",
    "session",
    "trust",
    "mcp",
    "config",
}
DOC_FOCUSED_QUERY_TERMS = {
    "setup",
    "guide",
    "docs",
    "documentation",
    "resume",
    "continue",
    "readme",
    "editor",
    "ide",
    "integration",
    "install",
    "configure",
    "configuration",
    "zed",
    "jetbrains",
    "neovim",
}
logger = logging.getLogger(__name__)


def _tool_error(code: str, message: str, **details: Any) -> ToolError:
    """Build a structured ``ToolError`` dict with a machine-readable code and human message."""
    return {
        "code": code,
        "message": message,
        "details": details,
    }


def _failure(code: str, message: str, **details: Any) -> dict:
    """Return a standard ``{"ok": False, "error": ...}`` envelope for tool failures."""
    return {"ok": False, "error": _tool_error(code, message, **details)}


def _success(**payload: Any) -> dict:
    """Return a standard ``{"ok": True, ...}`` envelope for successful tool responses."""
    return {"ok": True, **payload}


def _failure_from_error(error: ToolError, **details: Any) -> dict:
    """Convert an existing ``ToolError`` into a failure envelope, merging extra details."""
    merged = dict(error.get("details") or {})
    merged.update(details)
    return _failure(error["code"], error["message"], **merged)


def _user_db_unavailable_error(exc: Exception) -> ToolError:
    """Build a consistent error payload when the user memory DB cannot be opened."""
    return _tool_error("user_db_unavailable", f"user memory DB unavailable: {exc}")


def _optional_user_db() -> tuple[Any | None, ToolError | None]:
    """Return the user DB when readable, or ``(None, ToolError)`` when not."""
    try:
        return _get_user_db(), None
    except Exception as exc:
        return None, _user_db_unavailable_error(exc)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_paths(paths: list[str] | str | None) -> tuple[list[Path], Path] | str:
    project_root = Path.cwd().resolve()

    if paths is None:
        raw_paths = ["."]
    elif isinstance(paths, str):
        raw_paths = [paths]
    else:
        raw_paths = paths

    normalized: list[Path] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError:
            return f"Error: path '{raw_path}' is outside project root."
        normalized.append(resolved)

    return normalized, project_root


def _relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root))


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_query(query: str) -> ToolError | None:
    """Return a ``ToolError`` if *query* is empty or exceeds ``MAX_QUERY_LENGTH``, else ``None``."""
    if not query.strip():
        return _tool_error("empty_query", "query is empty")
    if len(query) > MAX_QUERY_LENGTH:
        return _tool_error("query_too_long", "query is too long", max_length=MAX_QUERY_LENGTH)
    return None


def _validate_memory_content(content: str) -> ToolError | None:
    """Return a ``ToolError`` if *content* is empty or exceeds ``MAX_MEMORY_LENGTH``, else ``None``."""
    if not content.strip():
        return _tool_error("empty_content", "content is empty")
    if len(content) > MAX_MEMORY_LENGTH:
        return _tool_error("content_too_large", "content is too large", max_length=MAX_MEMORY_LENGTH)
    return None


def _validate_tags(tags: str) -> ToolError | None:
    if len(tags) > MAX_TAGS_LENGTH:
        return _tool_error("tags_too_long", "tags are too long", max_length=MAX_TAGS_LENGTH)
    return None


def _validate_memory_kind(memory_kind: str) -> ToolError | None:
    if memory_kind not in ALLOWED_MEMORY_KINDS:
        allowed = ", ".join(sorted(ALLOWED_MEMORY_KINDS))
        return _tool_error(
            "invalid_memory_kind",
            f"memory_kind must be one of {allowed}",
            allowed=sorted(ALLOWED_MEMORY_KINDS),
        )
    return None


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\u2026"


def _distill_session_turn(task: str, response: str) -> tuple[str, str]:
    clean_task = _single_line(task.strip())
    clean_response = response.strip()
    response_preview = _truncate(_single_line(clean_response), 140)
    summary = _truncate(f"{clean_task}: {response_preview}", 200)
    details = f"Task: {clean_task}\n\nResponse:\n{clean_response}"
    return summary, _truncate(details, MAX_MEMORY_LENGTH)


def _distill_session_summary(turns: list[dict]) -> tuple[str, str]:
    cleaned_turns: list[tuple[str, str]] = []
    topics: list[str] = []
    for turn in turns:
        user = _single_line(str(turn.get("user", "")).strip())
        assistant = str(turn.get("assistant", "")).strip()
        if not user or not assistant:
            continue
        cleaned_turns.append((user, assistant))
        topics.append(user)

    if not cleaned_turns:
        raise ValueError(
            "turns must contain at least one user/assistant pair using the keys 'user' and 'assistant'"
        )

    topic_preview = "; ".join(topics[:3])
    if len(topics) > 3:
        topic_preview += "; ..."
    summary = _truncate(f"Session summary: {topic_preview}", 200)

    lines = [f"Session covered {len(cleaned_turns)} turns."]
    for idx, (user, assistant) in enumerate(cleaned_turns, start=1):
        lines.append(f"\nTurn {idx}")
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {_truncate(assistant, 800)}")
    return summary, _truncate("\n".join(lines), MAX_MEMORY_LENGTH)


def _metadata_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("memory metadata contains invalid JSON")
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_datetime_filter(value: str, field_name: str) -> tuple[datetime | None, ToolError | None]:
    if not value.strip():
        return None, None
    try:
        return _normalize_datetime(value), None
    except ValueError:
        return None, _tool_error(
            "invalid_datetime",
            f"{field_name} must be an ISO 8601 date or datetime",
            field=field_name,
            value=value,
        )


def _validate_thread_id(thread_id: str) -> ToolError | None:
    if not thread_id.strip():
        return _tool_error("empty_thread_id", "thread_id is empty")
    if len(thread_id) > MAX_THREAD_ID_LENGTH:
        return _tool_error(
            "thread_id_too_long",
            "thread_id is too long",
            max_length=MAX_THREAD_ID_LENGTH,
        )
    return None


def _memory_thread_fields(result: dict) -> tuple[str | None, str | None]:
    metadata = _metadata_dict(result.get("metadata"))
    thread_meta = metadata.get("thread")
    thread_id = ""
    thread_title = ""
    if isinstance(thread_meta, dict):
        thread_id = str(thread_meta.get("id") or "").strip()
        thread_title = str(thread_meta.get("title") or "").strip()
    if not thread_id:
        thread_id = str(metadata.get("thread_id") or "").strip()
    if not thread_title:
        thread_title = str(metadata.get("thread_title") or "").strip()
    if not thread_id:
        return None, None
    return thread_id, thread_title or None


def _memory_event_datetime(result: dict) -> datetime | None:
    metadata = _metadata_dict(result.get("metadata"))
    for raw_value in (
        metadata.get("event_at"),
        result.get("updated_at"),
        result.get("created_at"),
    ):
        if not raw_value:
            continue
        try:
            return _normalize_datetime(str(raw_value))
        except ValueError:
            continue
    return None


def _apply_memory_filters(
    results: list[dict],
    *,
    thread_id: str = "",
    since: str = "",
    until: str = "",
) -> tuple[ToolError | None, list[dict]]:
    normalized_thread_id = thread_id.strip()
    if normalized_thread_id:
        thread_error = _validate_thread_id(normalized_thread_id)
        if thread_error:
            return thread_error, []

    since_dt, since_error = _parse_datetime_filter(since, "since")
    if since_error:
        return since_error, []
    until_dt, until_error = _parse_datetime_filter(until, "until")
    if until_error:
        return until_error, []
    if since_dt and until_dt and since_dt > until_dt:
        return _tool_error(
            "invalid_time_range",
            "since must be earlier than or equal to until",
            since=since,
            until=until,
        ), []

    filtered: list[dict] = []
    for result in results:
        result_thread_id, _thread_title = _memory_thread_fields(result)
        if normalized_thread_id and result_thread_id != normalized_thread_id:
            continue

        event_dt = _memory_event_datetime(result)
        if since_dt and (event_dt is None or event_dt < since_dt):
            continue
        if until_dt and (event_dt is None or event_dt > until_dt):
            continue
        filtered.append(result)

    return None, filtered


def _time_ago(dt_str: str | None) -> str:
    if not dt_str:
        return "recently"
    try:
        normalized = dt_str.replace("Z", "+00:00")
        then = datetime.fromisoformat(normalized)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - then
        if delta.days > 0:
            return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        return "just now"
    except (ValueError, TypeError):
        return "recently"


def _infer_session_topic(task: str) -> str:
    terms = _query_terms(task) - _COMMON_TASK_VERBS
    if not terms:
        terms = _query_terms(task)
    return sorted(terms, key=len, reverse=True)[0] if terms else "general"


def _infer_session_outcome(response: str) -> str:
    lowered = response.lower()
    if any(pattern in lowered for pattern in ("blocked", "cannot", "stuck", "failed to")):
        return "blocked"
    if any(pattern in lowered for pattern in TRANSIENT_STATUS_PATTERNS):
        return "completed"
    return "in_progress"


def _infer_session_metadata(task: str, response: str, existing_metadata: dict) -> dict:
    enriched = dict(existing_metadata)
    if "topic" not in enriched:
        enriched["topic"] = _infer_session_topic(task)
    if "outcome" not in enriched:
        enriched["outcome"] = _infer_session_outcome(response)
    if "session_ended_at" not in enriched:
        enriched["session_ended_at"] = datetime.now(timezone.utc).isoformat()

    if "files_touched" not in enriched:
        output = _git_command(["diff", "--name-only", "HEAD~1"], Path.cwd())
        enriched["files_touched"] = output.splitlines() if output else []

    if "decisions_made" not in enriched:
        decisions: list[str] = []
        for sentence in response.split("."):
            sentence_terms = _query_terms(sentence)
            if sentence_terms & DECISION_TERMS and len(sentence.strip()) > 20:
                decisions.append(_truncate(sentence.strip(), 120))
        enriched["decisions_made"] = decisions[:3]
    return enriched


def _should_skip_session_capture(response: str) -> bool:
    normalized = _single_line(response).lower()
    return normalized.startswith("i have no durable memory") or normalized.startswith(
        "no durable memory"
    )


def _rank_score(result: dict) -> float:
    if result.get("rank_score") is not None:
        return float(result["rank_score"])
    distance = result.get("distance")
    if distance is None:
        return 1.0
    return 1.0 / (1.0 + max(0.0, float(distance)))


def _vector_match_score(result: dict) -> float | None:
    distance = result.get("vector_distance")
    if distance is None:
        return None
    return 1.0 - float(distance)


def _memory_recency_boost(result: dict) -> float:
    """Gentle recency tiebreaker: 0.05 for brand-new, decays to 0 at 90 days."""
    created_at = result.get("created_at")
    if not created_at:
        return 0.0
    try:
        if isinstance(created_at, str):
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            ts = created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
        return 0.05 * max(0.0, 1.0 - age_days / 90.0)
    except (ValueError, TypeError, OverflowError):
        return 0.0


def _memory_rank_score(result: dict) -> float:
    distance = result.get("distance")
    if distance is None:
        base = 1.0
    else:
        base = 1.0 / (1.0 + max(0.0, float(distance)))
    return base + _memory_recency_boost(result)


def _with_source_db(result: MemoryRow, source_db: SourceDB) -> MemoryRow:
    return {**result, "source_db": source_db}


def _parse_memory_locator(
    raw_memory_id: str,
    *,
    error_code: str,
    error_field: str,
    error_message: str,
) -> tuple[SourceDB | None, int] | ToolError:
    memory_id = str(raw_memory_id).strip()
    source_db: SourceDB | None = None
    if ":" in memory_id:
        source_candidate, _, raw_id = memory_id.partition(":")
        if source_candidate in {"project", "user"}:
            source_db = cast(SourceDB, source_candidate)
            memory_id = raw_id.strip()
    sqlite_id = _int_or_none(memory_id)
    if sqlite_id is None:
        return _tool_error(
            error_code,
            error_message,
            **{error_field: raw_memory_id},
        )
    return source_db, sqlite_id


def _resolve_superseded_memory(
    memory_id: int, current_project_id: str, source_db: SourceDB | None = None
) -> tuple[object, SourceDB, MemoryRow] | ToolError:
    candidates: list[tuple[object, SourceDB, MemoryRow]] = []
    candidate_dbs: list[tuple[object, SourceDB]]
    if source_db == "project":
        candidate_dbs = [(_get_db(), "project")]
    elif source_db == "user":
        candidate_dbs = [(_get_user_db(), "user")]
    else:
        candidate_dbs = [(_get_db(), "project"), (_get_user_db(), "user")]
    for db, db_source in candidate_dbs:
        memory = db.get_memory(memory_id)
        if memory:
            candidates.append((db, db_source, memory))

    if not candidates:
        return _tool_error(
            "memory_not_found",
            f"memory {memory_id} not found",
            memory_id=memory_id,
        )
    if len(candidates) == 1:
        return candidates[0]

    current_project_matches = [
        candidate for candidate in candidates if candidate[2].get("project_id") == current_project_id
    ]
    if len(current_project_matches) == 1:
        return current_project_matches[0]

    return _tool_error(
        "ambiguous_old_memory_id",
        f"memory {memory_id} exists in multiple memory stores",
        memory_id=memory_id,
        source_dbs=[candidate[1] for candidate in candidates],
    )


def _result_key(result: dict) -> tuple[str, int | str]:
    index = result.get("chunk_index")
    if index is None:
        index = result.get("start_line") or ""
    return str(result.get("file_path") or ""), index


def _result_order_index(result: dict) -> int:
    index = result.get("chunk_index")
    if index is None:
        index = result.get("start_line")
    return int(index or 0)


def _result_base_fields(result: dict) -> dict:
    base: dict = {}
    for field in (
        "file_path",
        "chunk_index",
        "start_line",
        "end_line",
        "content",
        "language",
        "symbol",
        "indexed_at",
    ):
        if field in result:
            base[field] = result[field]
    return base


def _memory_priority(memory_kind: str | None) -> int:
    if not memory_kind:
        return len(STRUCTURED_MEMORY_PRIORITY)
    return STRUCTURED_MEMORY_PRIORITY.get(memory_kind, len(STRUCTURED_MEMORY_PRIORITY))


def _memory_stale_reasons(result: dict, current_project_id: str | None) -> list[str]:
    reasons: list[str] = []
    if result.get("superseded_by") is not None:
        reasons.append("superseded")
    result_project_id = result.get("project_id")
    if current_project_id and result_project_id and result_project_id != current_project_id:
        reasons.append("project_id_mismatch")
    return reasons


def _memory_state(result: dict, current_project_id: str | None) -> dict:
    stale_reasons = _memory_stale_reasons(result, current_project_id)
    result_project_id = result.get("project_id")
    return {
        "is_current_project": bool(current_project_id and result_project_id == current_project_id),
        "is_stale": bool(stale_reasons),
        "stale_reasons": stale_reasons,
    }


def _is_low_signal_auto_memory(result: dict) -> bool:
    metadata = result.get("metadata") or {}
    capture_kind = str(metadata.get("capture_kind") or "").strip()
    if capture_kind not in {"session_rollup", "session_distillation"}:
        return False
    summary = str(result.get("summary") or "").strip()
    content = str(result.get("content") or "").strip()
    task = str(metadata.get("task") or "").strip().lower()
    if len(summary) < 24:
        return True
    if metadata.get("turn_count") == 1 and len(content) < 180:
        return True
    if task in {"hi", "hello", "hey", "test"}:
        return True
    if any(pattern in task for pattern in BOILERPLATE_TASK_PATTERNS):
        return True
    return False


def _memory_capture_kind(result: dict) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("capture_kind") or "").strip()


def _is_auto_capture_memory(result: dict) -> bool:
    return _memory_capture_kind(result) in {"session_rollup", "session_distillation"}


def _is_low_signal_auto_capture(
    *,
    task: str,
    summary: str,
    content: str,
    capture_kind: str,
    turn_count: int | None = None,
) -> bool:
    return _is_low_signal_auto_memory(
        {
            "summary": summary,
            "content": content,
            "metadata": {
                "capture_kind": capture_kind,
                "task": task,
                "turn_count": turn_count,
            },
        }
    )


def _text_term_similarity(left: str, right: str) -> float:
    left_terms = _query_terms(left)
    right_terms = _query_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    overlap = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    return overlap / union if union else 0.0


def _has_durable_auto_memory_signal(task: str, summary: str, content: str) -> bool:
    haystack_terms = _query_terms(" ".join((task, summary, content)))
    return bool(haystack_terms & DURABLE_MEMORY_TERMS)


def _is_transient_status_auto_capture(task: str, summary: str, content: str) -> bool:
    haystack = " ".join((task, summary, content)).lower()
    if _has_durable_auto_memory_signal(task, summary, content):
        return False
    return any(pattern in haystack for pattern in TRANSIENT_STATUS_PATTERNS)


def _infer_auto_memory_kind(task: str, summary: str, content: str) -> MemoryKind:
    haystack_terms = _query_terms(" ".join((task, summary, content)))
    if haystack_terms & TODO_TERMS:
        return "todo"
    if haystack_terms & CONSTRAINT_TERMS:
        return "constraint"
    if haystack_terms & DECISION_TERMS:
        return "decision"
    if haystack_terms & FACT_TERMS:
        return "fact"
    return "summary"


def _auto_memory_recent_cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=AUTO_MEMORY_RECENCY_DAYS)).strftime("%Y-%m-%d %H:%M:%S")


def _recent_project_auto_memory_candidates(project_id: str) -> list[dict]:
    cutoff = _auto_memory_recent_cutoff()
    project_db = _get_db()
    user_db = _get_user_db()
    candidates = [
        _with_source_db(item, "project")
        for item in project_db.list_memories(
            limit=AUTO_MEMORY_SCAN_LIMIT,
            include_superseded=False,
            project_id=project_id,
            updated_since=cutoff,
        )
    ]
    candidates.extend(
        _with_source_db(item, "user")
        for item in user_db.list_memories(
            limit=AUTO_MEMORY_SCAN_LIMIT,
            include_superseded=False,
            project_id=project_id,
            updated_since=cutoff,
        )
    )
    return candidates


def _recent_user_auto_memory_candidates(user_db, *, include_superseded: bool) -> list[dict]:
    return user_db.list_memories(
        limit=AUTO_MEMORY_SCAN_LIMIT,
        include_superseded=include_superseded,
        updated_since=_auto_memory_recent_cutoff(),
    )


def _find_non_novel_auto_memory(
    *,
    project_id: str,
    summary: str,
    content: str,
) -> dict | None:
    candidates = _recent_project_auto_memory_candidates(project_id)
    for item in candidates:
        existing_content = str(item.get("content") or "")
        if _text_term_similarity(content, existing_content) >= 0.55:
            return item
    return None


def _find_merge_candidate(
    *,
    project_id: str,
    summary: str,
    content: str,
    memory_kind: str,
) -> dict | None:
    candidates = _recent_project_auto_memory_candidates(project_id)
    best: tuple[float, dict] | None = None
    for item in candidates:
        if str(item.get("memory_kind") or "") != memory_kind:
            continue
        similarity = _text_term_similarity(content, str(item.get("content") or ""))
        if similarity < 0.3:
            continue
        if best is None or similarity > best[0]:
            best = (similarity, item)
    return best[1] if best else None


def _merge_suggestion_payload(candidate: dict | None, project_id: str) -> dict | None:
    if not candidate:
        return None
    payload = _memory_payload(candidate, current_project_id=project_id)
    return {
        "action": "supersede",
        "memory_id": payload["id"],
        "memory_kind": payload["memory_kind"],
        "summary": payload["summary"],
    }


def _memory_rank_penalty(result: dict, current_project_id: str | None) -> int:
    state = _memory_state(result, current_project_id)
    penalty = 0
    if not state["is_current_project"] and result.get("source_db") == "user":
        penalty += 3
    if state["is_stale"]:
        penalty += 4
    capture_kind = _memory_capture_kind(result)
    if capture_kind == "session_distillation":
        penalty += 1
    if capture_kind == "session_rollup":
        penalty += 2
    if _is_low_signal_auto_memory(result):
        penalty += 3
    return penalty


def _normalized_auto_memory_key(
    summary: str,
    content: str,
    capture_kind: str,
    project_id: str | None,
) -> tuple[str, str, str, str]:
    return (
        str(project_id or ""),
        capture_kind.strip().lower(),
        _single_line(summary.strip()).lower(),
        _single_line(content.strip()).lower(),
    )


def _find_duplicate_auto_memory(
    *,
    user_db,
    project_id: str,
    summary: str,
    content: str,
    capture_kind: str,
) -> dict | None:
    target = _normalized_auto_memory_key(summary, content, capture_kind, project_id)
    existing = _recent_user_auto_memory_candidates(user_db, include_superseded=True)
    for item in existing:
        metadata = _metadata_dict(item.get("metadata"))
        if str(metadata.get("capture_kind") or "").strip() not in {"session_rollup", "session_distillation"}:
            continue
        candidate = _normalized_auto_memory_key(
            str(item.get("summary") or ""),
            str(item.get("content") or ""),
            str(metadata.get("capture_kind") or ""),
            str(item.get("project_id") or ""),
        )
        if candidate == target:
            return item
    return None


def _sort_memory_results(results: list[dict], current_project_id: str | None = None) -> list[dict]:
    return sorted(
        results,
        key=lambda item: (
            _memory_rank_penalty(item, current_project_id),
            -_memory_rank_score(item),
            _memory_priority(item.get("memory_kind")),
            str(item.get("updated_at") or ""),
            int(item.get("id") or 0),
        ),
    )


def _query_terms(query: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 2}


def _text_term_overlap(query: str, *parts: str | None) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    haystack = " ".join(part or "" for part in parts).lower()
    matches = sum(1 for term in terms if term in haystack)
    return matches / len(terms)


def _query_intents(query: str) -> set[str]:
    terms = _query_terms(query)
    intents: set[str] = set()
    if {"release", "publish", "tag", "workflow", "notes"} & terms:
        intents.add("release")
    if {"bootstrap", "hook", "session", "mcp", "trust", "config"} & terms:
        intents.add("bootstrap")
    if PROCEDURAL_QUERY_TERMS & terms:
        intents.add("procedural")
    if DOC_FOCUSED_QUERY_TERMS & terms:
        intents.add("docs")
    return intents


def _path_intent_boost(query: str, file_path: str) -> float:
    path = file_path.lower()
    terms = _query_terms(query)
    intents = _query_intents(query)
    boost = 0.0
    is_template_path = "/templates/" in path or "/template_bundle/" in path
    is_release_procedure = bool({"maintainer", "procedure", "steps", "guide"} & terms)
    is_release_automation = "release" in intents and "workflow" in terms and not is_release_procedure

    if "procedural" in intents:
        if path.endswith("agents.md"):
            boost += 0.7
            if path == "agents.md":
                boost += 0.25
        if path.endswith("changelog.md"):
            boost += 0.55
            if path == "changelog.md":
                boost += 0.15
        if "/docs/" in path or path.startswith("docs/") or path.endswith("readme.md"):
            boost += 0.35
        if ".github/workflows/" in path:
            boost += 0.65
            if path.startswith(".github/workflows/"):
                boost += 0.2
        if "/scripts/" in path or path.startswith("scripts/"):
            boost += 0.2
        if path.startswith("tests/"):
            boost -= 0.25
        if is_template_path:
            boost -= 0.4

    if "release" in intents:
        if ".github/workflows/" in path:
            boost += 0.45
            if path.endswith("publish.yml"):
                boost += 3.0
        if path.endswith("changelog.md"):
            boost += 0.35
        if path.endswith("agents.md"):
            boost += 0.25
        if is_release_automation:
            if path.endswith("changelog.md"):
                boost += 0.45
            if path.endswith("agents.md"):
                boost -= 0.45
            if path.startswith("docs/") and "setup" in path:
                boost -= 0.25
        if is_release_procedure:
            if path.endswith("agents.md"):
                boost += 0.25
        if is_template_path:
            boost -= 0.2

    if "bootstrap" in intents:
        if "hook" in path or "mcp" in path:
            boost += 0.4
        if "config" in path:
            boost += 0.25
        if "trust" in path:
            boost += 0.25
        if path == "readme.md":
            boost += 0.55
        if path.startswith("docs/") or "/docs/" in path:
            boost += 0.45
        if "setup" in path or "acp" in path:
            boost += 0.45
        if path.endswith("agents.md") or path.endswith("changelog.md"):
            boost -= 0.5

    if "docs" in intents:
        if path == "readme.md":
            boost += 0.85
        elif path.endswith("readme.md"):
            boost += 0.45
        if path.startswith("docs/"):
            boost += 0.75
        elif "/docs/" in path:
            boost += 0.35
        if "setup" in path:
            boost += 0.55
        if "guide" in path:
            boost += 0.4
        if "install" in path or "integration" in path:
            boost += 0.3
        if path.endswith("agents.md") or path.endswith("changelog.md"):
            boost -= 0.2
        if {"resume", "continue"} & terms:
            if path == "readme.md":
                boost += 0.6
            if path.endswith("changelog.md"):
                boost -= 0.15

    return boost


def _path_query_term_boost(query: str, file_path: str) -> float:
    """Small boost (up to 0.05) when any query term appears in the file path."""
    terms = _query_terms(query)
    if not terms:
        return 0.0
    path_parts = file_path.lower().replace("/", " ").replace("\\", " ").replace("_", " ").replace("-", " ").replace(".", " ")
    if any(term in path_parts for term in terms):
        return 0.05
    return 0.0


def _rerank_results(query: str, results: list[dict], *, content_key: str = "content") -> list[dict]:
    """Re-sort *results* by combining rank score with path/content term-overlap boosts."""
    for item in results:
        file_path = str(item.get("file_path") or "")
        path_boost = _path_query_term_boost(query, file_path)
        item["rank_score"] = float(item.get("rank_score") or 0.0) + path_boost
    return sorted(
        results,
        key=lambda item: (
            -(
                _rank_score(item)
                + _path_intent_boost(query, str(item.get("file_path") or ""))
                + 0.35 * _text_term_overlap(query, str(item.get("file_path") or ""))
                + 0.15 * _text_term_overlap(query, str(item.get(content_key) or ""))
            ),
            str(item.get("file_path") or ""),
        ),
    )


def _rerank_doc_results(query: str, results: list[dict]) -> list[dict]:
    """Re-sort doc *results* with intent-aware bonuses (e.g. CLAUDE.md priority, bootstrap/release heuristics)."""
    terms = _query_terms(query)
    intents = _query_intents(query)

    def doc_bonus(item: dict) -> float:
        path = str(item.get("file_path") or "").lower()
        bonus = 0.0
        if (
            path == "claude.md"
            or path.startswith("spec/")
            or path.endswith("plan.md")
            or path.endswith("-plan.md")
        ):
            bonus -= 1.0
        if {"resume", "continue"} & terms and path == "readme.md":
            bonus += 2.0
        if "bootstrap" in intents:
            if path == "readme.md":
                bonus += 1.0
            if "setup" in path or "acp" in path:
                bonus += 1.1
            if path.endswith("agents.md") or path.endswith("changelog.md"):
                bonus -= 1.1
        if "release" in intents:
            if path.endswith("changelog.md"):
                bonus += 0.75
            if path.endswith("agents.md") and not ({"maintainer", "procedure", "steps", "guide"} & terms):
                bonus -= 0.6
        return bonus

    return sorted(
        results,
        key=lambda item: (
            -(
                _rank_score(item)
                + doc_bonus(item)
                + 1.15 * _path_intent_boost(query, str(item.get("file_path") or ""))
                + 0.45 * _text_term_overlap(query, str(item.get("file_path") or ""))
                + 0.2 * _text_term_overlap(query, str(item.get("content") or ""))
            ),
            str(item.get("file_path") or ""),
            int(item.get("chunk_index") or 0),
        ),
    )


def _rrf_merge(*result_sets: tuple[str, list[dict]], k: int = 60, limit: int) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each *result_set* is a ``(source_name, results)`` pair.  The fused
    ``rank_score`` for each unique result is the sum of ``1/(k + rank)``
    across all lists it appears in.  Returns the top *limit* results.
    """
    merged: dict[tuple[str, int | str], dict] = {}
    for source_name, results in result_sets:
        for rank, item in enumerate(results, start=1):
            key = _result_key(item)
            payload = merged.setdefault(
                key,
                {
                    **_result_base_fields(item),
                    "rank_score": 0.0,
                    "match_sources": [],
                },
            )
            payload["rank_score"] = float(payload["rank_score"]) + (1.0 / (k + rank))
            if source_name not in payload["match_sources"]:
                payload["match_sources"].append(source_name)
            if source_name == "vector" and item.get("distance") is not None:
                payload["vector_distance"] = float(item["distance"])

    results = list(merged.values())
    results.sort(
        key=lambda item: (
            -float(item.get("rank_score") or 0.0),
            str(item.get("file_path") or ""),
            _result_order_index(item),
        )
    )
    return results[:limit]


def _match_reason(query: str, content: str, score: float) -> str:
    """Generate a human-readable match reason."""
    reasons: list[str] = []
    query_terms = query.lower().split()
    content_lower = content.lower()

    # Check for keyword overlap
    matching_terms = [t for t in query_terms if t in content_lower]
    if matching_terms:
        reasons.append(f"keyword match: {', '.join(matching_terms)}")

    # Semantic similarity bucket
    if score >= 0.85:
        reasons.append("strong semantic match")
    elif score >= 0.7:
        reasons.append("moderate semantic match")
    else:
        reasons.append("weak semantic match")

    return " + ".join(reasons) if reasons else "semantic similarity"


def _code_result_payload(result: RankedCodeResult, query: str = "") -> CodeSearchResult:
    """Convert a ranked DB row into a ``CodeSearchResult`` dict with provenance and match reason."""
    provenance: SearchProvenance = {
        "source": "project-index",
        "indexed_at": result.get("indexed_at"),
    }
    content = str(result["content"])
    rank_score = round(float(result.get("rank_score") or 0.0), 6)
    return {
        "file_path": str(result["file_path"]),
        "start_line": int(result["start_line"]),
        "end_line": int(result["end_line"]),
        "content": content,
        "language": result.get("language"),
        "symbol": result.get("symbol"),
        "indexed_at": result.get("indexed_at"),
        "rank_score": rank_score,
        "match_sources": list(result.get("match_sources") or []),
        "provenance": provenance,
        "match_reason": _match_reason(query, content, rank_score) if query else "semantic similarity",
    }


def _doc_result_payload(result: RankedDocResult, query: str = "") -> DocSearchResult:
    """Convert a ranked DB row into a ``DocSearchResult`` dict with provenance and match reason."""
    content = str(result["content"])
    provenance: SearchProvenance = {
        "source": "project-index",
        "indexed_at": result.get("indexed_at"),
    }
    rank_score = round(float(result.get("rank_score") or 0.0), 6)
    return {
        "file_path": str(result["file_path"]),
        "chunk_index": int(result["chunk_index"]),
        "content": content,
        "preview": content.replace("\n", " ")[:160],
        "indexed_at": result.get("indexed_at"),
        "rank_score": rank_score,
        "match_sources": list(result.get("match_sources") or []),
        "provenance": provenance,
        "match_reason": _match_reason(query, content, rank_score) if query else "semantic similarity",
    }


def _memory_payload(result: MemoryRow, current_project_id: str | None = None, query: str = "") -> MemoryPayload:
    """Convert a ``MemoryRow`` into a ``MemoryPayload`` with staleness, provenance, and thread fields."""
    metadata = result.get("metadata") or {}
    capture_kind = str(metadata.get("capture_kind") or "").strip() or "unknown"
    superseded_by = _int_or_none(result.get("superseded_by"))
    supersedes = _int_or_none(result.get("supersedes"))
    thread_id, thread_title = _memory_thread_fields(cast(dict, result))
    state = _memory_state(result, current_project_id)
    provenance: MemoryProvenance = {
        "capture_kind": capture_kind,
        "source_type": "structured" if result.get("memory_kind") != "note" else "freeform",
        "is_current_project": state["is_current_project"],
    }
    if capture_kind == "session_distillation":
        provenance["source_type"] = "session_distillation"
    elif capture_kind == "session_rollup":
        provenance["source_type"] = "session_rollup"
    elif capture_kind == "manual" and result.get("memory_kind") != "note":
        provenance["source_type"] = "manual_structured"

    memory_kind = str(result.get("memory_kind") or "note")
    if memory_kind not in ALLOWED_MEMORY_KINDS:
        memory_kind = "note"

    payload: MemoryPayload = {
        "id": int(result["id"]),
        "source_db": result.get("source_db"),
        "summary": result.get("summary") or result.get("content", "")[:200],
        "content": result.get("content", ""),
        "score": round(_memory_rank_score(result), 4),
        "project_id": result.get("project_id"),
        "memory_kind": cast(MemoryKind, memory_kind),
        "tags": result.get("tags") or [],
        "created_at": str(result.get("created_at")) if result.get("created_at") is not None else None,
        "updated_at": str(result.get("updated_at")) if result.get("updated_at") is not None else None,
        "source_session_id": result.get("source_session_id"),
        "source_message_id": result.get("source_message_id"),
        "supersedes": supersedes,
        "superseded_by": superseded_by,
        "is_superseded": superseded_by is not None,
        "is_stale": state["is_stale"],
        "stale_reasons": state["stale_reasons"],
        "metadata": metadata,
        "provenance": provenance,
    }
    if thread_id:
        payload["thread_id"] = thread_id
    if thread_title:
        payload["thread_title"] = thread_title
    if query:
        payload["match_reason"] = _match_reason(query, result.get("content", ""), payload["score"])
    if isinstance(payload["tags"], str):
        payload["tags"] = [tag.strip() for tag in payload["tags"].split(",") if tag.strip()]
    return payload


def _cleanup_candidate_reasons(result: dict, current_project_id: str | None) -> list[str]:
    reasons: list[str] = []
    metadata = result.get("metadata") or {}
    capture_kind = str(metadata.get("capture_kind") or "").strip()
    if result.get("memory_kind") == "note":
        reasons.append("freeform_note")
    if capture_kind == "freeform":
        reasons.append("freeform_capture")
    if result.get("superseded_by") is not None:
        reasons.append("superseded")
    if (
        result.get("source_db") == "user"
        and current_project_id
        and result.get("project_id")
        and result.get("project_id") != current_project_id
    ):
        reasons.append("cross_project_user_memory")
    summary = str(result.get("summary") or "").strip()
    if len(summary) < 24:
        reasons.append("short_summary")
    if _is_low_signal_auto_memory(result):
        reasons.append("low_signal_auto_memory")
    return reasons


def _cleanup_candidate_score(result: dict, current_project_id: str | None) -> tuple[int, str, str]:
    reasons = _cleanup_candidate_reasons(result, current_project_id)
    priority = 0
    if "superseded" in reasons:
        priority += 4
    if "cross_project_user_memory" in reasons:
        priority += 3
    if "freeform_note" in reasons or "freeform_capture" in reasons:
        priority += 2
    if "short_summary" in reasons:
        priority += 1
    if "low_signal_auto_memory" in reasons:
        priority += 2
    return priority, str(result.get("updated_at") or ""), str(result.get("id") or "")


def _memory_cleanup_candidates(limit: int = 10) -> list[dict]:
    current_project_id = _ensure_project_id()
    project_db = _get_db()
    user_db, _ = _optional_user_db()
    candidates: list[dict] = []

    candidates.extend(
        _with_source_db(result, "project")
        for result in project_db.list_memories(
            limit=max(limit * 3, 20),
            include_superseded=True,
            project_id=current_project_id,
        )
    )
    if user_db is not None:
        candidates.extend(
            _with_source_db(result, "user")
            for result in user_db.list_memories(
                limit=max(limit * 5, 30), include_superseded=True, project_id=current_project_id
            )
        )

    ranked = []
    for result in candidates:
        reasons = _cleanup_candidate_reasons(result, current_project_id)
        if not reasons:
            continue
        payload = _memory_payload(result, current_project_id=current_project_id)
        payload["cleanup_reasons"] = reasons
        payload["cleanup_priority"] = _cleanup_candidate_score(result, current_project_id)[0]
        ranked.append(payload)

    ranked.sort(
        key=lambda item: (
            -int(item.get("cleanup_priority", 0)),
            str(item.get("updated_at") or ""),
            str(item.get("id") or ""),
        )
    )
    return ranked[:limit]


def _all_memory_payloads() -> list[dict]:
    current_project_id = _ensure_project_id()
    project_db = _get_db()
    user_db, _ = _optional_user_db()
    payloads: list[dict] = []

    for result in project_db.list_memories(
        limit=max(project_db.memory_count() + 10, 20),
        include_superseded=True,
        project_id=current_project_id,
    ):
        payloads.append(
            _memory_payload(_with_source_db(result, "project"), current_project_id=current_project_id)
        )
    if user_db is not None:
        for result in user_db.list_memories(
            limit=max(user_db.memory_count() + 10, 20),
            include_superseded=True,
            project_id=current_project_id,
        ):
            payloads.append(
                _memory_payload(_with_source_db(result, "user"), current_project_id=current_project_id)
            )
    return payloads


def _duplicate_auto_memory_groups(payloads: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for item in payloads:
        provenance = item.get("provenance", {})
        capture_kind = str(provenance.get("capture_kind") or "").strip()
        if capture_kind not in {"session_rollup", "session_distillation"}:
            continue
        key = _normalized_auto_memory_key(
            str(item.get("summary") or ""),
            str(item.get("content") or ""),
            capture_kind,
            str(item.get("project_id") or ""),
        )
        groups.setdefault(key, []).append(item)

    duplicates: list[dict] = []
    for (_, capture_kind, _, _), items in groups.items():
        if len(items) < 2:
            continue
        sorted_items = sorted(items, key=lambda item: (str(item.get("updated_at") or ""), str(item.get("id") or "")))
        duplicates.append(
            {
                "count": len(sorted_items),
                "capture_kind": capture_kind,
                "project_id": sorted_items[0].get("project_id"),
                "summary": sorted_items[0].get("summary"),
                "memory_ids": [int(item.get("id")) for item in sorted_items if item.get("id") is not None],
            }
        )
    duplicates.sort(key=lambda item: (-int(item["count"]), str(item.get("summary") or "")))
    return duplicates


def _delete_memory_by_source_db(source_db: str, memory_id: int) -> bool:
    if memory_id is None:
        return False
    db = _get_db() if source_db == "project" else _get_user_db()
    deleted = db.forget(memory_id)
    return deleted is not None


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _merge_memory_results(
    *result_sets: list[dict],
    limit: int,
    current_project_id: str | None = None,
) -> list[dict]:
    seen_ids: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for results in result_sets:
        for result in results:
            dedupe_key = (str(result.get("source_db") or "unknown"), str(result["id"]))
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            merged.append(result)

    current_project_results = []
    for result in merged:
        state = _memory_state(result, current_project_id)
        if state["is_current_project"] and not state["is_stale"]:
            current_project_results.append(result)
    current_project_manual_results = [
        result
        for result in current_project_results
        if not _is_auto_capture_memory(result)
    ]
    if current_project_manual_results:
        merged = [
            result
            for result in merged
            if not (
                _is_auto_capture_memory(result)
                and _memory_state(result, current_project_id)["is_current_project"]
            )
        ]
    if current_project_results:
        merged = [
            result
            for result in merged
            if "project_id_mismatch" not in _memory_state(result, current_project_id)["stale_reasons"]
        ]

    return _sort_memory_results(merged, current_project_id=current_project_id)[:limit]


def _current_git_head_state(project_root: Path | None = None) -> tuple[str | None, str | None]:
    cwd = project_root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None, "git executable not found"
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "not a git repository" in stderr:
            return None, None
        return None, f"git rev-parse HEAD failed: {(result.stderr or result.stdout or '').strip() or result.returncode}"
    head = result.stdout.strip()
    if not head:
        return None, "git rev-parse HEAD returned no output"
    return head, None


def _git_command(args: list[str], cwd: Path, timeout: float = 2.0) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\n")


def _project_pulse(project_root: Path) -> dict:
    """Snapshot the project's git state: branch, workspace status, recent commits, and ahead/behind counts."""
    branch = _git_command(["branch", "--show-current"], project_root)
    if branch is None:
        return {
            "branch": None,
            "is_default_branch": None,
            "default_branch": None,
            "workspace": None,
            "recent_commits": [],
        }

    status_output = _git_command(["status", "--short"], project_root) or ""
    modified, staged, untracked = [], [], []
    for line in status_output.splitlines():
        if len(line) < 4:
            continue
        index_status = line[0]
        work_status = line[1]
        file_path = line[3:].strip()
        if index_status == "?":
            untracked.append(file_path)
        elif index_status != " ":
            staged.append(file_path)
        if work_status not in (" ", "?"):
            modified.append(file_path)

    workspace = {
        "modified": modified,
        "staged": staged,
        "untracked": untracked,
        "is_clean": not modified and not staged and not untracked,
    }

    log_output = _git_command(["log", "--oneline", "-5"], project_root) or ""
    recent_commits = []
    for line in log_output.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            recent_commits.append({"sha": parts[0], "message": parts[1]})

    ref_output = _git_command(["symbolic-ref", "refs/remotes/origin/HEAD"], project_root)
    default_branch = None
    is_default_branch = None
    if ref_output:
        default_branch = ref_output.split("/")[-1]
        is_default_branch = branch == default_branch

    pulse: dict = {
        "branch": branch,
        "is_default_branch": is_default_branch,
        "default_branch": default_branch,
        "workspace": workspace,
        "recent_commits": recent_commits,
    }

    if default_branch and not is_default_branch:
        ahead_str = _git_command(["rev-list", "--count", f"{default_branch}..HEAD"], project_root)
        behind_str = _git_command(["rev-list", "--count", f"HEAD..{default_branch}"], project_root)
        if ahead_str is not None:
            pulse["ahead"] = int(ahead_str)
        if behind_str is not None:
            pulse["behind"] = int(behind_str)

    return pulse


def _current_git_head(project_root: Path | None = None) -> str | None:
    head, _ = _current_git_head_state(project_root)
    return head


def _load_toml_state(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return tomllib.loads(path.read_text()), None
    except OSError as exc:
        return None, f"{path} unreadable: {exc}"
    except tomllib.TOMLDecodeError as exc:
        return None, f"{path} contains invalid TOML: {exc}"


def _load_toml(path: Path) -> dict | None:
    data, _ = _load_toml_state(path)
    return data


def _vibe_trust_status(project_root: Path) -> dict:
    config_path = Path.home() / ".vibe" / "trusted_folders.toml"
    data, error = _load_toml_state(config_path)
    if error:
        return {"status": "unknown", "detail": error}
    if not data:
        return {"status": "unknown", "detail": f"{config_path} not found"}

    trusted = {Path(entry).resolve() for entry in data.get("trusted", []) if isinstance(entry, str) and "$" not in entry}
    resolved = project_root.resolve()
    if resolved in trusted:
        return {"status": "ok", "detail": f"trusted in {config_path}"}
    return {"status": "warn", "detail": f"{resolved} not trusted in {config_path}"}


def _codex_trust_status(project_root: Path) -> dict:
    config_path = Path.home() / ".codex" / "config.toml"
    data, error = _load_toml_state(config_path)
    if error:
        return {"status": "unknown", "detail": error}
    if not data:
        return {"status": "unknown", "detail": f"{config_path} not found"}

    projects = data.get("projects")
    if not isinstance(projects, dict):
        return {"status": "unknown", "detail": f"{config_path} has no [projects] entries"}

    resolved = project_root.resolve()
    candidates = [resolved, *resolved.parents]
    for candidate in candidates:
        entry = projects.get(str(candidate))
        if isinstance(entry, dict) and entry.get("trust_level") == "trusted":
            return {"status": "ok", "detail": f"trusted via {candidate} in {config_path}"}
    return {"status": "warn", "detail": f"{resolved} not trusted in {config_path}"}


def _index_metadata(db) -> dict:
    metadata, error = db.get_setting_json_status(INDEX_METADATA_KEY)
    return {
        "metadata": metadata,
        "error": error,
    }


def _normalize_embedding_profile(profile: object) -> dict[str, object] | None:
    if not isinstance(profile, dict):
        return None
    provider = str(profile.get("provider") or "").strip().lower()
    model = str(profile.get("model") or "").strip()
    dimensions = _int_or_none(profile.get("dimensions"))
    if not provider or not model or dimensions is None or dimensions <= 0:
        return None
    return {
        "provider": provider,
        "model": model,
        "dimensions": dimensions,
    }


def _format_embedding_profile(profile: dict[str, object] | None) -> str:
    normalized = _normalize_embedding_profile(profile)
    if normalized is None:
        return "unknown"
    return (
        f"{normalized['provider']}:{normalized['model']}"
        f"@{normalized['dimensions']}"
    )


def _embedding_profile_state(db, metadata: dict | None = None) -> dict:
    current_profile = _normalize_embedding_profile(resolve_embedding_profile())
    indexed_profile = _normalize_embedding_profile((metadata or {}).get("embedding_profile"))
    has_index = db.code_chunk_count() > 0 or db.doc_count() > 0

    if not has_index:
        return {
            "current_profile": current_profile,
            "indexed_profile": indexed_profile,
            "is_incompatible": False,
            "warning": None,
        }

    if indexed_profile is None:
        return {
            "current_profile": current_profile,
            "indexed_profile": None,
            "is_incompatible": True,
            "warning": {
                "kind": "embedding_profile_missing",
                "detail": "index predates embedding profile tracking; run vibe-rag reindex",
            },
        }

    if current_profile != indexed_profile:
        return {
            "current_profile": current_profile,
            "indexed_profile": indexed_profile,
            "is_incompatible": True,
            "warning": {
                "kind": "embedding_profile_changed",
                "detail": (
                    "embedding profile changed since last index "
                    f"({_format_embedding_profile(indexed_profile)} -> {_format_embedding_profile(current_profile)})"
                ),
            },
        }

    return {
        "current_profile": current_profile,
        "indexed_profile": indexed_profile,
        "is_incompatible": False,
        "warning": None,
    }


def _project_index_paths(db, project_root: Path) -> list[str]:
    missing: list[str] = []
    for kind in ("code", "doc"):
        for rel_path in db.get_file_hashes(kind).keys():
            if not (project_root / rel_path).exists():
                missing.append(rel_path)
    return sorted(set(missing))


def _current_file_counts(project_root: Path) -> tuple[int, int]:
    code_files, doc_files = collect_files([project_root])
    return len(code_files), len(doc_files)


def _session_narrative(user_db, project_id: str, limit: int = 3) -> str | None:
    """Build a short prose recap of the most recent session memories for the load_session_context briefing."""
    all_memories = user_db.list_memories(
        limit=max(user_db.memory_count(include_superseded=True) + 10, 20),
        include_superseded=False,
        project_id=project_id,
    )
    session_memories = [
        memory
        for memory in all_memories
        if _metadata_dict(memory.get("metadata")).get("capture_kind")
        in ("session_distillation", "session_rollup")
    ]
    session_memories.sort(key=lambda memory: str(memory.get("updated_at") or ""), reverse=True)
    session_memories = session_memories[:limit]

    if not session_memories:
        return None

    parts: list[str] = []
    for idx, memory in enumerate(session_memories):
        metadata = _metadata_dict(memory.get("metadata"))
        topic = metadata.get("topic")
        outcome = metadata.get("outcome")
        decisions = metadata.get("decisions_made") or []
        summary = str(memory.get("summary") or "")

        if topic and outcome:
            sentence = f"working on {topic}"
            if outcome == "completed":
                sentence += " (completed)"
            elif outcome == "blocked":
                sentence += " (blocked)"
            if decisions:
                sentence += f". Decided: {'; '.join(decisions[:2])}"
        elif topic:
            sentence = f"working on {topic}"
        else:
            normalized_summary = re.sub(r"^\s*\d+\.\s*", "", summary)
            normalized_summary = re.sub(r"\s+\d+\.\s*", "; ", normalized_summary).strip()
            sentence = f"Last session: {_truncate(normalized_summary or summary, 100)}"

        if idx == 0:
            time_ago = _time_ago(memory.get("updated_at") and str(memory.get("updated_at")))
            if sentence.startswith("Last session:"):
                parts.append(f"You were last here {time_ago}. {sentence}.")
            else:
                parts.append(f"You were last here {time_ago} {sentence}.")
        else:
            parts.append(f"Before that: {sentence}.")

    return " ".join(parts)


def _hazard_scan(project_db, project_root: Path, project_id: str, pulse: dict) -> list[dict]:
    """Detect actionable hazards (missing index, stale index, provider down, etc.) for the session briefing."""
    hazards: list[dict] = []

    if project_db.code_chunk_count() == 0:
        hazards.append({"level": "error", "category": "no_index", "message": "No code index — run index_project before searching"})

    stale_state = _stale_state(project_db, project_root, project_id)
    incompatible_error = _incompatible_index_error(stale_state)
    if incompatible_error:
        hazards.append(
            {
                "level": "error",
                "category": "incompatible_index",
                "message": f"Incompatible index — {incompatible_error['message']}",
            }
        )

    if pulse.get("recent_commits"):
        current_head = pulse["recent_commits"][0].get("sha")
        metadata_state = _index_metadata(project_db)
        indexed_head = (metadata_state.get("metadata") or {}).get("git_head")
        if indexed_head and current_head and not indexed_head.startswith(current_head) and not current_head.startswith(indexed_head):
            hazards.append({"level": "warning", "category": "stale_index", "message": "Index may be stale — HEAD has changed since last index"})

    workspace = pulse.get("workspace")
    if workspace and not workspace.get("is_clean", True):
        modified = workspace.get("modified", [])
        staged = workspace.get("staged", [])
        total = len(set(modified + staged))
        if total > 0:
            hazards.append({"level": "warning", "category": "uncommitted_work", "message": f"{total} files modified but not committed"})

    try:
        # Look up through the package namespace so monkeypatching
        # ``vibe_rag.tools.embedding_provider_status`` in tests works.
        import vibe_rag.tools as _tools_pkg
        _ep_status = getattr(_tools_pkg, "embedding_provider_status", embedding_provider_status)
        provider_state = _ep_status()
        if not bool(provider_state.get("ok", False)):
            hazards.append({"level": "error", "category": "provider_unavailable", "message": f"Embedding provider not available — {provider_state.get('detail', 'unknown error')}"})
    except RuntimeError as exc:
        hazards.append({"level": "error", "category": "provider_unavailable", "message": f"Embedding provider status check failed: {exc}"})

    try:
        candidates = _memory_cleanup_candidates(limit=10)
        if len(candidates) > 5:
            hazards.append({"level": "warning", "category": "cleanup_pressure", "message": f"{len(candidates)} memories are cleanup candidates"})
    except (RuntimeError, sqlite3.OperationalError):
        pass

    hazards.sort(key=lambda h: 0 if h["level"] == "error" else 1)
    return hazards


def _live_decisions(project_db, user_db, project_id: str, limit: int = 3) -> list[dict]:
    """Return the most recent decision/constraint memories for the session briefing."""
    candidates: list[dict] = []
    db_sources = [(project_db, "project")]
    if user_db is not None:
        db_sources.append((user_db, "user"))
    for db, source_db in db_sources:
        memories = db.list_memories(
            limit=max(db.memory_count() + 10, 20),
            include_superseded=False,
            project_id=project_id,
        )
        for memory in memories:
            if memory.get("memory_kind") in ("decision", "constraint"):
                candidates.append(_with_source_db(memory, source_db))

    candidates.sort(key=lambda m: str(m.get("updated_at") or ""), reverse=True)
    payloads: list[dict] = []
    for item in candidates[:limit]:
        payload = _memory_payload(item, current_project_id=project_id)
        payload.pop("score", None)
        payloads.append(payload)
    return payloads


def _briefing_header(pulse: dict, project_id: str) -> str:
    branch = pulse.get("branch") or "unknown"
    workspace = pulse.get("workspace")
    if workspace is None:
        workspace_summary = "no git"
    elif workspace.get("is_clean"):
        workspace_summary = "clean"
    else:
        def _change_label(count: int, state: str) -> str:
            return f"{count} {state} file{'s' if count != 1 else ''}"

        mod_count = len(workspace.get("modified", []))
        staged_count = len(workspace.get("staged", []))
        untracked_count = len(workspace.get("untracked", []))
        parts = []
        if mod_count:
            parts.append(_change_label(mod_count, "modified"))
        if staged_count:
            parts.append(_change_label(staged_count, "staged"))
        if untracked_count:
            parts.append(_change_label(untracked_count, "untracked"))
        workspace_summary = ", ".join(parts) if parts else "dirty"

    header = f"vibe-rag | {project_id} | {branch} | {workspace_summary}"
    ahead = pulse.get("ahead")
    behind = pulse.get("behind")
    if ahead is not None or behind is not None:
        divergence_parts = []
        if ahead:
            divergence_parts.append(f"{ahead} ahead")
        if behind:
            divergence_parts.append(f"{behind} behind")
        if divergence_parts:
            header += f" ({', '.join(divergence_parts)})"
    return header


def _briefing_task_context(task_results: dict, char_budget: int) -> str:
    lines: list[str] = []
    code = task_results.get("code") or []
    if code:
        code_parts = []
        for item in code[:5]:
            symbol = item.get("symbol")
            path = item.get("file_path", "")
            start = item.get("start_line", "")
            if symbol:
                label = f"{path}:{start} {symbol}"
            else:
                preview = _single_line(str(item.get("content") or ""))
                label = f"{path}:{start}"
                if preview:
                    label += f" {_truncate(preview, 40)}"
            code_parts.append(label)
        lines.append("Code: " + " | ".join(code_parts))

    docs = task_results.get("docs") or []
    if docs:
        doc_parts = [item.get("file_path", "") for item in docs[:3]]
        lines.append("Docs: " + " | ".join(doc_parts))

    memories = task_results.get("memories") or []
    if memories:
        has_manual_current_project_memory = any(
            not _is_auto_capture_memory(item)
            and not bool(item.get("is_stale"))
            and bool((item.get("provenance") or {}).get("is_current_project"))
            for item in memories
        )
        visible_memories: list[dict] = []
        for item in memories:
            if item.get("is_stale"):
                continue
            if _is_low_signal_auto_memory(item):
                continue
            if has_manual_current_project_memory and _is_auto_capture_memory(item):
                continue
            visible_memories.append(item)
        mem_parts = [
            f"[{item.get('memory_kind', 'note')}] {_truncate(item.get('summary', ''), 80)}"
            for item in visible_memories[:3]
        ]
        if mem_parts:
            lines.append("Memory: " + " | ".join(mem_parts))

    result = "\n".join(lines)
    return result[:char_budget]


def _format_briefing(
    pulse: dict,
    narrative: str | None,
    hazards: list[dict],
    live_decisions: list[dict],
    task_results: dict,
    project_id: str,
) -> str:
    """Assemble the full session-start briefing string from its component sections.

    Combines the git-state header, session narrative, hazard warnings,
    live decisions, and task-context search results into a single
    string capped at ``BRIEFING_CHAR_BUDGET`` characters.
    """
    remaining = BRIEFING_CHAR_BUDGET
    sections: list[str] = []

    header = _briefing_header(pulse, project_id)
    sections.append(header)
    remaining -= len(header)

    if narrative:
        trimmed = _truncate(narrative, min(400, max(remaining, 0)))
        sections.append(trimmed)
        remaining -= len(trimmed)

    if hazards:
        hazard_lines = [f"! {hazard['message']}" for hazard in hazards[:3]]
        hazard_block = "\n".join(hazard_lines)
        sections.append(hazard_block)
        remaining -= len(hazard_block)

    if live_decisions:
        decision_lines = ["Decisions:"]
        for decision in live_decisions[:3]:
            summary = _truncate(decision.get("summary", ""), 80)
            kind = decision.get("memory_kind", "decision")
            updated = _time_ago(decision.get("updated_at"))
            decision_lines.append(f"- {summary} ({kind}, {updated})")
        decision_block = "\n".join(decision_lines)
        sections.append(decision_block)
        remaining -= len(decision_block)

    task_block = _briefing_task_context(task_results, max(remaining, 500))
    if task_block.strip():
        sections.append(task_block)

    briefing = "\n\n".join(sections)
    return briefing[:BRIEFING_CHAR_BUDGET]


def _stale_state(db, project_root: Path, project_id: str) -> dict:
    """Check whether the project index is stale or incompatible with the current embedding profile.

    Returns a dict with ``is_stale``, ``is_incompatible``, ``warnings``,
    ``metadata``, and the current/indexed embedding profiles.
    """
    metadata_state = _index_metadata(db)
    metadata = metadata_state.get("metadata")
    metadata_error = metadata_state.get("error")
    warnings: list[dict] = []

    if metadata_error:
        warnings.append(
            {
                "kind": "index_metadata_invalid",
                "detail": str(metadata_error),
            }
        )

    profile_state = _embedding_profile_state(db, metadata)
    if profile_state["warning"] is not None:
        warnings.append(profile_state["warning"])

    if not metadata:
        return {
            "is_stale": bool(warnings),
            "is_incompatible": bool(profile_state["is_incompatible"]),
            "warnings": warnings,
            "metadata": None,
            "current_profile": profile_state["current_profile"],
            "indexed_profile": profile_state["indexed_profile"],
        }
    current_head, git_head_error = _current_git_head_state(project_root)
    if git_head_error:
        warnings.append(
            {
                "kind": "git_head_unavailable",
                "detail": git_head_error,
            }
        )
    indexed_head = metadata.get("git_head")
    head_changed = False
    if indexed_head and current_head and indexed_head != current_head:
        head_changed = True
        warnings.append(
            {
                "kind": "git_head_changed",
                "detail": f"git HEAD changed since index ({indexed_head[:12]} -> {current_head[:12]})",
            }
        )

    indexed_total = int(metadata.get("code_file_count", 0)) + int(metadata.get("doc_file_count", 0))
    if not head_changed:
        current_code_count, current_doc_count = _current_file_counts(project_root)
        current_total = current_code_count + current_doc_count
        drift = abs(current_total - indexed_total)
        drift_threshold = max(5, int(indexed_total * 0.2)) if indexed_total else 5
        if drift >= drift_threshold:
            warnings.append(
                {
                    "kind": "file_count_drift",
                    "detail": f"indexed {indexed_total} files, current tree has {current_total}",
                }
            )

    missing = _project_index_paths(db, project_root)
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        warnings.append(
            {
                "kind": "indexed_files_missing",
                "detail": f"{len(missing)} indexed files no longer exist ({preview}{suffix})",
            }
        )

    indexed_project_id = metadata.get("project_id")
    if indexed_project_id and indexed_project_id != project_id:
        warnings.append(
            {
                "kind": "project_id_changed",
                "detail": f"indexed project_id {indexed_project_id} does not match current {project_id}",
            }
        )

    return {
        "is_stale": bool(warnings),
        "is_incompatible": bool(profile_state["is_incompatible"]),
        "warnings": warnings,
        "metadata": metadata,
        "current_profile": profile_state["current_profile"],
        "indexed_profile": profile_state["indexed_profile"],
    }


def _incompatible_index_error(state: dict) -> ToolError | None:
    if not state.get("is_incompatible"):
        return None
    detail = next(
        (
            str(warning.get("detail") or "")
            for warning in state.get("warnings", [])
            if str(warning.get("kind", "")).startswith("embedding_profile_")
        ),
        "embedding profile changed since last index; rebuild the full project index",
    )
    return _tool_error(
        "incompatible_index",
        detail,
        current_profile=state.get("current_profile"),
        indexed_profile=state.get("indexed_profile"),
    )


def _memory_limit_split(limit: int) -> tuple[int, int]:
    if limit <= 1:
        return limit, limit
    project_limit = max(1, limit // 2)
    user_limit = max(1, limit - project_limit)
    return project_limit, user_limit


def _emit_progress(progress_callback: ProgressCallback | None, **event: object) -> None:
    if progress_callback is None:
        return
    progress_callback(event)


def _embed_sync_with_progress(
    embed_method,
    texts: list[str],
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[list[float]]:
    if progress_callback is None:
        return embed_method(texts)
    try:
        signature = inspect.signature(embed_method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "progress_callback" not in signature.parameters:
        return embed_method(texts)
    try:
        return embed_method(texts, progress_callback=progress_callback)
    except TypeError:
        if signature is None:
            return embed_method(texts)
        raise


def _validate_embedding_count(
    items: list[dict], embeddings: list[list[float]], *, kind: str
) -> None:
    if len(items) != len(embeddings):
        raise RuntimeError(
            f"Embedding count mismatch for {kind}: {len(items)} items but {len(embeddings)} embeddings"
        )


def _search_code_results(
    query: str,
    limit: int = 10,
    language: str | None = None,
    min_score: float = 0.0,
) -> tuple[ToolError | None, list[dict]]:
    """Run a semantic+lexical code search, returning ``(error_or_None, ranked_results)``.

    Combines vector search with lexical search via RRF merge, then re-ranks.
    Returns a ``ToolError`` as the first element on validation or embedding failures.
    """
    error = _validate_query(query)
    if error:
        return error, []

    if language and language not in set(EXT_TO_LANG.values()):
        return _tool_error("unknown_language", f"unknown language '{language}'", language=language), []

    db = _get_db()
    if db.code_chunk_count() == 0:
        return _tool_error("no_code_index", "no code index; run index_project first"), []
    state = _stale_state(db, Path.cwd(), _ensure_project_id())
    incompatible_error = _incompatible_index_error(state)
    if incompatible_error:
        return incompatible_error, []

    try:
        embeddings = _get_embedder().embed_code_query_sync([query])
    except Exception as e:
        return _tool_error("embedding_failed", f"embedding failed: {e}", operation="search_code"), []

    raw_results = db.search_code(embeddings[0], limit=max(limit * 5, 10), language=language)
    lexical_results = db.lexical_search_code(sorted(_query_terms(query)), limit=max(limit * 5, 10))
    workflow_results: list[dict] = []
    terms = _query_terms(query)
    if {"release", "workflow"} <= terms or ({"release", "publish"} <= terms and "tag" in terms):
        workflow_results = db.lexical_search_code(
            ["publish.yml", ".github/workflows", "release", "publish"],
            limit=max(limit * 2, 5),
        )
    reranked = _rerank_results(
        query,
        _rrf_merge(
            ("vector", raw_results),
            ("lexical", lexical_results),
            ("workflow", workflow_results),
            limit=max(limit * 10, 20),
        ),
    )
    if min_score <= 0:
        filtered = reranked[:limit]
    else:
        filtered = [
            result
            for result in reranked
            if (_vector_match_score(result) is not None and _vector_match_score(result) >= min_score)
        ][:limit]
    return None, filtered


def _search_docs_results(query: str, limit: int = 10) -> tuple[ToolError | None, list[dict]]:
    """Run a semantic+lexical doc search, returning ``(error_or_None, ranked_results)``."""
    error = _validate_query(query)
    if error:
        return error, []

    db = _get_db()
    if db.doc_count() == 0:
        return _tool_error("no_docs_index", "no docs indexed; run index_project first"), []
    state = _stale_state(db, Path.cwd(), _ensure_project_id())
    incompatible_error = _incompatible_index_error(state)
    if incompatible_error:
        return incompatible_error, []

    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return _tool_error("embedding_failed", f"embedding failed: {e}", operation="search_docs"), []

    raw_results = db.search_docs(embeddings[0], limit=max(limit * 5, 10))
    lexical_results = db.lexical_search_docs(sorted(_query_terms(query)), limit=max(limit * 5, 10))
    results = _rerank_doc_results(
        query,
        _rrf_merge(
            ("vector", raw_results),
            ("lexical", lexical_results),
            limit=max(limit * 10, 20),
        ),
    )
    results = results[:limit]
    return None, results


def _search_memory_results(
    query: str,
    limit: int = 10,
    tags: str = "",
    thread_id: str = "",
    since: str = "",
    until: str = "",
    *,
    search_all_user_projects: bool = False,
) -> tuple[ToolError | None, list[dict]]:
    """Run a semantic memory search across project and user DBs, returning ``(error_or_None, results)``.

    Merges results from both databases, then applies optional post-filters
    (tags, thread_id, date range).
    """
    error = _validate_query(query)
    if error:
        return error, []
    error = _validate_tags(tags)
    if error:
        return error, []

    project_db = _get_db()
    user_db, user_db_error = _optional_user_db()
    project_memory_count = project_db.memory_count()
    user_memory_count = user_db.memory_count() if user_db is not None else 0
    if project_memory_count == 0 and user_memory_count == 0:
        return user_db_error or _tool_error("no_memories", "no memories stored yet"), []
    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return _tool_error("embedding_failed", f"embedding failed: {e}", operation="search_memory"), []

    # Fetch more results when filtering so we have enough after the post-filter.
    has_filters = bool(tags or thread_id.strip() or since.strip() or until.strip())
    fetch_limit = limit * 5 if has_filters else limit
    current_project_id = _ensure_project_id()
    project_limit, user_limit = _memory_limit_split(fetch_limit)
    project_results = project_db.search_memories(
        embeddings[0], limit=project_limit, project_id=current_project_id
    )
    # Keep search_all_user_projects in-band for compatibility, but remain project-scoped to prevent
    # accidental cross-project leakage when project boundaries are expected.
    user_results = []
    if user_db is not None:
        user_results = user_db.search_memories(
            embeddings[0], limit=max(user_limit, 10), project_id=current_project_id
        )
    results = _merge_memory_results(
        [_with_source_db(result, "project") for result in project_results],
        [_with_source_db(result, "user") for result in user_results],
        limit=fetch_limit,
        current_project_id=current_project_id,
    )

    # Post-filter by tags when requested.
    if tags:
        requested_tags = {t.strip().lower() for t in tags.split(",") if t.strip()}
        if requested_tags:
            filtered: list[dict] = []
            for result in results:
                raw_tags = result.get("tags") or ""
                if isinstance(raw_tags, list):
                    memory_tags = {t.strip().lower() for t in raw_tags if t.strip()}
                else:
                    memory_tags = {t.strip().lower() for t in str(raw_tags).split(",") if t.strip()}
                if memory_tags & requested_tags:
                    filtered.append(result)
            results = filtered[:limit]

    filter_error, results = _apply_memory_filters(
        results,
        thread_id=thread_id,
        since=since,
        until=until,
    )
    if filter_error:
        return filter_error, []

    results = results[:limit]
    return user_db_error, results


def _list_thread_memory_results(
    thread_id: str,
    *,
    limit: int = 20,
    scope: str = "all",
    since: str = "",
    until: str = "",
    include_superseded: bool = False,
) -> tuple[ToolError | None, list[dict]]:
    thread_error = _validate_thread_id(thread_id)
    if thread_error:
        return thread_error, []
    if scope not in ("all", "project", "user"):
        return _tool_error(
            "invalid_scope",
            "scope must be 'all', 'project', or 'user'",
            scope=scope,
        ), []

    current_project_id = _ensure_project_id()
    project_db = _get_db()
    user_db = _get_user_db()
    candidates: list[dict] = []

    if scope in ("all", "project"):
        project_limit = max(project_db.memory_count(include_superseded=True) + 10, limit * 5, 20)
        for memory in project_db.list_memories(
            limit=project_limit,
            include_superseded=include_superseded,
            project_id=current_project_id,
        ):
            candidates.append(_with_source_db(memory, "project"))

    if scope in ("all", "user"):
        user_limit = max(user_db.memory_count(include_superseded=True) + 10, limit * 5, 20)
        for memory in user_db.list_memories(
            limit=user_limit,
            include_superseded=include_superseded,
        ):
            candidates.append(_with_source_db(memory, "user"))

    filter_error, filtered = _apply_memory_filters(
        candidates,
        thread_id=thread_id,
        since=since,
        until=until,
    )
    if filter_error:
        return filter_error, []

    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    filtered.sort(
        key=lambda item: (
            _memory_event_datetime(item) or min_dt,
            int(item.get("id") or 0),
        ),
        reverse=True,
    )
    return None, filtered[:limit]


def _index_skip_summary(code_skipped: int, doc_skipped: int) -> str:
    if code_skipped == 0 and doc_skipped == 0:
        return ""
    parts = []
    if code_skipped:
        parts.append(f"{code_skipped} code skipped")
    if doc_skipped:
        parts.append(f"{doc_skipped} docs skipped")
    return f" ({', '.join(parts)})"
