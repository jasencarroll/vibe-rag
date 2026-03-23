from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
import tomllib
from typing import Any, cast

from vibe_rag.chunking import chunk_doc, collect_files
from vibe_rag.constants import EXT_TO_LANG
from vibe_rag.indexing.code_chunker import chunk_code
from vibe_rag.server import (
    _ensure_project_id,
    _get_db,
    _get_embedder,
    _get_user_db,
    mcp,
)
from vibe_rag.indexing.embedder import ProgressCallback
from vibe_rag.types import CodeSearchResult, DocSearchResult, MemoryKind, MemoryPayload, ToolError

MAX_QUERY_LENGTH = 10_000
MAX_MEMORY_LENGTH = 10_000
MAX_TAGS_LENGTH = 512
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
STRUCTURED_MEMORY_PRIORITY = {
    "decision": 0,
    "constraint": 1,
    "summary": 2,
    "todo": 3,
    "fact": 4,
    "note": 5,
}
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
    return {
        "code": code,
        "message": message,
        "details": details,
    }


def _failure(code: str, message: str, **details: Any) -> dict:
    return {"ok": False, "error": _tool_error(code, message, **details)}


def _success(**payload: Any) -> dict:
    return {"ok": True, **payload}


def _failure_from_error(error: ToolError, **details: Any) -> dict:
    merged = dict(error.get("details") or {})
    merged.update(details)
    return _failure(error["code"], error["message"], **merged)


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
    if not query.strip():
        return _tool_error("empty_query", "query is empty")
    if len(query) > MAX_QUERY_LENGTH:
        return _tool_error("query_too_long", "query is too long", max_length=MAX_QUERY_LENGTH)
    return None


def _validate_memory_content(content: str) -> ToolError | None:
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
    return text[: max_chars - 1].rstrip() + "…"


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


def _memory_rank_score(result: dict) -> float:
    distance = result.get("distance")
    if distance is None:
        return 1.0
    return 1.0 / (1.0 + max(0.0, float(distance)))


def _with_source_db(result: dict, source_db: str) -> dict:
    return {**result, "source_db": source_db}


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


def _find_non_novel_auto_memory(
    *,
    project_id: str,
    summary: str,
    content: str,
) -> dict | None:
    current_project_id = project_id
    candidates: list[dict] = []
    project_db = _get_db()
    user_db = _get_user_db()
    candidates.extend(
        _with_source_db(item, "project")
        for item in project_db.list_memories(
            limit=max(project_db.memory_count(include_superseded=True) + 10, 20),
            include_superseded=False,
            project_id=current_project_id,
        )
    )
    for item in user_db.list_memories(limit=max(user_db.memory_count(include_superseded=True) + 10, 20), include_superseded=False):
        if str(item.get("project_id") or "") == current_project_id:
            candidates.append(_with_source_db(item, "user"))
    for item in candidates:
        existing_summary = str(item.get("summary") or "")
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
    project_db = _get_db()
    user_db = _get_user_db()
    candidates: list[dict] = []
    candidates.extend(
        _with_source_db(item, "project")
        for item in project_db.list_memories(
            limit=max(project_db.memory_count(include_superseded=True) + 10, 20),
            include_superseded=False,
            project_id=project_id,
        )
    )
    for item in user_db.list_memories(limit=max(user_db.memory_count(include_superseded=True) + 10, 20), include_superseded=False):
        if str(item.get("project_id") or "") == project_id:
            candidates.append(_with_source_db(item, "user"))
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
    existing = user_db.list_memories(
        limit=max(user_db.memory_count(include_superseded=True) + 10, 20),
        include_superseded=True,
    )
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
        ),
    )


def _query_terms(query: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 3}


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


def _rerank_results(query: str, results: list[dict], *, content_key: str = "content") -> list[dict]:
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


def _code_result_payload(result: dict) -> CodeSearchResult:
    return {
        "file_path": str(result["file_path"]),
        "start_line": int(result["start_line"]),
        "end_line": int(result["end_line"]),
        "content": str(result["content"]),
        "language": result.get("language"),
        "symbol": result.get("symbol"),
        "indexed_at": result.get("indexed_at"),
        "rank_score": round(float(result.get("rank_score") or 0.0), 6),
        "match_sources": list(result.get("match_sources") or []),
        "provenance": {
            "source": "project-index",
            "indexed_at": result.get("indexed_at"),
        },
    }


def _doc_result_payload(result: dict) -> DocSearchResult:
    content = str(result["content"])
    return {
        "file_path": str(result["file_path"]),
        "chunk_index": int(result["chunk_index"]),
        "content": content,
        "preview": content.replace("\n", " ")[:160],
        "indexed_at": result.get("indexed_at"),
        "rank_score": round(float(result.get("rank_score") or 0.0), 6),
        "match_sources": list(result.get("match_sources") or []),
        "provenance": {
            "source": "project-index",
            "indexed_at": result.get("indexed_at"),
        },
    }


def _memory_payload(result: dict, current_project_id: str | None = None) -> MemoryPayload:
    metadata = result.get("metadata") or {}
    capture_kind = str(metadata.get("capture_kind") or "").strip() or "unknown"
    superseded_by = _int_or_none(result.get("superseded_by"))
    supersedes = _int_or_none(result.get("supersedes"))
    state = _memory_state(result, current_project_id)
    provenance = {
        "source_db": result.get("source_db"),
        "project_id": result.get("project_id"),
        "capture_kind": capture_kind,
        "source_type": "structured" if result.get("memory_kind") != "note" else "freeform",
        "source_session_id": result.get("source_session_id"),
        "source_message_id": result.get("source_message_id"),
        "created_at": str(result.get("created_at")) if result.get("created_at") is not None else None,
        "updated_at": str(result.get("updated_at")) if result.get("updated_at") is not None else None,
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
    user_db = _get_user_db()
    candidates: list[dict] = []

    candidates.extend(
        _with_source_db(result, "project")
        for result in project_db.list_memories(
            limit=max(limit * 3, 20),
            include_superseded=True,
            project_id=current_project_id,
        )
    )
    candidates.extend(
        _with_source_db(result, "user")
        for result in user_db.list_memories(limit=max(limit * 5, 30), include_superseded=True)
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
    user_db = _get_user_db()
    payloads: list[dict] = []

    for result in project_db.list_memories(
        limit=max(project_db.memory_count() + 10, 20),
        include_superseded=True,
        project_id=current_project_id,
    ):
        payloads.append(
            _memory_payload(_with_source_db(result, "project"), current_project_id=current_project_id)
        )
    for result in user_db.list_memories(limit=max(user_db.memory_count() + 10, 20), include_superseded=True):
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

    current_project_results = [
        result
        for result in merged
        if _memory_state(result, current_project_id)["is_current_project"]
        and not _memory_state(result, current_project_id)["is_stale"]
    ]
    if current_project_results:
        merged = [
            result
            for result in merged
            if "project_id_mismatch" not in _memory_state(result, current_project_id)["stale_reasons"]
        ]

    if any(not _is_auto_capture_memory(result) for result in current_project_results):
        merged = [
            result
            for result in merged
            if not (
                _memory_state(result, current_project_id)["is_current_project"]
                and _is_auto_capture_memory(result)
            )
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


def _stale_state(db, project_root: Path, project_id: str) -> dict:
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

    if not metadata:
        return {
            "is_stale": bool(warnings),
            "warnings": warnings,
            "metadata": None,
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
    if indexed_head and current_head and indexed_head != current_head:
        warnings.append(
            {
                "kind": "git_head_changed",
                "detail": f"git HEAD changed since index ({indexed_head[:12]} -> {current_head[:12]})",
            }
        )

    indexed_total = int(metadata.get("code_file_count", 0)) + int(metadata.get("doc_file_count", 0))
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
        "warnings": warnings,
        "metadata": metadata,
    }


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
    error = _validate_query(query)
    if error:
        return error, []

    if language and language not in set(EXT_TO_LANG.values()):
        return _tool_error("unknown_language", f"unknown language '{language}'", language=language), []

    db = _get_db()
    if db.code_chunk_count() == 0:
        return _tool_error("no_code_index", "no code index; run index_project first"), []

    try:
        embeddings = _get_embedder().embed_code_sync([query])
    except RuntimeError as e:
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
    error = _validate_query(query)
    if error:
        return error, []

    db = _get_db()
    if db.doc_count() == 0:
        return _tool_error("no_docs_index", "no docs indexed; run index_project first"), []

    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except RuntimeError as e:
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


def _search_memory_results(query: str, limit: int = 10) -> tuple[ToolError | None, list[dict]]:
    error = _validate_query(query)
    if error:
        return error, []

    project_db = _get_db()
    user_db = _get_user_db()
    if project_db.memory_count() == 0 and user_db.memory_count() == 0:
        return _tool_error("no_memories", "no memories stored yet"), []
    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except RuntimeError as e:
        return _tool_error("embedding_failed", f"embedding failed: {e}", operation="search_memory"), []

    current_project_id = _ensure_project_id()
    project_limit, user_limit = _memory_limit_split(limit)
    project_results = project_db.search_memories(
        embeddings[0], limit=project_limit, project_id=current_project_id
    )
    user_results = user_db.search_memories(embeddings[0], limit=max(user_limit, 10))
    results = _merge_memory_results(
        [_with_source_db(result, "project") for result in project_results],
        [_with_source_db(result, "user") for result in user_results],
        limit=limit,
        current_project_id=current_project_id,
    )
    return None, results


def _index_project_impl(
    paths: list[str] | str | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    start = time.time()

    normalized = _normalize_paths(paths)
    if isinstance(normalized, str):
        return _failure("invalid_path", normalized.removeprefix("Error: ").strip(), paths=paths)
    root_paths, project_root = normalized

    try:
        embedder = _get_embedder()
    except RuntimeError as e:
        return _failure("embedding_provider_unavailable", str(e))

    code_files, doc_files = collect_files(root_paths)
    if not code_files and not doc_files:
        return _failure("no_files_found", "no files found to index", paths=paths)
    _emit_progress(
        progress_callback,
        phase="file_discovery_complete",
        code_file_total=len(code_files),
        doc_file_total=len(doc_files),
        project_root=str(project_root),
    )

    db = _get_db()
    code_hashes = db.get_file_hashes("code")
    doc_hashes = db.get_file_hashes("doc")
    if code_hashes and db.code_chunk_count() == 0:
        db.delete_file_hashes(list(code_hashes), kind="code")
        code_hashes = {}
    if doc_hashes and db.doc_count() == 0:
        db.delete_file_hashes(list(doc_hashes), kind="doc")
        doc_hashes = {}
    current_code_paths = {_relative_to_project(path, project_root) for path in code_files}
    current_doc_paths = {_relative_to_project(path, project_root) for path in doc_files}

    for stale in sorted(set(code_hashes) - current_code_paths):
        db.delete_file_chunks(stale, kind="code")
    for stale in sorted(set(doc_hashes) - current_doc_paths):
        db.delete_file_chunks(stale, kind="doc")

    code_chunks: list[dict] = []
    code_embeddings_input: list[str] = []
    code_updates: list[tuple[str, str, list[dict]]] = []
    doc_chunks: list[dict] = []
    doc_embeddings_input: list[str] = []
    doc_updates: list[tuple[str, str, list[dict]]] = []
    code_unchanged = 0
    doc_unchanged = 0
    code_skipped = 0
    doc_skipped = 0
    _emit_progress(progress_callback, phase="code_chunking_start", file_total=len(code_files))

    for path in code_files:
        rel_path = _relative_to_project(path, project_root)
        try:
            content = path.read_text(errors="replace")
        except Exception:
            code_skipped += 1
            continue
        digest = _content_hash(content)
        if code_hashes.get(rel_path) == digest:
            code_unchanged += 1
            continue

        language = EXT_TO_LANG.get(path.suffix)
        file_chunks = chunk_code(content, rel_path, language)
        code_updates.append((rel_path, digest, file_chunks))
        code_chunks.extend(file_chunks)
        code_embeddings_input.extend(chunk["content"] for chunk in file_chunks)

    _emit_progress(
        progress_callback,
        phase="code_chunking_complete",
        file_total=len(code_files),
        chunk_total=len(code_chunks),
        unchanged_total=code_unchanged,
        skipped_total=code_skipped,
    )
    _emit_progress(progress_callback, phase="doc_chunking_start", file_total=len(doc_files))

    for path in doc_files:
        rel_path = _relative_to_project(path, project_root)
        try:
            content = path.read_text(errors="replace")
        except Exception:
            doc_skipped += 1
            continue
        digest = _content_hash(content)
        if doc_hashes.get(rel_path) == digest:
            doc_unchanged += 1
            continue

        file_chunks = chunk_doc(content, rel_path)
        doc_updates.append((rel_path, digest, file_chunks))
        doc_chunks.extend(file_chunks)
        doc_embeddings_input.extend(chunk["content"] for chunk in file_chunks)

    _emit_progress(
        progress_callback,
        phase="doc_chunking_complete",
        file_total=len(doc_files),
        chunk_total=len(doc_chunks),
        unchanged_total=doc_unchanged,
        skipped_total=doc_skipped,
    )

    try:
        code_embeddings: list[list[float]] = []
        if code_chunks:
            _emit_progress(
                progress_callback,
                phase="code_embedding_start",
                chunk_total=len(code_chunks),
            )
            code_embeddings = _embed_sync_with_progress(
                embedder.embed_code_sync,
                code_embeddings_input,
                progress_callback=progress_callback,
            )
            _validate_embedding_count(code_chunks, code_embeddings, kind="code chunks")
        doc_embeddings: list[list[float]] = []
        if doc_chunks:
            _emit_progress(
                progress_callback,
                phase="doc_embedding_start",
                chunk_total=len(doc_chunks),
            )
            doc_embeddings = _embed_sync_with_progress(
                embedder.embed_text_sync,
                doc_embeddings_input,
                progress_callback=progress_callback,
            )
            _validate_embedding_count(doc_chunks, doc_embeddings, kind="doc chunks")
    except Exception as e:
        return _failure("indexing_failed", f"indexing failed: {e}")

    code_offset = 0
    for rel_path, digest, file_chunks in code_updates:
        db.delete_file_chunks(rel_path, kind="code")
        file_embedding_count = len(file_chunks)
        if file_embedding_count:
            db.upsert_chunks(
                file_chunks,
                code_embeddings[code_offset:code_offset + file_embedding_count],
            )
        db.set_file_hash(rel_path, digest, "code")
        code_offset += file_embedding_count

    doc_offset = 0
    for rel_path, digest, file_chunks in doc_updates:
        db.delete_file_chunks(rel_path, kind="doc")
        file_embedding_count = len(file_chunks)
        if file_embedding_count:
            db.upsert_docs(
                file_chunks,
                doc_embeddings[doc_offset:doc_offset + file_embedding_count],
            )
        db.set_file_hash(rel_path, digest, "doc")
        doc_offset += file_embedding_count

    db.set_setting_json(
        INDEX_METADATA_KEY,
        {
            "project_id": _ensure_project_id(),
            "project_root": str(project_root),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": _current_git_head(project_root),
            "code_file_count": len(current_code_paths),
            "doc_file_count": len(current_doc_paths),
            "code_chunk_count": db.code_chunk_count(),
            "doc_chunk_count": db.doc_count(),
        },
    )

    elapsed = time.time() - start
    _emit_progress(
        progress_callback,
        phase="index_complete",
        elapsed_seconds=round(elapsed, 1),
        code_file_total=len(code_files),
        doc_file_total=len(doc_files),
        code_chunk_total=db.code_chunk_count(),
        doc_chunk_total=db.doc_count(),
    )
    summary = (
        f"Indexed {len(code_files)} code files ({len(code_chunks)} chunks, {code_unchanged} unchanged), "
        f"{len(doc_files)} docs ({len(doc_chunks)} chunks, {doc_unchanged} unchanged) in {elapsed:.1f}s"
        f"{_index_skip_summary(code_skipped, doc_skipped)}"
    )
    return _success(
        summary=summary,
        project_id=_ensure_project_id(),
        project_root=str(project_root),
        elapsed_seconds=round(elapsed, 1),
        counts={
            "code_files": len(code_files),
            "doc_files": len(doc_files),
            "code_chunks": len(code_chunks),
            "doc_chunks": len(doc_chunks),
            "code_unchanged": code_unchanged,
            "doc_unchanged": doc_unchanged,
            "code_skipped": code_skipped,
            "doc_skipped": doc_skipped,
            "indexed_code_chunks": db.code_chunk_count(),
            "indexed_doc_chunks": db.doc_count(),
        },
        warnings=[],
    )


def _index_skip_summary(code_skipped: int, doc_skipped: int) -> str:
    if code_skipped == 0 and doc_skipped == 0:
        return ""
    parts = []
    if code_skipped:
        parts.append(f"{code_skipped} code skipped")
    if doc_skipped:
        parts.append(f"{doc_skipped} docs skipped")
    return f" ({', '.join(parts)})"


@mcp.tool()
def index_project(paths: list[str] | str | None = None) -> dict:
    """Index project source files and docs for semantic search. Prefer this before grep when exploring a repo."""
    return _index_project_impl(paths)


@mcp.tool()
def search_code(
    query: str,
    limit: int = 10,
    language: str | None = None,
    min_score: float = 0.0,
) -> dict:
    """Semantic code search. Prefer this over grep when you know behavior but not exact symbols or filenames."""
    error, filtered = _search_code_results(query, limit=limit, language=language, min_score=min_score)
    if error:
        return _failure_from_error(
            error,
            query=query,
            limit=limit,
            language=language,
            min_score=min_score,
        )

    results = [_code_result_payload(result) for result in filtered]
    return _success(
        query=query,
        limit=limit,
        language=language,
        min_score=min_score,
        result_total=len(results),
        results=results,
        warnings=[],
    )


@mcp.tool()
def search_docs(query: str, limit: int = 10) -> dict:
    """Semantic docs search for README, plans, specs, and other text files."""
    error, results = _search_docs_results(query, limit=limit)
    if error:
        return _failure_from_error(error, query=query, limit=limit)

    payloads = [_doc_result_payload(result) for result in results]
    return _success(
        query=query,
        limit=limit,
        result_total=len(payloads),
        results=payloads,
        warnings=[],
    )


@mcp.tool()
def remember(content: str, tags: str = "") -> dict:
    """Store durable project memory. Do not store secrets."""
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
    memory_kind = inferred_kind if inferred_kind != "summary" else "note"

    db = _get_db()
    memory_id = db.remember_structured(
        summary=_truncate(_single_line(content), 200),
        content=content,
        embedding=embeddings[0],
        tags=tags,
        project_id=_ensure_project_id(),
        memory_kind=memory_kind,
        metadata={"capture_kind": "freeform"},
    )
    stored = db.get_memory(memory_id)
    return _success(
        backend="project-sqlite",
        memory=_memory_payload(
            _with_source_db(
                stored
                or {
                    "id": memory_id,
                    "summary": _truncate(_single_line(content), 200),
                    "content": content,
                    "memory_kind": memory_kind,
                    "metadata": {"capture_kind": "freeform"},
                },
                "project",
            ),
            current_project_id=_ensure_project_id(),
        ),
    )


@mcp.tool()
def search_memory(query: str, limit: int = 10) -> dict:
    """Semantic memory search across remembered project decisions and notes."""
    error, results = _search_memory_results(query, limit=limit)
    if error:
        return _failure_from_error(error, query=query, limit=limit)

    payloads = [
        _memory_payload(result, current_project_id=_ensure_project_id())
        for result in results
    ]
    return _success(
        query=query,
        limit=limit,
        result_total=len(payloads),
        results=payloads,
        warnings=[],
    )


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

    payload: dict = {
        "ok": True,
        "task": task,
        "project_id": _ensure_project_id(),
        "index": None,
        "stale": None,
        "memories": [],
        "code": [],
        "docs": [],
        "errors": {},
    }

    if refresh_index:
        payload["index"] = index_project(paths=".")

    stale_state = _stale_state(_get_db(), Path.cwd(), payload["project_id"])
    payload["stale"] = stale_state

    memory_error, memory_results = _search_memory_results(task, limit=memory_limit)
    if memory_error:
        payload["errors"]["memory"] = memory_error
    else:
        payload["memories"] = [
            _memory_payload(result, current_project_id=payload["project_id"])
            for result in memory_results
        ]

    code_error, code_results = _search_code_results(task, limit=code_limit)
    if code_error:
        payload["errors"]["code"] = code_error
    else:
        payload["code"] = [_code_result_payload(result) for result in code_results]

    docs_error, docs_results = _search_docs_results(task, limit=docs_limit)
    if docs_error:
        payload["errors"]["docs"] = docs_error
    else:
        payload["docs"] = [_doc_result_payload(result) for result in docs_results]

    return payload


@mcp.tool()
def remember_structured(
    summary: str,
    details: str = "",
    memory_kind: MemoryKind = "decision",
    tags: str = "",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Store a structured durable memory for later automatic retrieval."""
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

    content = summary if not details else f"{summary}\n\n{details}"
    try:
        embeddings = _get_embedder().embed_text_sync([content])
    except RuntimeError as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    db = _get_db()
    memory_id = db.remember_structured(
        summary=summary,
        content=content,
        embedding=embeddings[0],
        tags=tags,
        project_id=_ensure_project_id(),
        memory_kind=memory_kind,
        metadata={"capture_kind": "manual", **(metadata or {})},
        source_session_id=source_session_id or None,
        source_message_id=source_message_id or None,
    )
    stored = db.get_memory(memory_id)
    return _success(
        backend="project-sqlite",
        memory=_memory_payload(
            _with_source_db(
                stored or {"id": memory_id, "summary": summary, "content": content},
                "project",
            ),
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
    """Persist a distilled durable memory from a completed chat turn."""
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
    """Maintain a rolling durable summary for the current session."""
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
    """Create a new structured memory that supersedes an older one."""
    if not old_memory_id:
        return _failure("missing_old_memory_id", "old_memory_id is required")
    old_memory_int = _int_or_none(old_memory_id)
    if old_memory_int is None:
        return _failure(
            "invalid_old_memory_id",
            "old_memory_id must be an integer",
            old_memory_id=old_memory_id,
        )
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

    content = summary if not details else f"{summary}\n\n{details}"
    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except RuntimeError as e:
        return _failure("embedding_failed", f"embedding failed: {e}")

    db = _get_user_db()
    new_memory_id = db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=_ensure_project_id(),
        memory_kind=memory_kind,
        metadata={"capture_kind": "manual", **(metadata or {})},
        source_session_id=source_session_id or None,
        source_message_id=source_message_id or None,
        supersedes=old_memory_int,
    )
    stored = db.get_memory(new_memory_id)
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
            current_project_id=_ensure_project_id(),
        ),
    )


@mcp.tool()
def forget(memory_id: str) -> dict:
    """Delete a remembered item by ID."""
    db = _get_db()
    sqlite_id = _int_or_none(memory_id)
    if sqlite_id is None:
        return _failure("invalid_memory_id", "memory_id must be an integer", memory_id=memory_id)
    if sqlite_id is not None:
        content = db.forget(sqlite_id)
        if content:
            return _success(
                backend="project-sqlite",
                deleted=True,
                memory_id=sqlite_id,
                content_preview=content[:200],
            )
    user_db = _get_user_db()
    content = user_db.forget(sqlite_id)
    if content:
        return _success(
            backend="user-sqlite",
            deleted=True,
            memory_id=sqlite_id,
            content_preview=content[:200],
        )
    return _failure("memory_not_found", f"memory {memory_id} not found", memory_id=sqlite_id)


@mcp.tool()
def project_status() -> dict:
    """Summarize the current project index and memory state."""
    db = _get_db()
    metadata_state = _index_metadata(db)
    stale = _stale_state(db, Path.cwd(), _ensure_project_id())
    language_stats = db.language_stats()
    cleanup_candidates = _memory_cleanup_candidates(limit=3)
    return _success(
        project_id=_ensure_project_id(),
        status={
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
        },
    )


@mcp.tool()
def memory_cleanup_report(limit: int = 10) -> dict:
    """List low-value or stale memories that are good cleanup candidates."""
    if limit < 1:
        return _failure("invalid_limit", "limit must be at least 1", limit=limit)
    candidates = _memory_cleanup_candidates(limit=limit)
    return {
        "ok": True,
        "project_id": _ensure_project_id(),
        "candidate_total": len(candidates),
        "candidates": candidates,
    }


@mcp.tool()
def memory_quality_report(limit: int = 10) -> dict:
    """Summarize memory quality, provenance mix, and cleanup pressure."""
    if limit < 1:
        return _failure("invalid_limit", "limit must be at least 1", limit=limit)

    payloads = _all_memory_payloads()
    stale = [item for item in payloads if item.get("is_stale") is True]
    superseded = [item for item in payloads if item.get("is_superseded") is True]
    current_project = [item for item in payloads if item.get("provenance", {}).get("is_current_project") is True]
    cleanup_candidates = _memory_cleanup_candidates(limit=limit)
    duplicate_groups = _duplicate_auto_memory_groups(payloads)

    capture_kind_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    stale_reason_counts: dict[str, int] = {}
    cleanup_reason_counts: dict[str, int] = {}
    for item in payloads:
        provenance = item.get("provenance", {})
        capture_kind = str(provenance.get("capture_kind") or "unknown")
        source_type = str(provenance.get("source_type") or "unknown")
        capture_kind_counts[capture_kind] = capture_kind_counts.get(capture_kind, 0) + 1
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        for reason in item.get("stale_reasons") or []:
            stale_reason_counts[reason] = stale_reason_counts.get(reason, 0) + 1
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
        "ok": True,
        "project_id": _ensure_project_id(),
        "summary": {
            "total_memories": len(payloads),
            "current_project_memories": len(current_project),
            "stale_memories": len(stale),
            "superseded_memories": len(superseded),
            "cleanup_candidate_total": len(cleanup_candidates),
            "duplicate_auto_memory_groups": len(duplicate_groups),
        },
        "by_source_db": _count_by(payloads, "source_db"),
        "by_memory_kind": _count_by(payloads, "memory_kind"),
        "by_capture_kind": capture_kind_counts,
        "by_source_type": source_type_counts,
        "stale_reasons": stale_reason_counts,
        "cleanup_reasons": cleanup_reason_counts,
        "recommended_actions": recommended_actions,
        "duplicate_auto_memory_groups": duplicate_groups[:limit],
        "top_cleanup_candidates": cleanup_candidates[:limit],
    }


@mcp.tool()
def cleanup_duplicate_auto_memories(limit: int = 20, apply: bool = False) -> dict:
    """Report or prune duplicate auto-captured memories, keeping the newest copy in each group."""
    if limit < 1:
        return _failure("invalid_limit", "limit must be at least 1", limit=limit)

    groups = _duplicate_auto_memory_groups(_all_memory_payloads())[:limit]
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
            source_db = "user"
            for item in _all_memory_payloads():
                if int(item.get("id") or 0) == int(memory_id):
                    source_db = str(item.get("source_db") or "user")
                    break
            if _delete_memory_by_source_db(source_db, memory_id):
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
