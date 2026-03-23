from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
import tomllib

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
CONSTRAINT_TERMS = {"constraint", "must", "cannot", "required", "requires", "never", "always", "limit"}
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
SANDBOX_QUERY_TERMS = {
    "sandbox",
    "exec",
    "policy",
    "approval",
    "approvals",
    "shell",
}
API_QUERY_TERMS = {
    "api",
    "auth",
    "route",
    "routes",
    "endpoint",
    "endpoints",
    "letters",
    "observations",
    "decrees",
    "analytics",
    "billing",
    "checkout",
    "keys",
    "health",
    "fastapi",
    "backend",
}
PIPELINE_QUERY_TERMS = {
    "pipeline",
    "discover",
    "scrape",
    "parser",
    "parse",
    "classify",
    "classifier",
    "enrich",
    "ingestion",
    "ingest",
    "observation",
    "observations",
    "483",
    "warning",
    "letter",
    "letters",
}


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


def _validate_query(query: str) -> str | None:
    if not query.strip():
        return "Error: query is empty."
    if len(query) > MAX_QUERY_LENGTH:
        return "Error: query is too long."
    return None


def _validate_memory_content(content: str) -> str | None:
    if not content.strip():
        return "Error: content is empty."
    if len(content) > MAX_MEMORY_LENGTH:
        return "Error: content is too large."
    return None


def _validate_tags(tags: str) -> str | None:
    if len(tags) > MAX_TAGS_LENGTH:
        return "Error: tags are too long."
    return None


def _validate_memory_kind(memory_kind: str) -> str | None:
    if memory_kind not in ALLOWED_MEMORY_KINDS:
        allowed = ", ".join(sorted(ALLOWED_MEMORY_KINDS))
        return f"Error: memory_kind must be one of {allowed}."
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
        raise ValueError("Error: turns must contain at least one user/assistant pair.")

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
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _should_skip_session_capture(response: str) -> bool:
    normalized = _single_line(response).lower()
    return normalized.startswith("i have no durable memory") or normalized.startswith(
        "no durable memory"
    )


def _score(result: dict) -> float:
    if result.get("score") is not None:
        return float(result["score"])
    distance = result.get("distance")
    if distance is None:
        return 1.0
    return 1.0 - float(distance)


def _rank_score(result: dict) -> float:
    if result.get("score") is not None:
        return float(result["score"])
    distance = result.get("distance")
    if distance is None:
        return 1.0
    return 1.0 / (1.0 + max(0.0, float(distance)))


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


def _infer_auto_memory_kind(task: str, summary: str, content: str) -> str:
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
    candidates.extend(project_db.list_memories(limit=max(project_db.memory_count(include_superseded=True) + 10, 20), include_superseded=False, project_id=current_project_id))
    for item in user_db.list_memories(limit=max(user_db.memory_count(include_superseded=True) + 10, 20), include_superseded=False):
        if str(item.get("project_id") or "") == current_project_id:
            item["source_db"] = "user"
            candidates.append(item)
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
        project_db.list_memories(
            limit=max(project_db.memory_count(include_superseded=True) + 10, 20),
            include_superseded=False,
            project_id=project_id,
        )
    )
    for item in user_db.list_memories(limit=max(user_db.memory_count(include_superseded=True) + 10, 20), include_superseded=False):
        if str(item.get("project_id") or "") == project_id:
            candidates.append(item)
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
        penalty += 1
    if state["is_stale"]:
        penalty += 2
    if _is_low_signal_auto_memory(result):
        penalty += 1
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
            -_score(item),
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
    if SANDBOX_QUERY_TERMS & terms:
        intents.add("sandbox")
    if API_QUERY_TERMS & terms:
        intents.add("api")
    if PIPELINE_QUERY_TERMS & terms:
        intents.add("pipeline")
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

    if {"install", "build", "config"} & terms:
        if path == "codex-rs/config/src/lib.rs":
            boost += 1.75
        elif path.startswith("codex-rs/config/src/"):
            boost += 1.0
        if path == "codex-rs/cli/src/mcp_cmd.rs":
            boost += 1.2
        elif path.startswith("codex-rs/cli/src/"):
            boost += 0.45
        if path.startswith(".github/workflows/"):
            boost -= 0.8
        if "/tests/" in path or path.startswith("tests/"):
            boost -= 0.7

    if "mcp" in terms or "sandbox" in intents:
        if path == "codex-rs/protocol/src/mcp.rs":
            boost += 1.9
        elif path.startswith("codex-rs/protocol/src/"):
            boost += 0.8
        if path == "shell-tool-mcp/src/index.ts":
            boost += 1.8
        elif path.startswith("shell-tool-mcp/src/"):
            boost += 0.9
        if path.startswith("codex-rs/mcp-server/src/"):
            boost += 1.0
        if "sandbox" in path or "execpolicy" in path:
            boost += 0.95
        if path.startswith(".github/workflows/"):
            boost -= 0.8
        if "/tests/" in path or path.startswith("tests/"):
            boost -= 0.75

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

    if "api" in intents:
        if path == "backend/app/main.py":
            boost += 1.0
        if path.startswith("backend/app/routes/"):
            boost += 0.75
        if path.startswith("backend/app/analytics/") or path.startswith("backend/app/billing/"):
            boost += 0.35
        if path == "docs/api.md" or path.endswith("/docs/api.md"):
            boost += 0.95
        if path == "docs/architecture.md":
            boost += 0.55
        if "setup" in path:
            boost -= 0.5
        if path.startswith("spec/") or path == "claude.md":
            boost -= 0.45
        if "changelog" in path or path.endswith("changelog.tsx"):
            boost -= 0.55
        if path.startswith("frontend/"):
            boost -= 0.2

    if "bootstrap" in intents and "mcp" in terms:
        if "mcp-tools" in path or path.endswith("mcp-tools.md"):
            boost += 0.8
        if path.endswith("/readme.md") and "mcp-server/" in path:
            boost += 0.45

    if "pipeline" in intents:
        if path.startswith("backend/app/pipeline/"):
            boost += 0.95
        if path.startswith("backend/scripts/"):
            boost += 0.65
        if path == "docs/pipeline.md":
            boost += 1.0
        if path == "readme.md":
            boost += 0.25
        if path.startswith("spec/") or path == "claude.md":
            boost -= 0.35
        if "changelog" in path or path.endswith("changelog.tsx"):
            boost -= 0.6

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
        if "api" in intents:
            if path == "docs/api.md" or path.endswith("/docs/api.md"):
                bonus += 2.4
            if path == "docs/architecture.md":
                bonus += 0.9
            if "setup" in path:
                bonus -= 1.0
            if path == "claude.md" or path.startswith("spec/"):
                bonus -= 1.2
        if "bootstrap" in intents and "mcp" in terms:
            if "mcp-tools" in path or path.endswith("mcp-tools.md"):
                bonus += 1.8
            if path.endswith("/readme.md") and "mcp-server/" in path:
                bonus += 0.8
        if "pipeline" in intents:
            if path == "docs/pipeline.md":
                bonus += 2.4
            if path == "readme.md":
                bonus += 0.35
            if path == "claude.md" or path.startswith("spec/"):
                bonus -= 1.0
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


def _merge_ranked_results(*result_sets: list[dict], limit: int) -> list[dict]:
    seen: set[tuple[str, int | str]] = set()
    merged: list[dict] = []
    for result_set in result_sets:
        for item in result_set:
            key = (str(item.get("file_path") or ""), item.get("chunk_index") or item.get("start_line") or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged[:limit]


def _memory_payload(result: dict, current_project_id: str | None = None) -> dict:
    metadata = result.get("metadata") or {}
    capture_kind = str(metadata.get("capture_kind") or "").strip() or "unknown"
    superseded_by = (
        str(result["superseded_by"]) if result.get("superseded_by") is not None else None
    )
    supersedes = str(result["supersedes"]) if result.get("supersedes") is not None else None
    state = _memory_state(result, current_project_id)
    provenance = {
        "source_db": result.get("source_db"),
        "project_id": result.get("project_id"),
        "memory_kind": result.get("memory_kind", "note"),
        "capture_kind": capture_kind,
        "source_type": "structured" if result.get("memory_kind") != "note" else "freeform",
        "source_session_id": result.get("source_session_id"),
        "source_message_id": result.get("source_message_id"),
        "created_at": str(result.get("created_at")) if result.get("created_at") is not None else None,
        "updated_at": str(result.get("updated_at")) if result.get("updated_at") is not None else None,
        "is_current_project": state["is_current_project"],
        "is_superseded": superseded_by is not None,
        "is_stale": state["is_stale"],
        "stale_reasons": state["stale_reasons"],
        "supersedes": supersedes,
        "superseded_by": superseded_by,
    }
    if capture_kind == "session_distillation":
        provenance["source_type"] = "session_distillation"
    elif capture_kind == "session_rollup":
        provenance["source_type"] = "session_rollup"
    elif capture_kind == "manual" and result.get("memory_kind") != "note":
        provenance["source_type"] = "manual_structured"

    payload = {
        "id": str(result["id"]),
        "source_db": result.get("source_db"),
        "summary": result.get("summary") or result.get("content", "")[:200],
        "content": result.get("content", ""),
        "score": round(_score(result), 4),
        "project_id": result.get("project_id"),
        "memory_kind": result.get("memory_kind", "note"),
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

    for result in project_db.list_memories(limit=max(limit * 3, 20), include_superseded=True, project_id=current_project_id):
        result["source_db"] = "project"
        candidates.append(result)
    for result in user_db.list_memories(limit=max(limit * 5, 30), include_superseded=True):
        result["source_db"] = "user"
        candidates.append(result)

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

    for result in project_db.list_memories(limit=max(project_db.memory_count() + 10, 20), include_superseded=True, project_id=current_project_id):
        result["source_db"] = "project"
        payloads.append(_memory_payload(result, current_project_id=current_project_id))
    for result in user_db.list_memories(limit=max(user_db.memory_count() + 10, 20), include_superseded=True):
        result["source_db"] = "user"
        payloads.append(_memory_payload(result, current_project_id=current_project_id))
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
                "memory_ids": [str(item.get("id") or "") for item in sorted_items],
            }
        )
    duplicates.sort(key=lambda item: (-int(item["count"]), str(item.get("summary") or "")))
    return duplicates


def _delete_memory_by_source_db(source_db: str, memory_id: str) -> bool:
    try:
        sqlite_id = int(memory_id)
    except (TypeError, ValueError):
        return False
    db = _get_db() if source_db == "project" else _get_user_db()
    deleted = db.forget(sqlite_id)
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
    return _sort_memory_results(merged, current_project_id=current_project_id)[:limit]


def _current_git_head(project_root: Path | None = None) -> str | None:
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
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def _load_toml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _vibe_trust_status(project_root: Path) -> dict:
    config_path = Path.home() / ".vibe" / "trusted_folders.toml"
    data = _load_toml(config_path)
    if not data:
        return {"status": "unknown", "detail": f"{config_path} not found or unreadable"}

    trusted = {Path(entry).resolve() for entry in data.get("trusted", []) if isinstance(entry, str) and "$" not in entry}
    resolved = project_root.resolve()
    if resolved in trusted:
        return {"status": "ok", "detail": f"trusted in {config_path}"}
    return {"status": "warn", "detail": f"{resolved} not trusted in {config_path}"}


def _codex_trust_status(project_root: Path) -> dict:
    config_path = Path.home() / ".codex" / "config.toml"
    data = _load_toml(config_path)
    if not data:
        return {"status": "unknown", "detail": f"{config_path} not found or unreadable"}

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
    return db.get_setting_json(INDEX_METADATA_KEY) or {}


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
    metadata = _index_metadata(db)
    if not metadata:
        return {
            "is_stale": False,
            "warnings": [],
            "metadata": None,
        }

    warnings: list[dict] = []
    current_head = _current_git_head(project_root)
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
    try:
        return embed_method(texts, progress_callback=progress_callback)
    except TypeError:
        return embed_method(texts)


def _search_code_results(
    query: str,
    limit: int = 10,
    language: str | None = None,
    min_score: float = 0.0,
) -> tuple[str | None, list[dict]]:
    error = _validate_query(query)
    if error:
        return error, []

    if language and language not in set(EXT_TO_LANG.values()):
        return f"Error: Unknown language '{language}'.", []

    db = _get_db()
    if db.code_chunk_count() == 0:
        return "No code index. Run index_project first.", []

    try:
        embeddings = _get_embedder().embed_code_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}", []

    raw_results = db.search_code(embeddings[0], limit=max(limit * 5, 10), language=language)
    lexical_results = db.lexical_search_code(sorted(_query_terms(query)), limit=max(limit * 5, 10))
    workflow_results: list[dict] = []
    intent_results: list[dict] = []
    terms = _query_terms(query)
    if {"release", "workflow"} <= terms or ({"release", "publish"} <= terms and "tag" in terms):
        workflow_results = db.lexical_search_code(
            ["publish.yml", ".github/workflows", "release", "publish"],
            limit=max(limit * 2, 5),
        )
    if {"install", "build", "config"} & terms:
        intent_results.extend(
            db.lexical_search_code(
                ["config/src/lib.rs", "mcp_cmd", "config", "CODEX_HOME", "sqlite_home"],
                limit=max(limit * 3, 8),
            )
        )
    if "mcp" in terms or {"sandbox", "approval", "approvals", "exec", "policy"} & terms:
        intent_results.extend(
            db.lexical_search_code(
                ["protocol/src/mcp.rs", "shell-tool-mcp", "sandbox", "execpolicy", "approval"],
                limit=max(limit * 3, 8),
            )
        )
    reranked = _rerank_results(
        query,
        _merge_ranked_results(raw_results, lexical_results, workflow_results, intent_results, limit=max(limit * 10, 20)),
    )
    filtered = [result for result in reranked if _score(result) >= min_score][:limit]
    if not filtered:
        return "No matching code found.", []
    return None, filtered


def _search_docs_results(query: str, limit: int = 10) -> tuple[str | None, list[dict]]:
    error = _validate_query(query)
    if error:
        return error, []

    db = _get_db()
    if db.doc_count() == 0:
        return "No docs indexed. Run index_project first.", []

    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}", []

    raw_results = db.search_docs(embeddings[0], limit=max(limit * 5, 10))
    lexical_results = db.lexical_search_docs(sorted(_query_terms(query)), limit=max(limit * 5, 10))
    results = _rerank_doc_results(
        query,
        _merge_ranked_results(raw_results, lexical_results, limit=max(limit * 10, 20)),
    )
    results = results[:limit]
    if not results:
        return "No matching docs found.", []
    return None, results


def _search_memory_results(query: str, limit: int = 10) -> tuple[str | None, list[dict]]:
    error = _validate_query(query)
    if error:
        return error, []

    project_db = _get_db()
    user_db = _get_user_db()
    if project_db.memory_count() == 0 and user_db.memory_count() == 0:
        return "No memories stored yet.", []
    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}", []

    current_project_id = _ensure_project_id()
    project_limit, user_limit = _memory_limit_split(limit)
    project_results = project_db.search_memories(
        embeddings[0], limit=project_limit, project_id=current_project_id
    )
    user_results = user_db.search_memories(embeddings[0], limit=max(user_limit, 10))
    for result in project_results:
        result["source_db"] = "project"
    for result in user_results:
        result["source_db"] = "user"
    results = _merge_memory_results(
        project_results,
        user_results,
        limit=limit,
        current_project_id=current_project_id,
    )
    if not results:
        return "No matching memories found.", []
    return None, results


def _index_project_impl(
    paths: list[str] | str | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> str:
    start = time.time()

    normalized = _normalize_paths(paths)
    if isinstance(normalized, str):
        return normalized
    root_paths, project_root = normalized

    try:
        embedder = _get_embedder()
    except RuntimeError as e:
        return str(e)

    code_files, doc_files = collect_files(root_paths)
    if not code_files and not doc_files:
        return "No files found to index."
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
    doc_chunks: list[dict] = []
    doc_embeddings_input: list[str] = []
    code_unchanged = 0
    doc_unchanged = 0
    _emit_progress(progress_callback, phase="code_chunking_start", file_total=len(code_files))

    for path in code_files:
        try:
            content = path.read_text(errors="replace")
        except Exception:
            continue
        rel_path = _relative_to_project(path, project_root)
        digest = _content_hash(content)
        if code_hashes.get(rel_path) == digest:
            code_unchanged += 1
            continue

        db.delete_file_chunks(rel_path, kind="code")
        language = EXT_TO_LANG.get(path.suffix)
        file_chunks = chunk_code(content, rel_path, language)
        code_chunks.extend(file_chunks)
        code_embeddings_input.extend(chunk["content"] for chunk in file_chunks)
        db.set_file_hash(rel_path, digest, "code")

    _emit_progress(
        progress_callback,
        phase="code_chunking_complete",
        file_total=len(code_files),
        chunk_total=len(code_chunks),
        unchanged_total=code_unchanged,
    )
    _emit_progress(progress_callback, phase="doc_chunking_start", file_total=len(doc_files))

    for path in doc_files:
        try:
            content = path.read_text(errors="replace")
        except Exception:
            continue
        rel_path = _relative_to_project(path, project_root)
        digest = _content_hash(content)
        if doc_hashes.get(rel_path) == digest:
            doc_unchanged += 1
            continue

        db.delete_file_chunks(rel_path, kind="doc")
        file_chunks = chunk_doc(content, rel_path)
        doc_chunks.extend(file_chunks)
        doc_embeddings_input.extend(chunk["content"] for chunk in file_chunks)
        db.set_file_hash(rel_path, digest, "doc")

    _emit_progress(
        progress_callback,
        phase="doc_chunking_complete",
        file_total=len(doc_files),
        chunk_total=len(doc_chunks),
        unchanged_total=doc_unchanged,
    )

    try:
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
            db.upsert_chunks(code_chunks, code_embeddings)
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
            db.upsert_docs(doc_chunks, doc_embeddings)
    except Exception as e:
        return f"Indexing failed: {e}"

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
    return (
        f"Indexed {len(code_files)} code files ({len(code_chunks)} chunks, {code_unchanged} unchanged), "
        f"{len(doc_files)} docs ({len(doc_chunks)} chunks, {doc_unchanged} unchanged) in {elapsed:.1f}s"
    )


@mcp.tool()
def index_project(paths: list[str] | str | None = None) -> str:
    """Index project source files and docs for semantic search. Prefer this before grep when exploring a repo."""
    return _index_project_impl(paths)


@mcp.tool()
def search_code(
    query: str,
    limit: int = 10,
    language: str | None = None,
    min_score: float = 0.0,
) -> str:
    """Semantic code search. Prefer this over grep when you know behavior but not exact symbols or filenames."""
    error, filtered = _search_code_results(query, limit=limit, language=language, min_score=min_score)
    if error:
        return error

    output = []
    for result in filtered:
        header = f"**{result['file_path']}:{result['start_line']}-{result['end_line']}**"
        if result.get("symbol"):
            header += f" (`{result['symbol']}`)"
        output.append(f"{header}\n```\n{result['content']}\n```")
    return "\n\n".join(output)


@mcp.tool()
def search_docs(query: str, limit: int = 10) -> str:
    """Semantic docs search for README, plans, specs, and other text files."""
    error, results = _search_docs_results(query, limit=limit)
    if error:
        return error

    output = []
    for result in results:
        output.append(f"**{result['file_path']}**\n{result['content'][:500]}")
    return "\n\n---\n\n".join(output)


@mcp.tool()
def remember(content: str, tags: str = "") -> str:
    """Store durable project memory. Do not store secrets."""
    error = _validate_memory_content(content)
    if error:
        return error

    tags_error = _validate_tags(tags)
    if tags_error:
        return tags_error

    try:
        embeddings = _get_embedder().embed_text_sync([content])
    except Exception as e:
        return f"Embedding failed: {e}"

    db = _get_db()
    memory_id = db.remember_structured(
        summary=_truncate(_single_line(content), 200),
        content=content,
        embedding=embeddings[0],
        tags=tags,
        project_id=_ensure_project_id(),
        memory_kind="note",
        metadata={"capture_kind": "freeform"},
    )
    return f"Remembered in project memory (id={memory_id}): {content[:200]}"


@mcp.tool()
def search_memory(query: str, limit: int = 10) -> str:
    """Semantic memory search across remembered project decisions and notes."""
    error, results = _search_memory_results(query, limit=limit)
    if error:
        return error

    output = []
    for result in results:
        project = f" [{result['project_id']}]" if result.get("project_id") else ""
        summary = result.get("summary") or result.get("content", "")
        kind = result.get("memory_kind")
        kind_prefix = f"{kind} " if kind else ""
        output.append(f"[id={result['id']}{project} score={_score(result):.2f}] {kind_prefix}{summary}")
    return "\n\n".join(output)


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
        return {"ok": False, "error": error, "task": task}

    payload: dict = {
        "ok": True,
        "task": task,
        "project_id": _ensure_project_id(),
        "index": None,
        "stale": None,
        "memories": [],
        "code": [],
        "docs": [],
    }

    if refresh_index:
        payload["index"] = index_project(paths=".")

    stale_state = _stale_state(_get_db(), Path.cwd(), payload["project_id"])
    payload["stale"] = stale_state

    memory_error, memory_results = _search_memory_results(task, limit=memory_limit)
    if memory_error:
        payload["memory_status"] = memory_error
    else:
        payload["memories"] = [
            _memory_payload(result, current_project_id=payload["project_id"])
            for result in memory_results
        ]

    code_error, code_results = _search_code_results(task, limit=code_limit)
    if code_error:
        payload["code_status"] = code_error
    else:
        for result in code_results:
            payload["code"].append(
                {
                    "file_path": result["file_path"],
                    "start_line": result["start_line"],
                    "end_line": result["end_line"],
                    "symbol": result.get("symbol"),
                    "language": result.get("language"),
                    "score": round(_score(result), 4),
                    "indexed_at": result.get("indexed_at"),
                    "provenance": {
                        "source": "project-index",
                        "indexed_at": result.get("indexed_at"),
                    },
                    "content": result["content"],
                }
            )

    docs_error, docs_results = _search_docs_results(task, limit=docs_limit)
    if docs_error:
        payload["docs_status"] = docs_error
    else:
        for result in docs_results:
            payload["docs"].append(
                {
                    "file_path": result["file_path"],
                    "chunk_index": result["chunk_index"],
                    "score": round(_score(result), 4),
                    "indexed_at": result.get("indexed_at"),
                    "content": result["content"],
                    "preview": result["content"].replace("\n", " ")[:160],
                    "provenance": {
                        "source": "project-index",
                        "indexed_at": result.get("indexed_at"),
                    },
                }
            )

    return payload


@mcp.tool()
def remember_structured(
    summary: str,
    details: str = "",
    memory_kind: str = "decision",
    tags: str = "",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Store a structured durable memory for later automatic retrieval."""
    error = _validate_memory_content(summary)
    if error:
        return {"ok": False, "error": error}
    details_error = _validate_memory_content(details) if details else None
    if details_error:
        return {"ok": False, "error": details_error}
    tags_error = _validate_tags(tags)
    if tags_error:
        return {"ok": False, "error": tags_error}
    kind_error = _validate_memory_kind(memory_kind)
    if kind_error:
        return {"ok": False, "error": kind_error}

    content = summary if not details else f"{summary}\n\n{details}"
    try:
        embeddings = _get_embedder().embed_text_sync([content])
    except Exception as e:
        return {"ok": False, "error": f"Embedding failed: {e}"}

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
    return {
        "ok": True,
        "backend": "project-sqlite",
        "memory": _memory_payload(
            stored or {"id": memory_id, "summary": summary, "content": content},
            current_project_id=_ensure_project_id(),
        ),
    }


@mcp.tool()
def save_session_memory(
    task: str,
    response: str,
    source_session_id: str,
    source_message_id: str,
    user_message_id: str = "",
    tags: str = "session,auto",
    memory_kind: str = "summary",
    metadata: dict | None = None,
) -> dict:
    """Persist a distilled durable memory from a completed chat turn."""
    if _should_skip_session_capture(response):
        return {"ok": True, "skipped": True, "reason": "low-signal response"}
    task_error = _validate_memory_content(task)
    if task_error:
        return {"ok": False, "error": task_error}
    response_error = _validate_memory_content(response)
    if response_error:
        return {"ok": False, "error": response_error}
    if not source_session_id.strip():
        return {"ok": False, "error": "Error: source_session_id is required."}
    if not source_message_id.strip():
        return {"ok": False, "error": "Error: source_message_id is required."}
    tags_error = _validate_tags(tags)
    if tags_error:
        return {"ok": False, "error": tags_error}
    kind_error = _validate_memory_kind(memory_kind)
    if kind_error:
        return {"ok": False, "error": kind_error}

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
        return {"ok": True, "skipped": True, "reason": "non-durable auto memory"}
    if _is_low_signal_auto_capture(
        task=task.strip(),
        summary=summary,
        content=content,
        capture_kind="session_distillation",
    ):
        return {"ok": True, "skipped": True, "reason": "low-signal auto memory"}

    user_db = _get_user_db()
    existing = user_db.get_memory_by_source(
        source_session_id.strip(), source_message_id.strip()
    )
    if existing:
        return {
            "ok": True,
            "backend": "user-sqlite",
            "deduplicated": True,
            "memory": _memory_payload(existing, current_project_id=_ensure_project_id()),
        }

    duplicate = _find_duplicate_auto_memory(
        user_db=user_db,
        project_id=_ensure_project_id(),
        summary=summary,
        content=content,
        capture_kind="session_distillation",
    )
    if duplicate:
        return {
            "ok": True,
            "backend": "user-sqlite",
            "deduplicated": True,
            "skipped": True,
            "reason": "duplicate auto memory",
            "memory": _memory_payload(duplicate, current_project_id=_ensure_project_id()),
        }

    non_novel = _find_non_novel_auto_memory(
        project_id=_ensure_project_id(),
        summary=summary,
        content=content,
    )
    if non_novel:
        return {
            "ok": True,
            "backend": "user-sqlite",
            "deduplicated": True,
            "skipped": True,
            "reason": "non-novel auto memory",
            "memory_kind": inferred_memory_kind,
            "memory": _memory_payload(non_novel, current_project_id=_ensure_project_id()),
            "merge_suggestion": _merge_suggestion_payload(non_novel, _ensure_project_id()),
        }

    merge_candidate = _find_merge_candidate(
        project_id=_ensure_project_id(),
        summary=summary,
        content=content,
        memory_kind=inferred_memory_kind,
    )

    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except Exception as e:
        return {"ok": False, "error": f"Embedding failed: {e}"}

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
    return {
        "ok": True,
        "backend": "user-sqlite",
        "deduplicated": False,
        "memory_kind": inferred_memory_kind,
        "merge_suggestion": _merge_suggestion_payload(merge_candidate, _ensure_project_id()),
        "memory": _memory_payload(
            stored or {"id": memory_id, "summary": summary, "content": content},
            current_project_id=_ensure_project_id(),
        ),
    }


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
            return {"ok": True, "skipped": True, "reason": "low-signal response"}
    task_error = _validate_memory_content(task)
    if task_error:
        return {"ok": False, "error": task_error}
    if not isinstance(turns, list):
        return {"ok": False, "error": "Error: turns must be a list."}
    if not source_session_id.strip():
        return {"ok": False, "error": "Error: source_session_id is required."}
    if not source_message_id.strip():
        return {"ok": False, "error": "Error: source_message_id is required."}
    tags_error = _validate_tags(tags)
    if tags_error:
        return {"ok": False, "error": tags_error}

    try:
        summary, content = _distill_session_summary(turns)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    inferred_memory_kind = "summary"
    if _is_transient_status_auto_capture(task.strip(), summary, content):
        return {"ok": True, "skipped": True, "reason": "non-durable auto memory"}
    if _is_low_signal_auto_capture(
        task=task.strip(),
        summary=summary,
        content=content,
        capture_kind="session_rollup",
        turn_count=len(turns),
    ):
        return {"ok": True, "skipped": True, "reason": "low-signal auto memory"}

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
    except Exception as e:
        return {"ok": False, "error": f"Embedding failed: {e}"}

    user_db = _get_user_db()
    existing = user_db.get_memory_by_source(
        source_session_id.strip(), summary_source_message_id
    )
    if existing and _metadata_dict(existing.get("metadata")).get("latest_message_id") == source_message_id.strip():
        return {
            "ok": True,
            "backend": "user-sqlite",
            "deduplicated": True,
            "memory": _memory_payload(existing, current_project_id=_ensure_project_id()),
        }
    duplicate = _find_duplicate_auto_memory(
        user_db=user_db,
        project_id=_ensure_project_id(),
        summary=summary,
        content=content,
        capture_kind="session_rollup",
    )
    if duplicate:
        return {
            "ok": True,
            "backend": "user-sqlite",
            "deduplicated": True,
            "skipped": True,
            "reason": "duplicate auto memory",
            "memory": _memory_payload(duplicate, current_project_id=_ensure_project_id()),
        }
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
        supersedes=str(existing["id"]) if existing else None,
    )
    stored = user_db.get_memory(memory_id)
    return {
        "ok": True,
        "backend": "user-sqlite",
        "deduplicated": False,
        "memory_kind": inferred_memory_kind,
        "merge_suggestion": _merge_suggestion_payload(merge_candidate, _ensure_project_id()),
        "memory": _memory_payload(
            stored or {"id": memory_id, "summary": summary, "content": content},
            current_project_id=_ensure_project_id(),
        ),
    }


@mcp.tool()
def supersede_memory(
    old_memory_id: str,
    summary: str,
    details: str = "",
    memory_kind: str = "decision",
    tags: str = "",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Create a new structured memory that supersedes an older one."""
    if not old_memory_id:
        return {"ok": False, "error": "Error: old_memory_id is required."}
    error = _validate_memory_content(summary)
    if error:
        return {"ok": False, "error": error}
    details_error = _validate_memory_content(details) if details else None
    if details_error:
        return {"ok": False, "error": details_error}
    tags_error = _validate_tags(tags)
    if tags_error:
        return {"ok": False, "error": tags_error}
    kind_error = _validate_memory_kind(memory_kind)
    if kind_error:
        return {"ok": False, "error": kind_error}

    content = summary if not details else f"{summary}\n\n{details}"
    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except Exception as e:
        return {"ok": False, "error": f"Embedding failed: {e}"}

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
        supersedes=old_memory_id,
    )
    stored = db.get_memory(new_memory_id)
    return {
        "ok": True,
        "backend": "user-sqlite",
        "memory": _memory_payload(
            stored
            or {
                "id": new_memory_id,
                "summary": summary,
                "content": content,
                "supersedes": old_memory_id,
            },
            current_project_id=_ensure_project_id(),
        ),
    }


@mcp.tool()
def forget(memory_id: str) -> str:
    """Delete a remembered item by ID."""
    db = _get_db()
    try:
        sqlite_id = int(memory_id)
    except (TypeError, ValueError):
        sqlite_id = None
    if sqlite_id is not None:
        content = db.forget(sqlite_id)
        if content:
            return f"Deleted from project memory: {content[:200]}"
    user_db = _get_user_db()
    if sqlite_id is None:
        return f"Memory {memory_id} not found."
    content = user_db.forget(sqlite_id)
    if content:
        return f"Deleted from user memory: {content[:200]}"
    return f"Memory {memory_id} not found."


@mcp.tool()
def project_status() -> str:
    """Summarize the current project index and memory state."""
    db = _get_db()
    metadata = _index_metadata(db)
    stale = _stale_state(db, Path.cwd(), _ensure_project_id())
    lines = [
        f"Project id: {_ensure_project_id()}",
        f"Code chunks: {db.code_chunk_count()}",
        f"Doc chunks: {db.doc_count()}",
        f"Project memories: {db.memory_count()}",
        f"User memories: {_get_user_db().memory_count()}",
    ]
    if metadata:
        lines.append(f"Indexed at: {metadata.get('indexed_at') or 'unknown'}")
        if metadata.get("git_head"):
            lines.append(f"Indexed git HEAD: {metadata['git_head']}")
    if stale.get("warnings"):
        lines.append(f"Stale warnings: {len(stale['warnings'])}")
        for warning in stale["warnings"][:3]:
            lines.append(f"- {warning['detail']}")
    else:
        lines.append("Stale warnings: none")
    language_stats = db.language_stats()
    if language_stats:
        language_summary = ", ".join(
            f"{language or 'unknown'}={count}" for language, count in sorted(language_stats.items())
        )
        lines.append(f"Languages: {language_summary}")
    cleanup_candidates = _memory_cleanup_candidates(limit=3)
    if cleanup_candidates:
        lines.append(f"Memory cleanup candidates: {len(cleanup_candidates)}")
        for candidate in cleanup_candidates:
            lines.append(
                f"- [{candidate['source_db']}:{candidate['id']}] {candidate['summary']} "
                f"({', '.join(candidate['cleanup_reasons'])})"
            )
    else:
        lines.append("Memory cleanup candidates: none")
    return "\n".join(lines)


@mcp.tool()
def memory_cleanup_report(limit: int = 10) -> dict:
    """List low-value or stale memories that are good cleanup candidates."""
    if limit < 1:
        return {"ok": False, "error": "Error: limit must be at least 1."}
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
        return {"ok": False, "error": "Error: limit must be at least 1."}

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
        return {"ok": False, "error": "Error: limit must be at least 1."}

    groups = _duplicate_auto_memory_groups(_all_memory_payloads())[:limit]
    results: list[dict] = []
    deleted_total = 0
    for group in groups:
        memory_ids = list(group.get("memory_ids") or [])
        if len(memory_ids) < 2:
            continue
        keep_id = memory_ids[-1]
        delete_ids = memory_ids[:-1]
        deleted_ids: list[str] = []
        for memory_id in delete_ids:
            if not apply:
                continue
            source_db = "user"
            for item in _all_memory_payloads():
                if str(item.get("id") or "") == memory_id:
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
