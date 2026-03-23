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


def _sort_memory_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda item: (
            _memory_priority(item.get("memory_kind")),
            -_score(item),
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
        if path.endswith("changelog.md"):
            boost += 0.35
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
        if path.endswith("agents.md"):
            boost += 0.15

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
        if path == "docs/api.md":
            boost += 0.95
        if path == "docs/architecture.md":
            boost += 0.55
        if path.startswith("spec/") or path == "claude.md":
            boost -= 0.45
        if "changelog" in path or path.endswith("changelog.tsx"):
            boost -= 0.55
        if path.startswith("frontend/"):
            boost -= 0.2

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
        if {"resume", "continue"} & terms and path == "readme.md":
            bonus += 2.0
        if "api" in intents:
            if path == "docs/api.md":
                bonus += 2.4
            if path == "docs/architecture.md":
                bonus += 0.9
            if path == "claude.md" or path.startswith("spec/"):
                bonus -= 1.2
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


def _memory_payload(result: dict) -> dict:
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
        "supersedes": str(result["supersedes"]) if result.get("supersedes") is not None else None,
        "superseded_by": str(result["superseded_by"]) if result.get("superseded_by") is not None else None,
        "metadata": result.get("metadata") or {},
        "provenance": {
            "source_db": result.get("source_db"),
            "project_id": result.get("project_id"),
            "source_session_id": result.get("source_session_id"),
            "source_message_id": result.get("source_message_id"),
            "created_at": str(result.get("created_at")) if result.get("created_at") is not None else None,
            "updated_at": str(result.get("updated_at")) if result.get("updated_at") is not None else None,
        },
    }
    if isinstance(payload["tags"], str):
        payload["tags"] = [tag.strip() for tag in payload["tags"].split(",") if tag.strip()]
    return payload


def _merge_memory_results(*result_sets: list[dict], limit: int) -> list[dict]:
    seen_ids: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for results in result_sets:
        for result in results:
            dedupe_key = (str(result.get("source_db") or "unknown"), str(result["id"]))
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            merged.append(result)
    return _sort_memory_results(merged)[:limit]


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
    reranked = _rerank_results(query, _merge_ranked_results(raw_results, lexical_results, limit=max(limit * 10, 20)))
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
    results = _merge_memory_results(project_results, user_results, limit=limit)
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
        payload["memories"] = [_memory_payload(result) for result in memory_results]

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
        "memory": _memory_payload(stored or {"id": memory_id, "summary": summary, "content": content}),
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

    user_db = _get_user_db()
    existing = user_db.get_memory_by_source(
        source_session_id.strip(), source_message_id.strip()
    )
    if existing:
        return {
            "ok": True,
            "backend": "user-sqlite",
            "deduplicated": True,
            "memory": _memory_payload(existing),
        }

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
        memory_kind=memory_kind,
        metadata=enriched_metadata,
        source_session_id=source_session_id.strip(),
        source_message_id=source_message_id.strip(),
    )
    stored = user_db.get_memory(memory_id)
    return {
        "ok": True,
        "backend": "user-sqlite",
        "deduplicated": False,
        "memory": _memory_payload(stored or {"id": memory_id, "summary": summary, "content": content}),
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
            "memory": _memory_payload(existing),
        }
    memory_id = user_db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        project_id=_ensure_project_id(),
        memory_kind="summary",
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
        "memory": _memory_payload(stored or {"id": memory_id, "summary": summary, "content": content}),
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
            }
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
    return "\n".join(lines)
