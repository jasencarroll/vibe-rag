from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from vibe_rag.chunking import chunk_doc, collect_files
from vibe_rag.constants import EXT_TO_LANG
from vibe_rag.indexing.code_chunker import chunk_code
from vibe_rag.server import (
    _ensure_project_id,
    _get_db,
    _get_embedder,
    _get_pg,
    _run_async,
    mcp,
)

MAX_QUERY_LENGTH = 10_000
MAX_MEMORY_LENGTH = 10_000
MAX_TAGS_LENGTH = 512
ALLOWED_MEMORY_KINDS = {"note", "decision", "constraint", "todo", "summary", "fact"}


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


def _score(result: dict) -> float:
    if result.get("score") is not None:
        return float(result["score"])
    distance = result.get("distance")
    if distance is None:
        return 1.0
    return 1.0 - float(distance)


def _memory_payload(result: dict) -> dict:
    payload = {
        "id": str(result["id"]),
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
    }
    if isinstance(payload["tags"], str):
        payload["tags"] = [tag.strip() for tag in payload["tags"].split(",") if tag.strip()]
    return payload


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

    results = db.search_code(embeddings[0], limit=limit, language=language)
    filtered = [result for result in results if _score(result) >= min_score]
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

    results = db.search_docs(embeddings[0], limit=limit)
    if not results:
        return "No matching docs found.", []
    return None, results


def _search_memory_results(query: str, limit: int = 10) -> tuple[str | None, list[dict]]:
    error = _validate_query(query)
    if error:
        return error, []

    pg = _get_pg()
    if pg:
        try:
            count = _run_async(pg.memory_count())
        except Exception as e:
            return f"Error checking pgvector memories: {e}", []
        if count == 0:
            return "No memories stored yet.", []
        try:
            embeddings = _get_embedder().embed_text_sync([query])
        except Exception as e:
            return f"Embedding failed: {e}", []
        try:
            local_results = _run_async(
                pg.search_memories(embeddings[0], limit=limit, project_id=_ensure_project_id())
            )
        except Exception as e:
            return f"Error searching pgvector memories: {e}", []
        try:
            global_results = _run_async(pg.search_memories(embeddings[0], limit=limit, project_id=None))
        except Exception as e:
            return f"Error searching pgvector memories: {e}", []

        seen_ids: set[str] = set()
        results: list[dict] = []
        for result in local_results + global_results:
            memory_id = str(result["id"])
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            results.append(result)
            if len(results) >= limit:
                break

        if not results:
            return "No matching memories found.", []
        return None, results

    db = _get_db()
    if db.memory_count() == 0:
        return "No memories stored yet.", []
    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}", []
    results = db.search_memories(embeddings[0], limit=limit)
    if not results:
        return "No matching memories found.", []
    return None, results


@mcp.tool()
def index_project(paths: list[str] | str | None = None) -> str:
    """Index project source files and docs for semantic search. Prefer this before grep when exploring a repo."""
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

    try:
        if code_chunks:
            code_embeddings = embedder.embed_code_sync(code_embeddings_input)
            db.upsert_chunks(code_chunks, code_embeddings)
        if doc_chunks:
            doc_embeddings = embedder.embed_text_sync(doc_embeddings_input)
            db.upsert_docs(doc_chunks, doc_embeddings)
    except Exception as e:
        return f"Indexing failed: {e}"

    elapsed = time.time() - start
    return (
        f"Indexed {len(code_files)} code files ({len(code_chunks)} chunks, {code_unchanged} unchanged), "
        f"{len(doc_files)} docs ({len(doc_chunks)} chunks, {doc_unchanged} unchanged) in {elapsed:.1f}s"
    )


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

    pg = _get_pg()
    if pg:
        try:
            memory_id = _run_async(pg.remember(content, embeddings[0], tags, _ensure_project_id()))
        except Exception as e:
            return f"Error storing in pgvector: {e}"
        return f"Remembered in pgvector (id={memory_id}): {content[:200]}"

    db = _get_db()
    memory_id = db.remember(content, embeddings[0], tags)
    return f"Remembered (id={memory_id}): {content[:200]}"


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
        "memories": [],
        "code": [],
        "docs": [],
    }

    if refresh_index:
        payload["index"] = index_project(paths=".")

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
                    "content": result["content"],
                    "preview": result["content"].replace("\n", " ")[:160],
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

    pg = _get_pg()
    if pg:
        try:
            memory_id = _run_async(
                pg.remember_structured(
                    summary=summary,
                    content=content,
                    embedding=embeddings[0],
                    tags=tags,
                    project_id=_ensure_project_id(),
                    memory_kind=memory_kind,
                    metadata=metadata,
                    source_session_id=source_session_id or None,
                    source_message_id=source_message_id or None,
                )
            )
        except Exception as e:
            return {"ok": False, "error": f"Error storing in pgvector: {e}"}
        return {
            "ok": True,
            "backend": "pgvector",
            "memory": {
                "id": str(memory_id),
                "summary": summary,
                "content": content,
                "project_id": _ensure_project_id(),
                "memory_kind": memory_kind,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                "metadata": metadata or {},
                "source_session_id": source_session_id or None,
                "source_message_id": source_message_id or None,
            },
        }

    db = _get_db()
    memory_id = db.remember_structured(
        summary=summary,
        content=content,
        embedding=embeddings[0],
        tags=tags,
        memory_kind=memory_kind,
        metadata=metadata,
        source_session_id=source_session_id or None,
        source_message_id=source_message_id or None,
    )
    return {
        "ok": True,
        "backend": "sqlite",
        "memory": {
            "id": str(memory_id),
            "summary": summary,
            "content": content,
            "project_id": _ensure_project_id(),
            "memory_kind": memory_kind,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "metadata": metadata or {},
            "source_session_id": source_session_id or None,
            "source_message_id": source_message_id or None,
        },
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

    pg = _get_pg()
    if pg:
        try:
            existing = _run_async(
                pg.get_memory_by_source(
                    source_session_id=source_session_id.strip(),
                    source_message_id=source_message_id.strip(),
                )
            )
        except Exception as e:
            return {"ok": False, "error": f"Error checking pgvector memory: {e}"}
        if existing:
            return {
                "ok": True,
                "backend": "pgvector",
                "deduplicated": True,
                "memory": _memory_payload(existing),
            }

    else:
        existing = _get_db().get_memory_by_source(
            source_session_id.strip(), source_message_id.strip()
        )
        if existing:
            return {
                "ok": True,
                "backend": "sqlite",
                "deduplicated": True,
                "memory": _memory_payload(existing),
            }

    try:
        embedding = _get_embedder().embed_text_sync([content])[0]
    except Exception as e:
        return {"ok": False, "error": f"Embedding failed: {e}"}

    if pg:
        try:
            memory_id = _run_async(
                pg.remember_structured(
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
            )
            stored = _run_async(pg.get_memory(memory_id))
        except Exception as e:
            return {"ok": False, "error": f"Error storing session memory in pgvector: {e}"}
        return {
            "ok": True,
            "backend": "pgvector",
            "deduplicated": False,
            "memory": _memory_payload(stored or {"id": memory_id, "summary": summary, "content": content}),
        }

    db = _get_db()
    memory_id = db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        memory_kind=memory_kind,
        metadata=enriched_metadata,
        source_session_id=source_session_id.strip(),
        source_message_id=source_message_id.strip(),
    )
    stored = db.get_memory(memory_id)
    return {
        "ok": True,
        "backend": "sqlite",
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

    pg = _get_pg()
    if pg:
        try:
            existing = _run_async(
                pg.get_memory_by_source(
                    source_session_id=source_session_id.strip(),
                    source_message_id=summary_source_message_id,
                )
            )
        except Exception as e:
            return {"ok": False, "error": f"Error checking pgvector summary memory: {e}"}
        if existing and _metadata_dict(existing.get("metadata")).get("latest_message_id") == source_message_id.strip():
            return {
                "ok": True,
                "backend": "pgvector",
                "deduplicated": True,
                "memory": _memory_payload(existing),
            }
        try:
            memory_id = _run_async(
                pg.remember_structured(
                    summary=summary,
                    content=content,
                    embedding=embedding,
                    tags=tags,
                    project_id=_ensure_project_id(),
                    memory_kind="summary",
                    metadata=enriched_metadata,
                    source_session_id=source_session_id.strip(),
                    source_message_id=summary_source_message_id,
                    supersedes=existing["id"] if existing else None,
                )
            )
            stored = _run_async(pg.get_memory(memory_id))
        except Exception as e:
            return {"ok": False, "error": f"Error storing session summary in pgvector: {e}"}
        return {
            "ok": True,
            "backend": "pgvector",
            "deduplicated": False,
            "memory": _memory_payload(stored or {"id": memory_id, "summary": summary, "content": content}),
        }

    db = _get_db()
    existing = db.get_memory_by_source(
        source_session_id.strip(), summary_source_message_id
    )
    if existing and _metadata_dict(existing.get("metadata")).get("latest_message_id") == source_message_id.strip():
        return {
            "ok": True,
            "backend": "sqlite",
            "deduplicated": True,
            "memory": _memory_payload(existing),
        }
    memory_id = db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        memory_kind="summary",
        metadata=enriched_metadata,
        source_session_id=source_session_id.strip(),
        source_message_id=summary_source_message_id,
        supersedes=str(existing["id"]) if existing else None,
    )
    stored = db.get_memory(memory_id)
    return {
        "ok": True,
        "backend": "sqlite",
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

    pg = _get_pg()
    if pg:
        try:
            new_memory_id = _run_async(
                pg.remember_structured(
                    summary=summary,
                    content=content,
                    embedding=embedding,
                    tags=tags,
                    project_id=_ensure_project_id(),
                    memory_kind=memory_kind,
                    metadata=metadata,
                    source_session_id=source_session_id or None,
                    source_message_id=source_message_id or None,
                    supersedes=old_memory_id,
                )
            )
        except Exception as e:
            return {"ok": False, "error": f"Error superseding in pgvector: {e}"}
        return {
            "ok": True,
            "backend": "pgvector",
            "memory": {
                "id": str(new_memory_id),
                "summary": summary,
                "content": content,
                "project_id": _ensure_project_id(),
                "memory_kind": memory_kind,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                "metadata": metadata or {},
                "source_session_id": source_session_id or None,
                "source_message_id": source_message_id or None,
                "supersedes": old_memory_id,
            },
        }

    db = _get_db()
    new_memory_id = db.remember_structured(
        summary=summary,
        content=content,
        embedding=embedding,
        tags=tags,
        memory_kind=memory_kind,
        metadata=metadata,
        source_session_id=source_session_id or None,
        source_message_id=source_message_id or None,
        supersedes=old_memory_id,
    )
    return {
        "ok": True,
        "backend": "sqlite",
        "memory": {
            "id": str(new_memory_id),
            "summary": summary,
            "content": content,
            "project_id": _ensure_project_id(),
            "memory_kind": memory_kind,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "metadata": metadata or {},
            "source_session_id": source_session_id or None,
            "source_message_id": source_message_id or None,
            "supersedes": old_memory_id,
        },
    }


@mcp.tool()
def forget(memory_id: str) -> str:
    """Delete a remembered item by ID."""
    pg = _get_pg()
    if pg:
        try:
            content = _run_async(pg.forget(memory_id))
        except Exception as e:
            return f"Error deleting from pgvector: {e}"
        if content:
            return f"Deleted from pgvector: {content[:200]}"
        return f"Memory {memory_id} not found."

    db = _get_db()
    try:
        sqlite_id = int(memory_id)
    except (TypeError, ValueError):
        return f"Memory {memory_id} not found."
    content = db.forget(sqlite_id)
    if content:
        return f"Deleted: {content[:200]}"
    return f"Memory {memory_id} not found."


@mcp.tool()
def project_status() -> str:
    """Summarize the current project index and memory state."""
    db = _get_db()
    lines = [
        f"Code chunks: {db.code_chunk_count()}",
        f"Doc chunks: {db.doc_count()}",
        f"Local sqlite memories: {db.memory_count()}",
    ]
    pg = _get_pg()
    if pg:
        try:
            lines.append(f"pgvector memories: {_run_async(pg.memory_count())}")
        except Exception as e:
            lines.append(f"pgvector memories: error ({e})")
    else:
        lines.append("pgvector memories: not configured")
    language_stats = db.language_stats()
    if language_stats:
        language_summary = ", ".join(
            f"{language or 'unknown'}={count}" for language, count in sorted(language_stats.items())
        )
        lines.append(f"Languages: {language_summary}")
    return "\n".join(lines)
