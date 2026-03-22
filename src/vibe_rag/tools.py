from __future__ import annotations

import hashlib
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


def _score(result: dict) -> float:
    if result.get("score") is not None:
        return float(result["score"])
    distance = result.get("distance")
    if distance is None:
        return 1.0
    return 1.0 - float(distance)


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
    error = _validate_query(query)
    if error:
        return error

    if language and language not in set(EXT_TO_LANG.values()):
        return f"Error: Unknown language '{language}'."

    db = _get_db()
    if db.code_chunk_count() == 0:
        return "No code index. Run index_project first."

    try:
        embeddings = _get_embedder().embed_code_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}"

    results = db.search_code(embeddings[0], limit=limit, language=language)
    filtered = [result for result in results if _score(result) >= min_score]
    if not filtered:
        return "No matching code found."

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
    error = _validate_query(query)
    if error:
        return error

    db = _get_db()
    if db.doc_count() == 0:
        return "No docs indexed. Run index_project first."

    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}"

    results = db.search_docs(embeddings[0], limit=limit)
    if not results:
        return "No matching docs found."

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
    error = _validate_query(query)
    if error:
        return error

    pg = _get_pg()
    if pg:
        try:
            count = _run_async(pg.memory_count())
        except Exception as e:
            return f"Error checking pgvector memories: {e}"
        if count == 0:
            return "No memories stored yet."
        try:
            embeddings = _get_embedder().embed_text_sync([query])
        except Exception as e:
            return f"Embedding failed: {e}"
        try:
            results = _run_async(
                pg.search_memories(embeddings[0], limit=limit, project_id=_ensure_project_id())
            )
        except Exception as e:
            return f"Error searching pgvector memories: {e}"
        if not results:
            return "No matching memories found."
        output = []
        for result in results:
            project = f" [{result['project_id']}]" if result.get("project_id") else " [global]"
            output.append(f"[id={result['id']}{project} score={_score(result):.2f}] {result['content']}")
        return "\n\n".join(output)

    db = _get_db()
    if db.memory_count() == 0:
        return "No memories stored yet."
    try:
        embeddings = _get_embedder().embed_text_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}"
    results = db.search_memories(embeddings[0], limit=limit)
    if not results:
        return "No matching memories found."
    output = []
    for result in results:
        output.append(f"[id={result['id']} score={_score(result):.2f}] {result['content']}")
    return "\n\n".join(output)


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
        f"Memories: {db.memory_count()}",
    ]
    language_stats = db.language_stats()
    if language_stats:
        language_summary = ", ".join(
            f"{language or 'unknown'}={count}" for language, count in sorted(language_stats.items())
        )
        lines.append(f"Languages: {language_summary}")
    return "\n".join(lines)
