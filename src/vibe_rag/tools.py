from __future__ import annotations
import time
from pathlib import Path

from vibe_rag.server import mcp, _get_db, _get_embedder
from vibe_rag.chunking import chunk_doc, collect_files
from vibe_rag.indexing.code_chunker import chunk_code
from vibe_rag.constants import EXT_TO_LANG


@mcp.tool()
def index_project(paths: list[str] | None = None) -> str:
    """Index project source files and docs for semantic search. Full re-index."""
    start = time.time()

    try:
        embedder = _get_embedder()
    except RuntimeError as e:
        return str(e)

    root_paths = [Path(p) for p in paths] if paths else [Path.cwd()]
    code_files, doc_files = collect_files(root_paths)

    if not code_files and not doc_files:
        return "No files found to index."

    db = _get_db()
    project_root = Path.cwd()
    code_chunk_count = 0
    doc_chunk_count = 0

    # Pass 1: Code
    if code_files:
        all_code_chunks: list[dict] = []
        for f in code_files:
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            language = EXT_TO_LANG.get(f.suffix)
            rel_path = str(f.relative_to(project_root))
            all_code_chunks.extend(chunk_code(content, rel_path, language))

        if all_code_chunks:
            try:
                embeddings = embedder.embed_code_sync([c["content"] for c in all_code_chunks])
            except Exception as e:
                return f"Code embedding failed: {e}"
            db.clear_code()
            db.upsert_chunks(all_code_chunks, embeddings)
            code_chunk_count = len(all_code_chunks)

    # Pass 2: Docs
    if doc_files:
        all_doc_chunks: list[dict] = []
        for f in doc_files:
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            rel_path = str(f.relative_to(project_root))
            all_doc_chunks.extend(chunk_doc(content, rel_path))

        if all_doc_chunks:
            try:
                embeddings = embedder.embed_text_sync([c["content"] for c in all_doc_chunks])
            except Exception as e:
                return f"Doc embedding failed: {e}"
            db.clear_docs()
            db.upsert_docs(all_doc_chunks, embeddings)
            doc_chunk_count = len(all_doc_chunks)

    elapsed = time.time() - start
    return f"Indexed {len(code_files)} code files ({code_chunk_count} chunks), {len(doc_files)} docs ({doc_chunk_count} chunks) in {elapsed:.1f}s"


@mcp.tool()
def search_code(query: str, limit: int = 10, language: str | None = None) -> str:
    """Search project code by semantic meaning."""
    db = _get_db()
    if db.code_chunk_count() == 0:
        return "No code index. Run index_project first."

    try:
        embeddings = _get_embedder().embed_code_sync([query])
    except Exception as e:
        return f"Embedding failed: {e}"

    results = db.search_code(embeddings[0], limit=limit, language=language)
    if not results:
        return "No matching code found."

    output = []
    for r in results:
        header = f"**{r['file_path']}:{r['start_line']}-{r['end_line']}**"
        if r.get("symbol"):
            header += f" (`{r['symbol']}`)"
        output.append(f"{header}\n```\n{r['content']}\n```")
    return "\n\n".join(output)


@mcp.tool()
def search_docs(query: str, limit: int = 10) -> str:
    """Search project documentation (markdown, text) by semantic meaning."""
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
    for r in results:
        output.append(f"**{r['file_path']}**\n{r['content'][:500]}")
    return "\n\n---\n\n".join(output)


@mcp.tool()
def remember(content: str, tags: str = "") -> str:
    """Store a memory that can be recalled later via search_memory."""
    try:
        embeddings = _get_embedder().embed_text_sync([content])
    except Exception as e:
        return f"Embedding failed: {e}"

    db = _get_db()
    memory_id = db.remember(content, embeddings[0], tags)
    return f"Remembered (id={memory_id}): {content[:200]}"


@mcp.tool()
def search_memory(query: str, limit: int = 10) -> str:
    """Search stored memories by semantic meaning."""
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
    for r in results:
        output.append(f"[id={r['id']} | {r['created_at']}] {r['content']}")
    return "\n\n".join(output)


@mcp.tool()
def forget(memory_id: int) -> str:
    """Delete a memory by ID."""
    db = _get_db()
    content = db.forget(memory_id)
    if content:
        return f"Deleted: {content[:200]}"
    return f"Memory {memory_id} not found."
