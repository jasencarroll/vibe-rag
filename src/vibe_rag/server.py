from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vibe_rag.db.sqlite import SqliteVecDB
from vibe_rag.indexing.embedder import Embedder
from vibe_rag.indexing.code_chunker import chunk_code

logger = logging.getLogger(__name__)

mcp = FastMCP(name="vibe-rag")

_db_path = Path(os.environ.get("VIBE_RAG_DB", Path.cwd() / ".vibe" / "index.db"))
_api_key = os.environ.get("MISTRAL_API_KEY", "")

_db: SqliteVecDB | None = None
_embedder: Embedder | None = None


def _get_db() -> SqliteVecDB:
    global _db
    if _db is None:
        _db = SqliteVecDB(_db_path)
        _db.initialize()
    return _db


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        if not _api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        _embedder = Embedder(api_key=_api_key)
    return _embedder


# --- Extension sets ---

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".sh", ".bash", ".zsh", ".sql",
    ".toml", ".yaml", ".yml", ".json",
}
DOC_EXTENSIONS = {".md", ".txt", ".rst"}
SKIP_DIRS = {
    ".git", ".vibe", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", ".tox",
}
EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
    ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
}
MAX_FILE_SIZE = 100_000


# --- Doc chunking (inline, no separate module) ---

def _chunk_markdown(text: str, file_path: str) -> list[dict]:
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]
    chunks = []
    for section in sections:
        if len(section) <= 2000:
            chunks.append(section)
        else:
            paragraphs = re.split(r"\n\n+", section)
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > 2000 and current:
                    chunks.append(current.strip())
                    current = current[-200:] + "\n\n" + para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append(current.strip())
    return [
        {"file_path": file_path, "chunk_index": i, "content": c}
        for i, c in enumerate(chunks)
    ]


def _chunk_plain_text(text: str, file_path: str) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + 2000, len(text))
        chunks.append({"file_path": file_path, "chunk_index": idx, "content": text[start:end]})
        idx += 1
        if end >= len(text):
            break
        start += 1800  # 2000 - 200 overlap
    return chunks


def _chunk_doc(content: str, file_path: str) -> list[dict]:
    if file_path.endswith(".md"):
        return _chunk_markdown(content, file_path)
    return _chunk_plain_text(content, file_path)


# --- Collect files ---

def _collect_files(root_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    code_files: list[Path] = []
    doc_files: list[Path] = []
    for root in root_paths:
        for path in root.rglob("*"):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if not path.is_file() or path.stat().st_size > MAX_FILE_SIZE:
                continue
            if path.suffix in CODE_EXTENSIONS:
                code_files.append(path)
            elif path.suffix in DOC_EXTENSIONS:
                doc_files.append(path)
    return code_files, doc_files


# --- Tools ---

@mcp.tool()
def index_project(paths: list[str] | None = None) -> str:
    """Index project source files and docs for semantic search. Full re-index."""
    import time
    start = time.time()

    try:
        embedder = _get_embedder()
    except RuntimeError as e:
        return str(e)

    root_paths = [Path(p) for p in paths] if paths else [Path.cwd()]
    code_files, doc_files = _collect_files(root_paths)

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
            all_doc_chunks.extend(_chunk_doc(content, rel_path))

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


def run_server() -> None:
    mcp.run(transport="stdio")
