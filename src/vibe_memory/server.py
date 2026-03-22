from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vibe_memory.db.sqlite import SqliteVecDB
from vibe_memory.indexing.embedder import Embedder
from vibe_memory.indexing.code_chunker import chunk_code

logger = logging.getLogger(__name__)

mcp = FastMCP(name="vibe-memory")

# Resolve DB path and keys at import time
_db_path = Path(os.environ.get("VIBE_MEMORY_DB", Path.cwd() / ".vibe" / "index.db"))
_mistral_key = os.environ.get("MISTRAL_API_KEY", "")
_codestral_key = os.environ.get("CODESTRAL_API_KEY", _mistral_key)

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
        if not _mistral_key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        _embedder = Embedder(mistral_api_key=_mistral_key, codestral_api_key=_codestral_key)
    return _embedder


# --- Tools ---

DEFAULT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".sh", ".bash", ".zsh", ".sql", ".toml", ".yaml",
    ".yml", ".json", ".md", ".txt",
}
SKIP_DIRS = {
    ".git", ".vibe", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", ".tox",
}
EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
    ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
}


@mcp.tool()
def index_project(paths: list[str] | None = None) -> str:
    """Index project source files for semantic code search. Full re-index."""
    import time
    start = time.time()

    try:
        embedder = _get_embedder()
    except RuntimeError as e:
        return str(e)

    root_paths = [Path(p) for p in paths] if paths else [Path.cwd()]
    files: list[Path] = []
    for root in root_paths:
        for path in root.rglob("*"):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if path.is_file() and path.suffix in DEFAULT_EXTENSIONS and path.stat().st_size <= 100_000:
                files.append(path)

    if not files:
        return "No files found to index."

    all_chunks: list[dict] = []
    project_root = Path.cwd()
    for f in files:
        try:
            content = f.read_text(errors="replace")
        except Exception:
            continue
        language = EXT_TO_LANG.get(f.suffix)
        rel_path = str(f.relative_to(project_root))
        all_chunks.extend(chunk_code(content, rel_path, language))

    if not all_chunks:
        return "No code chunks generated."

    try:
        texts = [c["content"] for c in all_chunks]
        embeddings = embedder.embed_code_sync(texts)
    except Exception as e:
        return f"Embedding failed: {e}"

    db = _get_db()
    db.clear_code()
    db.upsert_chunks(all_chunks, embeddings)

    elapsed = time.time() - start
    return f"Indexed {len(files)} files, {len(all_chunks)} chunks in {elapsed:.1f}s"


@mcp.tool()
def search_code(query: str, limit: int = 10, language: str | None = None) -> str:
    """Search project code by semantic meaning."""
    db = _get_db()
    if db.code_chunk_count() == 0:
        return "No code index. Run index_project first."

    try:
        embedder = _get_embedder()
        embeddings = embedder.embed_code_sync([query])
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
def remember(content: str, tags: str = "") -> str:
    """Store a memory that can be recalled later via search_memory."""
    try:
        embedder = _get_embedder()
        embeddings = embedder.embed_text_sync([content])
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
        embedder = _get_embedder()
        embeddings = embedder.embed_text_sync([query])
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
