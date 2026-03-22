from __future__ import annotations
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP

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
MAX_FILE_SIZE = 100_000


def _detect_language(path: Path) -> str | None:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
        ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
        ".rb": "ruby", ".php": "php", ".swift": "swift",
    }
    return ext_map.get(path.suffix)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def index_project(paths: list[str] | None = None, extensions: list[str] | None = None) -> str:
        """Index project source files for semantic code search. Full re-index on each call."""
        import vibe_memory.server as srv
        from vibe_memory.indexing.code_chunker import chunk_code
        from vibe_memory.db.sqlite import SqliteVecDB

        if not srv._embedder:
            return "Embedding API unavailable."

        start = time.time()
        root_paths = [Path(p) for p in paths] if paths else [Path.cwd()]
        allowed_ext = {f".{e.lstrip('.')}" for e in extensions} if extensions else DEFAULT_EXTENSIONS

        files: list[Path] = []
        for root in root_paths:
            for path in root.rglob("*"):
                if any(skip in path.parts for skip in SKIP_DIRS):
                    continue
                if path.is_file() and path.suffix in allowed_ext and path.stat().st_size <= MAX_FILE_SIZE:
                    files.append(path)

        if not files:
            return "No files found to index."

        all_chunks: list[dict] = []
        project_root = Path.cwd()
        for f in files:
            content = f.read_text(errors="replace")
            language = _detect_language(f)
            rel_path = str(f.relative_to(project_root))
            file_chunks = chunk_code(content, rel_path, language)
            all_chunks.extend(file_chunks)

        if not all_chunks:
            return "No code chunks generated."

        try:
            texts = [c["content"] for c in all_chunks]
            embeddings = await srv._embedder.embed_code(texts)
        except Exception as e:
            return f"Embedding API unavailable: {e}"

        index_path = Path.cwd() / ".vibe" / "index.db"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if srv._sqlite is None:
            srv._sqlite = SqliteVecDB(index_path)
        srv._sqlite.initialize()
        srv._sqlite.clear()
        srv._sqlite.upsert_chunks(all_chunks, embeddings)

        elapsed = time.time() - start
        return f"Indexed {len(files)} files, {len(all_chunks)} chunks in {elapsed:.1f}s"
