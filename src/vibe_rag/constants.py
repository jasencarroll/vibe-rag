"""Constants for vibe-rag file processing and chunking.

Centralises file-extension sets, directory/file skip-lists, size limits,
and chunk-size parameters used by the indexing and chunking pipelines.
"""

from __future__ import annotations

# File extensions recognized as code files
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".sh", ".bash", ".zsh", ".sql",
    ".toml", ".yaml", ".yml", ".json",
}

# File extensions recognized as documentation files
DOC_EXTENSIONS = {".md", ".txt", ".rst"}

# Directories to skip during file collection
SKIP_DIRS = {
    ".git", ".vibe", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", ".tox", "evals",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
    ".claude", ".codex", ".gemini",
}

# Individual files to skip during indexing
SKIP_FILES = {
    ".mcp.json",
}

# Mapping from file extensions to language identifiers for syntax-aware chunking
EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
    ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".sql": "sql",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
}

# Maximum file size (in bytes) to consider for indexing
MAX_FILE_SIZE = 250_000

# Maximum number of characters per documentation chunk.
# Markdown sections or plain-text windows are split to stay within this limit.
DOC_CHUNK_SIZE = 2000

# Number of characters of overlap between consecutive documentation chunks.
# Provides context continuity across chunk boundaries during vector search.
DOC_CHUNK_OVERLAP = 200
