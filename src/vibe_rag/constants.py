"""Constants for vibe-rag file processing and chunking."""

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
    "dist", "build", ".next", ".nuxt", "target", ".tox",
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
MAX_FILE_SIZE = 100_000

# Embedding dimension (codestral-embed default output)
EMBEDDING_DIM = 1536

# Doc chunking
DOC_CHUNK_SIZE = 2000
DOC_CHUNK_OVERLAP = 200
