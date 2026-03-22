from __future__ import annotations

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
