from __future__ import annotations
import re
from pathlib import Path
from vibe_rag.constants import (
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    SKIP_DIRS,
    SKIP_FILES,
    MAX_FILE_SIZE,
    DOC_CHUNK_SIZE,
    DOC_CHUNK_OVERLAP,
)


def chunk_markdown(text: str, file_path: str) -> list[dict]:
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]
    chunks = []
    for section in sections:
        if len(section) <= DOC_CHUNK_SIZE:
            chunks.append(section)
        else:
            paragraphs = re.split(r"\n\n+", section)
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > DOC_CHUNK_SIZE and current:
                    chunks.append(current.strip())
                    current = current[-DOC_CHUNK_OVERLAP:] + "\n\n" + para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append(current.strip())
    return [
        {"file_path": file_path, "chunk_index": i, "content": c}
        for i, c in enumerate(chunks)
    ]


def chunk_plain_text(text: str, file_path: str) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + DOC_CHUNK_SIZE, len(text))
        chunks.append({"file_path": file_path, "chunk_index": idx, "content": text[start:end]})
        idx += 1
        if end >= len(text):
            break
        start += DOC_CHUNK_SIZE - DOC_CHUNK_OVERLAP
    return chunks


def chunk_doc(content: str, file_path: str) -> list[dict]:
    if file_path.endswith(".md"):
        return chunk_markdown(content, file_path)
    return chunk_plain_text(content, file_path)


def collect_files(root_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Collect code and doc files in a single directory traversal."""
    code_files: list[Path] = []
    doc_files: list[Path] = []
    all_extensions = CODE_EXTENSIONS | DOC_EXTENSIONS

    for root in root_paths:
        for path in root.rglob("*"):
            if path.suffix not in all_extensions:
                continue
            if not _should_include_file(path):
                continue
            if path.suffix in CODE_EXTENSIONS:
                code_files.append(path)
            else:
                doc_files.append(path)

    return code_files, doc_files


def _should_include_file(path: Path) -> bool:
    """Check if a file should be included in indexing."""
    if not path.is_file():
        return False
    if path.is_symlink():
        return False
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False
    if path.name in SKIP_FILES:
        return False
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
    except (OSError, PermissionError):
        return False
    return True
