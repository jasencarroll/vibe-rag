from __future__ import annotations
import re
import stat as statlib
from pathlib import Path
from typing import cast
from vibe_rag.constants import (
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    SKIP_DIRS,
    SKIP_FILES,
    MAX_FILE_SIZE,
    DOC_CHUNK_SIZE,
    DOC_CHUNK_OVERLAP,
)
from vibe_rag.types import CollectedFileSkip, DocChunk


def chunk_markdown(text: str, file_path: str) -> list[DocChunk]:
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
    results: list[DocChunk] = []
    for i, chunk in enumerate(chunks):
        results.append(cast(DocChunk, {"file_path": file_path, "chunk_index": i, "content": chunk}))
    return results


def chunk_plain_text(text: str, file_path: str) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + DOC_CHUNK_SIZE, len(text))
        chunks.append(cast(DocChunk, {"file_path": file_path, "chunk_index": idx, "content": text[start:end]}))
        idx += 1
        if end >= len(text):
            break
        start += DOC_CHUNK_SIZE - DOC_CHUNK_OVERLAP
    return chunks


def chunk_doc(content: str, file_path: str) -> list[DocChunk]:
    if file_path.endswith(".md"):
        return chunk_markdown(content, file_path)
    return chunk_plain_text(content, file_path)


def collect_files(root_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    code_files, doc_files, _ = collect_files_with_skips(root_paths)
    return code_files, doc_files


def collect_files_with_skips(root_paths: list[Path]) -> tuple[list[Path], list[Path], list[CollectedFileSkip]]:
    """Collect code and doc files in a single directory traversal."""
    code_files: list[Path] = []
    doc_files: list[Path] = []
    skipped: list[CollectedFileSkip] = []
    all_extensions = CODE_EXTENSIONS | DOC_EXTENSIONS

    for root in root_paths:
        for path in root.rglob("*"):
            if path.suffix not in all_extensions:
                continue
            include, reason = _should_include_file_state(path)
            if not include:
                if reason:
                    skipped.append(
                        {
                            "path": str(path),
                            "kind": "code" if path.suffix in CODE_EXTENSIONS else "doc",
                            "reason": reason,
                        }
                    )
                continue
            if path.suffix in CODE_EXTENSIONS:
                code_files.append(path)
            else:
                doc_files.append(path)

    return code_files, doc_files, skipped


def _should_include_file(path: Path) -> bool:
    """Check if a file should be included in indexing."""
    include, _ = _should_include_file_state(path)
    return include


def _should_include_file_state(path: Path) -> tuple[bool, str | None]:
    """Check if a file should be included and return a reportable skip reason when relevant."""
    try:
        if path.is_symlink():
            return False, None
    except PermissionError as exc:
        return False, f"permission denied during symlink check: {exc}"
    except OSError as exc:
        return False, f"symlink check failed: {exc}"
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False, None
    if path.name in SKIP_FILES:
        return False, None
    try:
        stat_result = path.stat()
        if not statlib.S_ISREG(stat_result.st_mode):
            return False, None
        if stat_result.st_size > MAX_FILE_SIZE:
            return False, f"exceeds max file size ({MAX_FILE_SIZE} bytes)"
    except PermissionError as exc:
        return False, f"permission denied during stat: {exc}"
    except OSError as exc:
        return False, f"stat failed: {exc}"
    return True, None
