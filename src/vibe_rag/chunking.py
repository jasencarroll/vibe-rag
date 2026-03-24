"""Document chunking utilities for the vibe-rag indexing pipeline.

Provides functions to split documentation files into overlapping chunks
suitable for embedding and vector search.  Markdown files are split on
``## `` section headings first, then by paragraph boundaries; plain-text
files use a fixed-size sliding window.  The module also exposes helpers
for collecting code and doc files from a directory tree while respecting
skip-lists and size limits defined in :mod:`vibe_rag.constants`.

Chunk sizes and overlap are controlled by :data:`~vibe_rag.constants.DOC_CHUNK_SIZE`
(default 2000 chars) and :data:`~vibe_rag.constants.DOC_CHUNK_OVERLAP`
(default 200 chars).
"""

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
    """Split a Markdown document into overlapping chunks.

    The text is first split on level-2 headings (``## ``).  Sections that
    fit within :data:`~vibe_rag.constants.DOC_CHUNK_SIZE` are kept whole;
    larger sections are further divided on paragraph boundaries (blank
    lines) with an overlap of :data:`~vibe_rag.constants.DOC_CHUNK_OVERLAP`
    characters carried forward from the end of the previous chunk.

    Args:
        text: Raw Markdown content.
        file_path: Path recorded in each returned :class:`DocChunk`.

    Returns:
        Ordered list of :class:`DocChunk` dicts, each containing
        ``file_path``, ``chunk_index``, and ``content``.
    """
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
    """Split plain text into fixed-size overlapping chunks.

    Uses a sliding window of :data:`~vibe_rag.constants.DOC_CHUNK_SIZE`
    characters, advancing by ``DOC_CHUNK_SIZE - DOC_CHUNK_OVERLAP`` on
    each step so that consecutive chunks share ``DOC_CHUNK_OVERLAP``
    characters of context.

    Args:
        text: Raw text content.
        file_path: Path recorded in each returned :class:`DocChunk`.

    Returns:
        Ordered list of :class:`DocChunk` dicts.
    """
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
    """Chunk a document file, dispatching by file type.

    Markdown files (``*.md``) are processed with :func:`chunk_markdown`;
    all other files fall through to :func:`chunk_plain_text`.

    Args:
        content: Full file content as a string.
        file_path: File path used for type detection and recorded in chunks.

    Returns:
        Ordered list of :class:`DocChunk` dicts.
    """
    if file_path.endswith(".md"):
        return chunk_markdown(content, file_path)
    return chunk_plain_text(content, file_path)


def collect_files(root_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Collect code and doc files from *root_paths*, discarding skip info.

    Convenience wrapper around :func:`collect_files_with_skips` that drops
    the third element (the list of skipped-file records).

    Args:
        root_paths: Directory roots to walk recursively.

    Returns:
        A ``(code_files, doc_files)`` tuple of :class:`~pathlib.Path` lists.
    """
    code_files, doc_files, _ = collect_files_with_skips(root_paths)
    return code_files, doc_files


def collect_files_with_skips(root_paths: list[Path]) -> tuple[list[Path], list[Path], list[CollectedFileSkip]]:
    """Collect code and doc files in a single directory traversal.

    Walks each directory in *root_paths* recursively, classifying files by
    extension into code (:data:`~vibe_rag.constants.CODE_EXTENSIONS`) or
    documentation (:data:`~vibe_rag.constants.DOC_EXTENSIONS`).  Files are
    excluded when they live under a directory listed in
    :data:`~vibe_rag.constants.SKIP_DIRS`, match
    :data:`~vibe_rag.constants.SKIP_FILES`, exceed
    :data:`~vibe_rag.constants.MAX_FILE_SIZE`, are symlinks, or trigger
    permission / OS errors.

    Args:
        root_paths: Directory roots to walk.

    Returns:
        A three-tuple ``(code_files, doc_files, skipped)`` where *skipped*
        is a list of :class:`~vibe_rag.types.CollectedFileSkip` records
        describing files that were excluded with a reportable reason.
    """
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
