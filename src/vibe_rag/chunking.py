from __future__ import annotations
import re
from pathlib import Path
from vibe_rag.constants import CODE_EXTENSIONS, DOC_EXTENSIONS, SKIP_DIRS, MAX_FILE_SIZE


def chunk_markdown(text: str, file_path: str) -> list[dict]:
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


def chunk_plain_text(text: str, file_path: str) -> list[dict]:
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


def chunk_doc(content: str, file_path: str) -> list[dict]:
    if file_path.endswith(".md"):
        return chunk_markdown(content, file_path)
    return chunk_plain_text(content, file_path)


def collect_files(root_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Collect code and documentation files from root paths, excluding skipped directories.
    
    This function uses more efficient glob patterns instead of rglob('*') for better performance
    on large directories. It also checks file size and type before including files.
    """
    code_files: list[Path] = []
    doc_files: list[Path] = []
    
    # Combine all code extensions into glob patterns
    code_patterns = [f"**/*{ext}" for ext in CODE_EXTENSIONS]
    doc_patterns = [f"**/*{ext}" for ext in DOC_EXTENSIONS]
    
    for root in root_paths:
        # Collect code files
        for pattern in code_patterns:
            for path in root.glob(pattern):
                if _should_include_file(path):
                    code_files.append(path)
        
        # Collect doc files
        for pattern in doc_patterns:
            for path in root.glob(pattern):
                if _should_include_file(path):
                    doc_files.append(path)
    
    return code_files, doc_files


def _should_include_file(path: Path) -> bool:
    """Check if a file should be included in indexing."""
    # Skip directories and non-files
    if not path.is_file():
        return False
    
    # Skip files that are too large
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
    except (OSError, PermissionError):
        return False
    
    # Skip files in excluded directories
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False
    
    return True
