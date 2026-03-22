from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LANGUAGE_MAP: dict[str, str] = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "rust": "rust", "go": "go", "java": "java", "c": "c", "cpp": "cpp",
}

SYMBOL_NODE_TYPES: set[str] = {
    "function_definition", "class_definition", "function_declaration",
    "class_declaration", "method_definition", "impl_item", "function_item", "struct_item",
}


def chunk_code_sliding_window(content: str, file_path: str, window: int = 60, overlap: int = 10) -> list[dict]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(lines):
        next_start = start + window - overlap
        # If the next chunk would be smaller than a full window, absorb into current chunk
        if next_start >= len(lines) or len(lines) - next_start < window:
            end = len(lines)
        else:
            end = min(start + window, len(lines))
        chunk_content = "".join(lines[start:end])
        chunks.append({
            "file_path": file_path, "chunk_index": chunk_index, "content": chunk_content,
            "language": None, "symbol": None, "start_line": start + 1, "end_line": end,
        })
        chunk_index += 1
        if end >= len(lines):
            break
        start = next_start
    return chunks


def _try_tree_sitter_chunk(content: str, file_path: str, language: str) -> list[dict] | None:
    ts_lang = LANGUAGE_MAP.get(language)
    if not ts_lang:
        return None
    try:
        import tree_sitter_languages
        parser = tree_sitter_languages.get_parser(ts_lang)
    except (ImportError, Exception):
        return None

    tree = parser.parse(content.encode())
    root = tree.root_node
    lines = content.splitlines(keepends=True)
    chunks = []
    chunk_index = 0

    def walk(node):
        nonlocal chunk_index
        if node.type in SYMBOL_NODE_TYPES:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            chunk_content = "".join(lines[start_line : end_line + 1])
            symbol = None
            for child in node.children:
                if child.type in ("identifier", "name"):
                    symbol = content[child.start_byte : child.end_byte]
                    break
            chunks.append({
                "file_path": file_path, "chunk_index": chunk_index, "content": chunk_content,
                "language": language, "symbol": symbol, "start_line": start_line + 1, "end_line": end_line + 1,
            })
            chunk_index += 1
        else:
            for child in node.children:
                walk(child)

    walk(root)
    if not chunks:
        return None
    return chunks


def chunk_code(content: str, file_path: str, language: str | None = None) -> list[dict]:
    if language:
        ts_chunks = _try_tree_sitter_chunk(content, file_path, language)
        if ts_chunks:
            return ts_chunks
    return chunk_code_sliding_window(content, file_path)
