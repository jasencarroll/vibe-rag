"""Syntax-aware code chunking using tree-sitter with sliding-window fallback.

This module splits source files into :class:`~vibe_rag.types.CodeChunk` dicts
suitable for embedding and vector search.  The primary entry point is
:func:`chunk_code`, which implements a two-stage strategy:

1. **Tree-sitter pass** -- if *language* is provided and a tree-sitter grammar
   is available, top-level symbols (functions, classes, structs, etc.) are
   extracted as individual chunks.
2. **Sliding-window fallback** -- when tree-sitter is unavailable or produces
   no chunks, the file is split into fixed-size overlapping windows.

Oversized tree-sitter chunks are automatically sub-split via
:func:`_subsplit_large_chunks` so no single chunk exceeds
:data:`MAX_CHUNK_LINES`.
"""

from __future__ import annotations

import ctypes
import logging
import site
import sysconfig
from functools import lru_cache
from pathlib import Path

from vibe_rag.types import CodeChunk

logger = logging.getLogger(__name__)

LANGUAGE_MAP: dict[str, str] = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "rust": "rust", "go": "go", "java": "java", "c": "c", "cpp": "cpp",
}

# Chunking constants
MAX_CHUNK_LINES = 200  # ~800 tokens — if a symbol is bigger, sub-split it
SLIDING_WINDOW_SIZE = 60  # Lines per chunk in sliding window mode
SLIDING_WINDOW_OVERLAP = 10  # Lines to overlap between chunks

SYMBOL_NODE_TYPES: set[str] = {
    "function_definition", "class_definition", "function_declaration",
    "class_declaration", "method_definition", "impl_item", "function_item", "struct_item",
}


def _is_tree_sitter_languages_path_trusted(tree_sitter_module_path: Path) -> bool:
    """Return True if *tree_sitter_module_path* resides under a known site-packages dir."""
    search_paths = {Path(p).resolve() for p in sysconfig.get_paths().values()}
    search_paths.update(Path(p).resolve() for p in site.getsitepackages())
    usersite = site.getusersitepackages()
    if usersite:
        search_paths.add(Path(usersite).resolve())

    resolved = tree_sitter_module_path.resolve()
    return any(resolved.is_relative_to(sp) for sp in search_paths if sp)


def chunk_code_sliding_window(
    content: str,
    file_path: str,
    window: int = SLIDING_WINDOW_SIZE,
    overlap: int = SLIDING_WINDOW_OVERLAP,
    language: str | None = None,
) -> list[CodeChunk]:
    """Split *content* into fixed-size overlapping line windows.

    Used as the fallback when tree-sitter chunking is unavailable or yields no
    symbols, and also used internally by :func:`_subsplit_large_chunks` to
    break up oversized tree-sitter chunks.

    Args:
        content: Full source text of the file.
        file_path: Path stored in each returned chunk (not read from disk).
        window: Number of lines per chunk (default :data:`SLIDING_WINDOW_SIZE`).
        overlap: Lines shared between consecutive chunks
            (default :data:`SLIDING_WINDOW_OVERLAP`).
        language: Optional language tag propagated to each chunk.

    Returns:
        List of :class:`CodeChunk` dicts with 1-based ``start_line`` /
        ``end_line`` and ``symbol`` set to ``None``.
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    step = max(1, window - overlap)
    start = 0
    chunk_index = 0
    while start < len(lines):
        end = min(start + window, len(lines))
        chunk_content = "".join(lines[start:end])
        chunks.append({
            "file_path": file_path, "chunk_index": chunk_index, "content": chunk_content,
            "language": language, "symbol": None, "start_line": start + 1, "end_line": end,
        })
        chunk_index += 1
        if end >= len(lines):
            break
        start += step
    return chunks


def _try_tree_sitter_chunk(content: str, file_path: str, language: str) -> list[CodeChunk] | None:
    """Attempt to chunk *content* by extracting top-level symbols via tree-sitter.

    Parses the file into an AST and walks it looking for nodes whose type is in
    :data:`SYMBOL_NODE_TYPES` (e.g. ``function_definition``,
    ``class_declaration``).  Each matching node becomes one chunk; the symbol
    name is extracted from the first ``identifier`` / ``name`` child.

    Args:
        content: Full source text.
        file_path: Path stored in each chunk.
        language: Language key (must be present in :data:`LANGUAGE_MAP`).

    Returns:
        A list of :class:`CodeChunk` dicts, or ``None`` when tree-sitter is
        unavailable, the language is unsupported, or no symbols are found.
    """
    ts_lang = LANGUAGE_MAP.get(language)
    if not ts_lang:
        return None
    try:
        parser = _tree_sitter_parser(ts_lang)
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("tree-sitter parser init failed for %s: %s", file_path, exc)
        return None

    tree = parser.parse(content.encode())
    root = tree.root_node
    lines = content.splitlines(keepends=True)
    chunks: list[CodeChunk] = []
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


@lru_cache(maxsize=1)
def _tree_sitter_languages_lib() -> ctypes.CDLL:
    import tree_sitter_languages

    module_path = Path(tree_sitter_languages.__file__).resolve()
    if not _is_tree_sitter_languages_path_trusted(module_path):
        raise ImportError("Refusing to load tree_sitter_languages.languages.so from untrusted location")
    lib_path = module_path.with_name("languages.so")
    return ctypes.CDLL(str(lib_path))


@lru_cache(maxsize=len(LANGUAGE_MAP))
def _tree_sitter_language(ts_lang: str):
    import tree_sitter_languages
    from tree_sitter import Language

    try:
        return tree_sitter_languages.get_language(ts_lang)
    except Exception:
        lib = _tree_sitter_languages_lib()
        language_fn = getattr(lib, f"tree_sitter_{ts_lang}")
        language_fn.restype = ctypes.c_void_p
        return Language(language_fn())


def _tree_sitter_parser(ts_lang: str):
    import tree_sitter_languages
    from tree_sitter import Parser

    try:
        return tree_sitter_languages.get_parser(ts_lang)
    except Exception:
        parser = Parser()
        language = _tree_sitter_language(ts_lang)
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
        return parser


def _subsplit_large_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Re-split any chunk that exceeds :data:`MAX_CHUNK_LINES` (200 lines).

    Chunks within the limit are passed through unchanged (with reindexed
    ``chunk_index``).  Oversized chunks are broken up via
    :func:`chunk_code_sliding_window`; sub-chunks inherit the parent's
    ``language`` and ``symbol`` and have their ``start_line`` / ``end_line``
    adjusted to absolute file positions.
    """
    result: list[CodeChunk] = []
    idx = 0
    for chunk in chunks:
        line_count = chunk["content"].count("\n") + 1
        if line_count <= MAX_CHUNK_LINES:
            chunk["chunk_index"] = idx
            result.append(chunk)
            idx += 1
        else:
            sub_chunks = chunk_code_sliding_window(
                chunk["content"], chunk["file_path"], window=SLIDING_WINDOW_SIZE, overlap=SLIDING_WINDOW_OVERLAP,
            )
            for sc in sub_chunks:
                sc["chunk_index"] = idx
                sc["language"] = chunk.get("language")
                sc["symbol"] = chunk.get("symbol")
                # Adjust line numbers relative to parent chunk
                offset = chunk["start_line"] - 1
                sc["start_line"] += offset
                sc["end_line"] += offset
                result.append(sc)
                idx += 1
    return result


def chunk_code(content: str, file_path: str, language: str | None = None) -> list[CodeChunk]:
    """Chunk a source file for embedding -- main entry point.

    Implements a two-stage strategy:

    1. If *language* is provided, attempt tree-sitter symbol extraction via
       :func:`_try_tree_sitter_chunk`.  Oversized symbols are sub-split by
       :func:`_subsplit_large_chunks`.
    2. If tree-sitter is unavailable or yields nothing, fall back to
       :func:`chunk_code_sliding_window`.

    Args:
        content: Full source text of the file.
        file_path: Stored in each returned chunk (not read from disk).
        language: Language identifier (e.g. ``"python"``, ``"rust"``).
            Must match a key in :data:`LANGUAGE_MAP` for tree-sitter to engage.

    Returns:
        List of :class:`~vibe_rag.types.CodeChunk` dicts with 1-based line
        numbers, ready for embedding.
    """
    if language:
        ts_chunks = _try_tree_sitter_chunk(content, file_path, language)
        if ts_chunks:
            return _subsplit_large_chunks(ts_chunks)
    return chunk_code_sliding_window(content, file_path, language=language)
