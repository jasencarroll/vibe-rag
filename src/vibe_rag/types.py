"""Shared TypedDict definitions and type aliases for vibe-rag.

All structured data shapes used across the MCP tools, database layer, and
search pipeline are defined here so that type-checkers and IDE tooling can
validate field access consistently.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


# Allowed memory classification kinds produced by the auto-classifier in tools.py.
MemoryKind = Literal["note", "decision", "constraint", "todo", "summary", "fact"]

# Identifies which SQLite database a row originates from.
SourceDB = Literal["project", "user"]


class ToolError(TypedDict):
    """Structured error payload included in :class:`ToolFailure` responses."""

    code: str
    message: str
    details: dict[str, Any]


class ToolFailure(TypedDict):
    """Canonical failure envelope returned by MCP tool handlers."""

    ok: Literal[False]
    error: ToolError


class SearchProvenance(TypedDict):
    """Origin metadata attached to code and doc search results."""

    source: Literal["project-index"]
    indexed_at: str | None


class MemoryProvenance(TypedDict):
    """Origin metadata attached to memory search results."""

    capture_kind: str
    source_type: str
    is_current_project: bool


class CodeChunk(TypedDict):
    """A single chunk of source code produced by the code chunker.

    Fields capture the file location, positional index, raw content, detected
    language, top-level symbol name (if any), and the inclusive line range.
    """

    file_path: str
    chunk_index: int
    content: str
    language: str | None
    symbol: str | None
    start_line: int
    end_line: int


class CodeChunkRow(CodeChunk, total=False):
    """A :class:`CodeChunk` extended with an optional ``indexed_at`` timestamp
    as stored in the project SQLite database."""

    indexed_at: str | None


class RankedCodeResult(CodeChunkRow, total=False):
    """Intermediate ranking record for code search, carrying vector distance
    and composite rank scores used by the fusion ranking pipeline."""

    distance: float
    score: float
    rank_score: float
    match_sources: list[str]
    vector_distance: float


class CodeSearchResult(TypedDict):
    """Final code search result returned to MCP tool callers, enriched with
    provenance and a human-readable ``match_reason``."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str | None
    symbol: str | None
    indexed_at: str | None
    rank_score: float
    match_sources: list[str]
    provenance: SearchProvenance
    match_reason: str


class DocChunk(TypedDict):
    """A single chunk of a documentation file (Markdown, plain text, etc.).

    Produced by :func:`~vibe_rag.chunking.chunk_doc` and stored in the
    project index database.
    """

    file_path: str
    chunk_index: int
    content: str


class DocChunkRow(DocChunk, total=False):
    """A :class:`DocChunk` extended with an optional ``indexed_at`` timestamp
    as stored in the project SQLite database."""

    indexed_at: str | None


class RankedDocResult(DocChunkRow, total=False):
    """Intermediate ranking record for doc search, carrying vector distance
    and composite rank scores used by the fusion ranking pipeline."""

    distance: float
    score: float
    rank_score: float
    match_sources: list[str]
    vector_distance: float


class DocSearchResult(TypedDict):
    """Final documentation search result returned to MCP tool callers,
    enriched with a text ``preview``, provenance, and ``match_reason``."""

    file_path: str
    chunk_index: int
    content: str
    preview: str
    indexed_at: str | None
    rank_score: float
    match_sources: list[str]
    provenance: SearchProvenance
    match_reason: str


class MemoryRow(TypedDict, total=False):
    """Row shape for a memory record as retrieved from the SQLite database.

    All fields are optional (``total=False``) because different queries may
    populate different subsets.  The ``source_db`` field indicates whether
    the row came from the project or user memory database.
    """

    id: int
    content: str
    tags: str | list[str]
    project_id: str | None
    memory_kind: MemoryKind
    summary: str | None
    metadata: dict[str, Any]
    source_session_id: str | None
    source_message_id: str | None
    supersedes: int | None
    superseded_by: int | None
    created_at: str | None
    updated_at: str | None
    distance: float
    source_db: SourceDB


class CollectedFileSkip(TypedDict):
    """Record for a file that was skipped during directory traversal.

    Returned by :func:`~vibe_rag.chunking.collect_files_with_skips` to let
    callers report which files were excluded and why.
    """

    path: str
    kind: Literal["code", "doc"]
    reason: str


class MemoryPayload(TypedDict):
    """Rich memory record returned to MCP tool callers.

    Combines the persisted memory fields with computed attributes such as
    ``is_superseded``, ``is_stale``, ``stale_reasons``, and provenance.
    Optional ``NotRequired`` fields are present only in specific contexts
    (e.g., ``match_reason`` in search results, ``cleanup_*`` in health
    reports).
    """

    id: int
    source_db: SourceDB | None
    summary: str
    content: str
    score: float
    project_id: str | None
    memory_kind: MemoryKind
    tags: list[str]
    created_at: str | None
    updated_at: str | None
    source_session_id: str | None
    source_message_id: str | None
    supersedes: int | None
    superseded_by: int | None
    is_superseded: bool
    is_stale: bool
    stale_reasons: list[str]
    metadata: dict[str, Any]
    provenance: MemoryProvenance
    thread_id: NotRequired[str | None]
    thread_title: NotRequired[str | None]
    match_reason: NotRequired[str]
    cleanup_reasons: NotRequired[list[str]]
    cleanup_priority: NotRequired[int]


class ToolSuccess(TypedDict, total=False):
    """Canonical success envelope returned by MCP tool handlers.

    All fields are optional (``total=False``) because each tool populates
    only the subset relevant to its response (e.g., ``results`` for search,
    ``memory`` for remember, ``status`` for project_status).
    """

    ok: Literal[True]
    summary: str
    query: str
    task: str
    project_id: str
    result_total: int
    results: list[Any]
    warnings: list[dict[str, Any]]
    counts: dict[str, Any]
    backend: str
    memory: MemoryPayload
    deleted: bool
    memory_id: int
    status: dict[str, Any]
