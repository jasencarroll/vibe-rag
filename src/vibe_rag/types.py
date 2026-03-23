from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


MemoryKind = Literal["note", "decision", "constraint", "todo", "summary", "fact"]
SourceDB = Literal["project", "user"]


class ToolError(TypedDict):
    code: str
    message: str
    details: dict[str, Any]


class ToolFailure(TypedDict):
    ok: Literal[False]
    error: ToolError


class SearchProvenance(TypedDict):
    source: Literal["project-index"]
    indexed_at: str | None


class MemoryProvenance(TypedDict):
    capture_kind: str
    source_type: str
    is_current_project: bool


class CodeChunk(TypedDict):
    file_path: str
    chunk_index: int
    content: str
    language: str | None
    symbol: str | None
    start_line: int
    end_line: int


class CodeChunkRow(CodeChunk, total=False):
    indexed_at: str | None


class RankedCodeResult(CodeChunkRow, total=False):
    distance: float
    score: float
    rank_score: float
    match_sources: list[str]
    vector_distance: float


class CodeSearchResult(TypedDict):
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


class DocChunk(TypedDict):
    file_path: str
    chunk_index: int
    content: str


class DocChunkRow(DocChunk, total=False):
    indexed_at: str | None


class RankedDocResult(DocChunkRow, total=False):
    distance: float
    score: float
    rank_score: float
    match_sources: list[str]
    vector_distance: float


class DocSearchResult(TypedDict):
    file_path: str
    chunk_index: int
    content: str
    preview: str
    indexed_at: str | None
    rank_score: float
    match_sources: list[str]
    provenance: SearchProvenance


class MemoryRow(TypedDict, total=False):
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
    path: str
    kind: Literal["code", "doc"]
    reason: str


class MemoryPayload(TypedDict):
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
    cleanup_reasons: NotRequired[list[str]]
    cleanup_priority: NotRequired[int]


class ToolSuccess(TypedDict, total=False):
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
