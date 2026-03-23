from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


MemoryKind = Literal["note", "decision", "constraint", "todo", "summary", "fact"]


class ToolError(TypedDict):
    code: str
    message: str
    details: dict[str, Any]


class ToolFailure(TypedDict):
    ok: Literal[False]
    error: ToolError


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
    provenance: dict[str, Any]


class DocSearchResult(TypedDict):
    file_path: str
    chunk_index: int
    content: str
    preview: str
    indexed_at: str | None
    rank_score: float
    match_sources: list[str]
    provenance: dict[str, Any]


class MemoryPayload(TypedDict):
    id: int
    source_db: str | None
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
    provenance: dict[str, Any]
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
