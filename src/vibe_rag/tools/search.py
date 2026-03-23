from __future__ import annotations

import math

from vibe_rag.server import _ensure_project_id, mcp
from vibe_rag.tools._helpers import (
    _code_result_payload,
    _doc_result_payload,
    _failure,
    _failure_from_error,
    _memory_payload,
    _search_code_results,
    _search_docs_results,
    _search_memory_results,
    _success,
)


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    scope: str = "all",
    language: str | None = None,
    min_score: float = 0.0,
) -> dict:
    """Semantic search across code and docs. Use scope to narrow. Prefer this over grep when you know behavior but not exact symbols."""
    if scope not in ("all", "code", "docs"):
        return _failure("invalid_scope", f"scope must be 'all', 'code', or 'docs', got '{scope}'")

    warnings: list[dict] = []
    code_results: list[dict] = []
    doc_results: list[dict] = []

    if scope in ("all", "code"):
        code_limit = math.ceil(limit * 0.6) if scope == "all" else limit
        error, filtered = _search_code_results(query, limit=code_limit, language=language, min_score=min_score)
        if error:
            if scope == "code":
                return _failure_from_error(error, query=query, limit=limit, scope=scope, language=language, min_score=min_score)
            warnings.append({"scope": "code", "error": error})
        else:
            code_results = [
                {**_code_result_payload(result, query=query), "result_type": "code"}
                for result in filtered
            ]

    if scope in ("all", "docs"):
        docs_limit = limit - len(code_results) if scope == "all" else limit
        docs_limit = max(docs_limit, 1) if scope == "all" else docs_limit
        error, results = _search_docs_results(query, limit=docs_limit)
        if error:
            if scope == "docs":
                return _failure_from_error(error, query=query, limit=limit, scope=scope)
            warnings.append({"scope": "docs", "error": error})
        else:
            doc_results = [
                {**_doc_result_payload(result, query=query), "result_type": "doc"}
                for result in results
            ]

    if scope == "all":
        merged = sorted(
            code_results + doc_results,
            key=lambda r: r.get("rank_score", 0.0),
            reverse=True,
        )[:limit]
    elif scope == "code":
        merged = code_results
    else:
        merged = doc_results

    return _success(
        query=query,
        limit=limit,
        scope=scope,
        language=language,
        min_score=min_score,
        result_total=len(merged),
        results=merged,
        warnings=warnings,
    )


# Keep search_code and search_docs as thin wrappers for backward compatibility
# (not registered as MCP tools, but importable for tests and internal use).
def search_code(
    query: str,
    limit: int = 10,
    language: str | None = None,
    min_score: float = 0.0,
) -> dict:
    """Backward-compatible wrapper around the unified search tool."""
    return search(query, limit=limit, scope="code", language=language, min_score=min_score)


def search_docs(query: str, limit: int = 10) -> dict:
    """Backward-compatible wrapper around the unified search tool."""
    return search(query, limit=limit, scope="docs")


@mcp.tool()
def search_memory(query: str, limit: int = 10) -> dict:
    """Semantic memory search across remembered project decisions and notes."""
    error, results = _search_memory_results(query, limit=limit)
    if error:
        return _failure_from_error(error, query=query, limit=limit)

    payloads = [
        _memory_payload(result, current_project_id=_ensure_project_id(), query=query)
        for result in results
    ]
    return _success(
        query=query,
        limit=limit,
        result_total=len(payloads),
        results=payloads,
        warnings=[],
    )
