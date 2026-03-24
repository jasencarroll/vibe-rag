"""Search MCP tools for the vibe-rag server.

Provides two registered ``@mcp.tool()`` endpoints:

* **search** -- unified semantic search across the project code index and
  documentation index, with configurable scope.
* **search_memory** -- semantic search over stored memories (project-scoped
  and user-scoped), with optional tag / thread / date filters.

Two **deprecated** thin wrappers (``search_code``, ``search_docs``) are kept
for backward compatibility but are *not* registered as MCP tools.
"""

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
    """Semantic search across code and documentation.

    Use ``scope='code'`` for implementation details, ``scope='docs'`` for
    guides and specs, or ``scope='all'`` (default) for both.  Prefer over
    grep when you know the *behavior* but not exact symbols or filenames.

    Parameters
    ----------
    query : str
        Natural-language search query (max 2 000 chars).
    limit : int, default 10
        Maximum number of results to return.
    scope : ``"all"`` | ``"code"`` | ``"docs"``, default ``"all"``
        Which index to search.  When ``scope="all"`` the budget is split
        **60 % code / 40 % docs** (``math.ceil(limit * 0.6)`` code slots),
        then remaining slots go to docs so the total never exceeds *limit*.
    language : str or None, default None
        Filter code results by language (e.g. ``"python"``).  Ignored when
        ``scope="docs"``.
    min_score : float, default 0.0
        Drop code results whose vector similarity is below this threshold.
        Set to 0 to return all results regardless of score.

    Returns
    -------
    dict
        ``{"ok": True, ...}`` on success with keys:

        * **query**, **limit**, **scope**, **language**, **min_score** --
          echo of the request parameters.
        * **result_total** -- number of results returned.
        * **results** -- list of result dicts.  Each code result contains
          ``file_path``, ``start_line``, ``end_line``, ``content``,
          ``language``, ``symbol``, ``rank_score``, ``match_sources``,
          ``match_reason``, ``provenance``, and ``result_type="code"``.
          Each doc result contains ``file_path``, ``chunk_index``,
          ``content``, ``preview``, ``rank_score``, ``match_sources``,
          ``match_reason``, ``provenance``, and ``result_type="doc"``.
        * **warnings** -- list of per-scope warning dicts when one scope
          fails but the other succeeds (only possible when ``scope="all"``).

        ``{"ok": False, "error": {...}}`` on failure (invalid scope, empty
        index, embedding error, etc.).

    Notes
    -----
    Under the hood, each scope performs a **hybrid search**: vector
    similarity *and* lexical keyword search are run independently, then
    merged via **Reciprocal Rank Fusion (RRF)** (k=60) and **reranked**
    with path-intent boosts and term-overlap scoring.  When
    ``scope="all"``, code and doc results are combined and sorted by
    ``rank_score`` descending, then trimmed to *limit*.
    """
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
    """Search code index only.

    .. deprecated::
        Use ``search(query, scope='code')`` instead.

    Thin backward-compatible wrapper kept for existing tests and internal
    callers.  Not registered as an MCP tool.  Delegates directly to
    :func:`search` with ``scope="code"``, forwarding all parameters.
    """
    return search(query, limit=limit, scope="code", language=language, min_score=min_score)


def search_docs(query: str, limit: int = 10) -> dict:
    """Search documentation index only.

    .. deprecated::
        Use ``search(query, scope='docs')`` instead.

    Thin backward-compatible wrapper kept for existing tests and internal
    callers.  Not registered as an MCP tool.  Delegates directly to
    :func:`search` with ``scope="docs"``, forwarding all parameters.
    """
    return search(query, limit=limit, scope="docs")


@mcp.tool()
def search_memory(
    query: str,
    limit: int = 10,
    tags: str = "",
    thread_id: str = "",
    since: str = "",
    until: str = "",
) -> dict:
    """Search stored memories by semantic similarity.

    Queries both the project-scoped memory DB and the user-scoped memory DB,
    merges the results, and returns them ranked by a composite score that
    factors in vector similarity, recency, and structured-memory priority.

    Parameters
    ----------
    query : str
        Natural-language search query (max 2 000 chars).
    limit : int, default 10
        Maximum number of results to return.
    tags : str, default ``""``
        Comma-separated tag filter.  Only memories whose tags overlap with
        the requested set are returned.  Pass ``""`` to skip filtering.
    thread_id : str, default ``""``
        Restrict results to a specific conversation thread.
    since : str, default ``""``
        ISO 8601 datetime lower bound (inclusive).  Only memories created
        at or after this timestamp are returned.
    until : str, default ``""``
        ISO 8601 datetime upper bound (inclusive).  Only memories created
        at or before this timestamp are returned.

    Returns
    -------
    dict
        ``{"ok": True, ...}`` on success with keys:

        * **query**, **limit**, **tags**, **thread_id**, **since**,
          **until** -- echo of the request parameters.
        * **result_total** -- number of results returned.
        * **results** -- list of memory payload dicts, each containing
          ``id``, ``source_db``, ``summary``, ``content``, ``score``,
          ``project_id``, ``memory_kind``, ``tags``, ``created_at``,
          ``updated_at``, ``supersedes``, ``superseded_by``,
          ``match_reason``, ``stale_reasons``, and ``provenance``.
        * **warnings** -- always ``[]`` (reserved for future use).

        ``{"ok": False, "error": {...}}`` on failure (invalid query,
        invalid tags, no memories stored, embedding error, etc.).

    Notes
    -----
    Results include ``match_reason`` explaining why each memory matched
    and ``stale_reasons`` indicating potential staleness (e.g. superseded,
    old age).  When tag / thread / date filters are active the internal
    fetch limit is multiplied by 5 to ensure enough candidates survive
    post-filtering.
    """
    error, results = _search_memory_results(
        query,
        limit=limit,
        tags=tags,
        thread_id=thread_id,
        since=since,
        until=until,
    )
    if error:
        return _failure_from_error(
            error,
            query=query,
            limit=limit,
            tags=tags,
            thread_id=thread_id,
            since=since,
            until=until,
        )

    payloads = [
        _memory_payload(result, current_project_id=_ensure_project_id(), query=query)
        for result in results
    ]
    return _success(
        query=query,
        limit=limit,
        tags=tags,
        thread_id=thread_id,
        since=since,
        until=until,
        result_total=len(payloads),
        results=payloads,
        warnings=[],
    )
