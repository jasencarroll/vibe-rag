"""tools package -- split from the former monolithic tools.py.

Importing this package registers all @mcp.tool() decorators (submodules are
imported below) and re-exports every public *and* private name so that
``from vibe_rag.tools import X`` keeps working for all existing call-sites.
"""

from __future__ import annotations

# -- submodule imports (registers @mcp.tool() decorators) --------------------
from vibe_rag.tools import index as _mod_index  # noqa: F401
from vibe_rag.tools import memory as _mod_memory  # noqa: F401
from vibe_rag.tools import search as _mod_search  # noqa: F401
from vibe_rag.tools import session as _mod_session  # noqa: F401
from vibe_rag.tools import status as _mod_status  # noqa: F401

# -- re-export MCP tool functions -------------------------------------------
from vibe_rag.tools.index import index_project, _index_project_impl  # noqa: F401
from vibe_rag.tools.search import search, search_code, search_docs, search_memory  # noqa: F401
from vibe_rag.tools.memory import (  # noqa: F401
    forget,
    ingest_daily_note,
    ingest_pr_outcome,
    remember,
    remember_structured,
    save_session_memory,
    save_session_summary,
    summarize_thread,
    supersede_memory,
    update_memory,
)
from vibe_rag.tools.session import load_session_context  # noqa: F401
from vibe_rag.tools.status import (  # noqa: F401
    cleanup_duplicate_auto_memories,
    project_status,
    _memory_health_summary,
)

# -- re-export helpers, constants, and private names -------------------------
# Tests and internal callers import these from ``vibe_rag.tools``.
from vibe_rag.tools._helpers import (  # noqa: F401
    # constants
    ALLOWED_MEMORY_KINDS,
    BOILERPLATE_TASK_PATTERNS,
    BRIEFING_CHAR_BUDGET,
    CONSTRAINT_TERMS,
    DECISION_TERMS,
    DOC_FOCUSED_QUERY_TERMS,
    DURABLE_MEMORY_TERMS,
    FACT_TERMS,
    INDEX_METADATA_KEY,
    MAX_MEMORY_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_TAGS_LENGTH,
    MAX_THREAD_ID_LENGTH,
    PROCEDURAL_QUERY_TERMS,
    STRUCTURED_MEMORY_PRIORITY,
    TODO_TERMS,
    TRANSIENT_STATUS_PATTERNS,
    _COMMON_TASK_VERBS,
    _apply_memory_filters,
    # logger
    logger,
    # private helpers
    _all_memory_payloads,
    _briefing_header,
    _briefing_task_context,
    _cleanup_candidate_reasons,
    _cleanup_candidate_score,
    _code_result_payload,
    _codex_trust_status,
    _content_hash,
    _count_by,
    _current_file_counts,
    _current_git_head,
    _current_git_head_state,
    _delete_memory_by_source_db,
    _distill_session_summary,
    _distill_session_turn,
    _doc_result_payload,
    _duplicate_auto_memory_groups,
    _embed_sync_with_progress,
    _emit_progress,
    _failure,
    _failure_from_error,
    _find_duplicate_auto_memory,
    _find_merge_candidate,
    _find_non_novel_auto_memory,
    _format_briefing,
    _git_command,
    _has_durable_auto_memory_signal,
    _hazard_scan,
    _index_metadata,
    _index_skip_summary,
    _infer_auto_memory_kind,
    _infer_session_metadata,
    _infer_session_outcome,
    _infer_session_topic,
    _int_or_none,
    _is_auto_capture_memory,
    _is_low_signal_auto_capture,
    _is_low_signal_auto_memory,
    _is_transient_status_auto_capture,
    _live_decisions,
    _load_toml,
    _load_toml_state,
    _match_reason,
    _memory_capture_kind,
    _memory_cleanup_candidates,
    _memory_event_datetime,
    _memory_limit_split,
    _memory_payload,
    _memory_priority,
    _memory_rank_penalty,
    _memory_rank_score,
    _memory_recency_boost,
    _memory_stale_reasons,
    _memory_state,
    _memory_thread_fields,
    _merge_memory_results,
    _merge_suggestion_payload,
    _metadata_dict,
    _normalize_paths,
    _normalize_datetime,
    _normalized_auto_memory_key,
    _parse_memory_locator,
    _parse_datetime_filter,
    _path_intent_boost,
    _path_query_term_boost,
    _project_index_paths,
    _project_pulse,
    _query_intents,
    _query_terms,
    _rank_score,
    _relative_to_project,
    _rerank_doc_results,
    _rerank_results,
    _resolve_superseded_memory,
    _result_base_fields,
    _result_key,
    _result_order_index,
    _rrf_merge,
    _list_thread_memory_results,
    _search_code_results,
    _search_docs_results,
    _search_memory_results,
    _session_narrative,
    _should_skip_session_capture,
    _single_line,
    _sort_memory_results,
    _stale_state,
    _success,
    _text_term_overlap,
    _text_term_similarity,
    _time_ago,
    _tool_error,
    _truncate,
    _validate_embedding_count,
    _validate_memory_content,
    _validate_memory_kind,
    _validate_query,
    _validate_tags,
    _validate_thread_id,
    _vector_match_score,
    _vibe_trust_status,
    _with_source_db,
)

# Also re-export ``embedding_provider_status`` so that tests which
# monkeypatch ``vibe_rag.tools.embedding_provider_status`` keep working.
from vibe_rag.indexing.embedder import embedding_provider_status  # noqa: F401

# Re-export server helpers that tests monkeypatch via ``vibe_rag.tools``.
from vibe_rag.server import (  # noqa: F401
    _ensure_project_id,
    _get_db,
    _get_embedder,
    _get_user_db,
)
