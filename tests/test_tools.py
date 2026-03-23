import pytest
from pathlib import Path

from vibe_rag.tools import (
    _embed_sync_with_progress,
    _memory_cleanup_candidates,
    _sort_memory_results,
    _merge_memory_results,
    _format_briefing,
    _query_terms,
    _hazard_scan,
    _rrf_merge,
    _index_project_impl,
    _live_decisions,
    _project_pulse,
    cleanup_duplicate_auto_memories,
    forget,
    ingest_daily_note,
    ingest_pr_outcome,
    index_project,
    load_session_context,
    project_status,
    remember,
    remember_structured,
    save_session_memory,
    save_session_summary,
    search,
    search_code,
    search_docs,
    search_memory,
    summarize_thread,
    supersede_memory,
    update_memory,
)


def _error_message(result):
    return result["error"]["message"]


def _search_paths(result):
    return [item["file_path"] for item in result["results"]]


def test_remember_and_search_memory(tmp_db, mock_embedder):
    result = remember("sqlite-vec is local and simple")
    assert result["ok"] is True
    assert result["memory"]["id"] == 1

    result = search_memory("what is good for vectors?")
    assert result["ok"] is True
    assert result["results"][0]["summary"] == "sqlite-vec is local and simple"


def test_fake_embedder_prefers_semantically_closer_memory(tmp_db, mock_embedder):
    remember_structured(
        summary="gateway token validation",
        details="gateway validates bearer tokens before routing requests",
        memory_kind="decision",
    )
    remember_structured(
        summary="billing invoice generation",
        details="billing creates invoices and receipts for customer orders",
        memory_kind="decision",
    )

    result = search_memory("who validates bearer tokens")

    assert result["ok"] is True
    assert result["results"][0]["summary"] == "gateway token validation"


def test_remember_handles_non_runtime_embedding_errors(tmp_db, mock_embedder, monkeypatch):
    import vibe_rag.server as srv

    monkeypatch.setattr(
        srv._embedder,
        "embed_text_sync",
        lambda _texts: (_ for _ in ()).throw(ValueError("provider unreachable")),
    )

    result = remember("embedding outage for freeform memory")

    assert result["ok"] is False
    assert result["error"]["code"] == "embedding_failed"
    assert "provider unreachable" in result["error"]["message"]


def test_search_code_handles_non_runtime_embedding_errors(indexed_project, mock_embedder, monkeypatch):
    import vibe_rag.server as srv

    monkeypatch.setattr(
        srv._embedder,
        "embed_code_query_sync",
        lambda _texts: (_ for _ in ()).throw(ValueError("provider unreachable")),
    )

    result = search_code("authentication")

    assert result["ok"] is False
    assert result["error"]["code"] == "embedding_failed"
    assert "provider unreachable" in result["error"]["message"]


def test_stale_state_reports_embedding_profile_mismatch(indexed_project, mock_embedder):
    import vibe_rag.server as srv
    from vibe_rag.tools import _stale_state

    metadata = srv._get_db().get_setting_json("project_index_metadata")
    assert metadata is not None
    metadata["embedding_profile"] = {
        "provider": "openrouter",
        "model": "custom/openrouter-model",
        "dimensions": 2560,
    }
    srv._get_db().set_setting_json("project_index_metadata", metadata)

    stale = _stale_state(srv._get_db(), indexed_project, srv._ensure_project_id())

    assert stale["is_incompatible"] is True
    assert any(warning["kind"] == "embedding_profile_changed" for warning in stale["warnings"])


def test_search_code_refuses_incompatible_index(indexed_project, mock_embedder):
    import vibe_rag.server as srv

    metadata = srv._get_db().get_setting_json("project_index_metadata")
    assert metadata is not None
    metadata["embedding_profile"] = {
        "provider": "openrouter",
        "model": "custom/openrouter-model",
        "dimensions": 2560,
    }
    srv._get_db().set_setting_json("project_index_metadata", metadata)

    result = search_code("authenticate")

    assert result["ok"] is False
    assert result["error"]["code"] == "incompatible_index"
    assert "embedding profile changed since last index" in result["error"]["message"]


def test_index_project_forces_full_rebuild_on_embedding_profile_change(tmp_db, mock_embedder, tmp_path: Path):
    import os
    import vibe_rag.server as srv

    (tmp_path / "auth.py").write_text("def authenticate(user, password):\n    return True\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        initial = index_project()
        assert initial["ok"] is True

        metadata = srv._get_db().get_setting_json("project_index_metadata")
        assert metadata is not None
        metadata["embedding_profile"] = {
            "provider": "openrouter",
            "model": "custom/openrouter-model",
            "dimensions": 2560,
        }
        srv._get_db().set_setting_json("project_index_metadata", metadata)

        rebuilt = index_project()
    finally:
        os.chdir(old_cwd)

    assert rebuilt["ok"] is True
    assert rebuilt["counts"]["code_unchanged"] == 0
    assert rebuilt["warnings"][0]["kind"] == "full_rebuild_required"
    assert rebuilt["summary"].startswith("Rebuilt index with openrouter:perplexity/pplx-embed-v1-4b@2560.")


def test_index_project_force_full_rebuild_resets_incremental_state_without_profile_change(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    (tmp_path / "auth.py").write_text("def authenticate(user, password):\n    return True\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        initial = index_project()
        assert initial["ok"] is True

        rebuilt = _index_project_impl(force_full_rebuild=True, rebuild_reason="test_force_full_rebuild")
    finally:
        os.chdir(old_cwd)

    assert rebuilt["ok"] is True
    assert rebuilt["full_rebuild"] is True
    assert rebuilt["counts"]["code_unchanged"] == 0
    assert rebuilt["warnings"][0]["kind"] == "full_rebuild_requested"
    assert rebuilt["warnings"][0]["reason"] == "test_force_full_rebuild"
    assert rebuilt["summary"].startswith("Rebuilt index with openrouter:perplexity/pplx-embed-v1-4b@2560.")


def test_search_memory_handles_non_runtime_embedding_errors(tmp_db, mock_embedder, monkeypatch):
    import vibe_rag.server as srv

    remember("gateway validates tokens")

    monkeypatch.setattr(
        srv._embedder,
        "embed_text_sync",
        lambda _texts: (_ for _ in ()).throw(ValueError("provider unavailable")),
    )

    result = search_memory("gateway")

    assert result["ok"] is False
    assert result["error"]["code"] == "embedding_failed"
    assert "provider unavailable" in result["error"]["message"]


def test_search_memory_exposes_thread_metadata_and_filters_by_thread_id(tmp_db, mock_embedder):
    remember(
        "gateway validates tokens for the auth refactor",
        metadata={"thread": {"id": "auth-refactor", "title": "Auth Refactor"}},
    )
    remember(
        "billing generates invoices for monthly statements",
        metadata={"thread_id": "billing-cleanup", "thread_title": "Billing Cleanup"},
    )

    result = search_memory("gateway validates tokens", thread_id="auth-refactor")

    assert result["ok"] is True
    assert result["result_total"] == 1
    assert result["results"][0]["thread_id"] == "auth-refactor"
    assert result["results"][0]["thread_title"] == "Auth Refactor"


def test_search_memory_filters_by_time_window(tmp_db, mock_embedder):
    remember(
        "retrospective auth note from earlier in the week",
        metadata={"thread_id": "daily-log", "event_at": "2026-03-20T12:00:00Z"},
    )
    remember(
        "retrospective auth note from today",
        metadata={"thread_id": "daily-log", "event_at": "2026-03-23T12:00:00Z"},
    )

    result = search_memory(
        "retrospective auth note",
        thread_id="daily-log",
        since="2026-03-22T00:00:00Z",
    )

    assert result["ok"] is True
    assert result["result_total"] == 1
    assert result["results"][0]["summary"] == "retrospective auth note from today"


def test_summarize_thread_returns_counts_and_latest_first(tmp_db, mock_embedder):
    remember(
        content="gateway owns token validation during the auth refactor",
        summary="gateway owns token validation",
        memory_kind="decision",
        metadata={"thread_id": "auth-refactor", "thread_title": "Auth Refactor", "event_at": "2026-03-22T09:00:00Z"},
    )
    remember(
        content="still need to remove legacy middleware after auth refactor",
        summary="remove legacy middleware",
        memory_kind="todo",
        scope="user",
        metadata={"thread_id": "auth-refactor", "thread_title": "Auth Refactor", "event_at": "2026-03-23T09:00:00Z"},
    )

    result = summarize_thread("auth-refactor")

    assert result["ok"] is True
    assert result["thread_title"] == "Auth Refactor"
    assert result["result_total"] == 2
    assert result["counts"]["by_kind"] == {"decision": 1, "todo": 1}
    assert result["counts"]["by_source_db"] == {"project": 1, "user": 1}
    assert result["results"][0]["summary"] == "remove legacy middleware"
    assert "Latest: remove legacy middleware" in result["summary"]


def test_ingest_daily_note_defaults_user_scope_and_convention_metadata(tmp_db, mock_embedder):
    result = ingest_daily_note(
        note_date="2026-03-23",
        summary="Worked through the auth refactor",
        details="Removed one legacy middleware branch and queued a follow-up.",
    )

    assert result["ok"] is True
    assert result["backend"] == "user-sqlite"
    assert result["adapter"] == "daily_note"
    assert result["convention"] == "memory_event_v1"
    assert result["memory"]["memory_kind"] == "summary"
    assert result["memory"]["thread_id"] == "daily:2026-03-23"
    assert result["memory"]["thread_title"] == "Daily Note 2026-03-23"
    assert result["memory"]["metadata"]["capture_kind"] == "adapter_daily_note"
    assert result["memory"]["metadata"]["note_date"] == "2026-03-23"
    assert result["memory"]["metadata"]["event_at"] == "2026-03-23T00:00:00Z"


def test_ingest_pr_outcome_defaults_project_scope_and_convention_metadata(tmp_db, mock_embedder):
    result = ingest_pr_outcome(
        pr_number=42,
        title="Fix auth refresh ordering",
        outcome="merged",
        issue_id="AUTH-17",
        branch="fix/auth-refresh-ordering",
        commit_sha="abc1234",
        pr_url="https://example.test/pr/42",
        details="Merged after review and smoke test.",
    )

    assert result["ok"] is True
    assert result["backend"] == "project-sqlite"
    assert result["adapter"] == "pr_outcome"
    assert result["convention"] == "memory_event_v1"
    assert result["memory"]["thread_id"] == "pr:42"
    assert result["memory"]["thread_title"] == "PR #42: Fix auth refresh ordering"
    assert result["memory"]["metadata"]["capture_kind"] == "adapter_pr_outcome"
    assert result["memory"]["metadata"]["pr_number"] == 42
    assert result["memory"]["metadata"]["outcome"] == "merged"
    assert result["memory"]["metadata"]["issue_id"] == "AUTH-17"
    assert "Outcome: merged" in result["memory"]["content"]
    assert "Merged after review and smoke test." in result["memory"]["content"]


def test_ingest_daily_note_rejects_invalid_date(tmp_db, mock_embedder):
    result = ingest_daily_note(
        note_date="03/23/2026",
        summary="Worked through the auth refactor",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_note_date"


def test_ingest_pr_outcome_rejects_invalid_pr_number(tmp_db, mock_embedder):
    result = ingest_pr_outcome(
        pr_number=0,
        title="Fix auth refresh ordering",
        outcome="merged",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_pr_number"


def test_remember_inferrs_constraint_kind_from_freeform_content(tmp_db, mock_embedder):
    remember("Only demo tokens are allowed in this smoke-test API.")

    result = load_session_context("demo tokens", memory_limit=4, code_limit=0, docs_limit=0)

    assert result["memories"][0]["memory_kind"] == "constraint"


def test_forget_existing(tmp_db, mock_embedder):
    result = remember("temporary fact")
    assert result["memory"]["id"] == 1

    result = forget(1)
    assert result["ok"] is True
    assert result["deleted"] is True
    assert result["content_preview"] == "temporary fact"


def test_forget_nonexistent(tmp_db, mock_embedder):
    result = forget(999)
    assert result["ok"] is False
    assert _error_message(result) == "memory 999 not found"


def test_forget_with_source_qualified_id(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    project_id = srv._get_db().remember_structured(
        summary="project fact",
        content="project fact",
        embedding=[0.0] * 2560,
        project_id=srv._ensure_project_id(),
        memory_kind="fact",
        metadata={"capture_kind": "manual"},
    )
    user_id = srv._get_user_db().remember_structured(
        summary="user fact",
        content="user fact",
        embedding=[0.0] * 2560,
        project_id=srv._ensure_project_id(),
        memory_kind="fact",
        metadata={"capture_kind": "manual"},
    )

    assert project_id == user_id == 1

    ambiguous = forget("1")
    assert ambiguous["ok"] is False
    assert ambiguous["error"]["code"] == "ambiguous_memory_id"

    deleted = forget("project:1")
    assert deleted["ok"] is True
    assert deleted["backend"] == "project-sqlite"
    assert srv._get_db().get_memory(1) is None
    assert srv._get_user_db().get_memory(1) is not None


def test_search_code_empty(tmp_db, mock_embedder):
    result = search_code("anything")
    assert result["ok"] is False
    assert result["error"]["code"] == "no_code_index"


def test_search_docs_empty(tmp_db, mock_embedder):
    result = search_docs("anything")
    assert result["ok"] is False
    assert result["error"]["code"] == "no_docs_index"


def test_index_project_real_dir(tmp_db, mock_embedder, tmp_path: Path):
    import os
    (tmp_path / "hello.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "readme.md").write_text("## Overview\n\nThis is a test project.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert "1 code files" in result["summary"]
    assert "1 docs" in result["summary"]


def test_index_project_emits_progress_events(tmp_db, mock_embedder, tmp_path: Path):
    import os

    events = []
    (tmp_path / "hello.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "readme.md").write_text("## Overview\n\nThis is a test project.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = _index_project_impl(progress_callback=events.append)
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert "Indexed 1 code files" in result["summary"]
    phases = [event["phase"] for event in events]
    assert "file_discovery_complete" in phases
    assert "code_chunking_complete" in phases
    assert "doc_chunking_complete" in phases
    assert "index_complete" in phases


def test_index_project_accepts_relative_paths_argument(tmp_db, mock_embedder, tmp_path: Path):
    import os
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "hello.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "notes.md").write_text("## Overview\n\nRelative path indexing works.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project(paths=["."])
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert "1 code files" in result["summary"]
    assert "1 docs" in result["summary"]


def test_index_project_accepts_string_paths_argument(tmp_db, mock_embedder, tmp_path: Path):
    import os
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "hello.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "notes.md").write_text("## Overview\n\nString path indexing works.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project(paths=".")
        search_result = search_code("hello")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert "1 code files" in result["summary"]
    assert "1 docs" in result["summary"]
    assert "pkg/hello.py" in _search_paths(search_result)


def test_search_code_after_index(tmp_db, mock_embedder, tmp_path: Path):
    import os
    (tmp_path / "auth.py").write_text("def authenticate(user, password):\n    return True\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_code("authentication")
    finally:
        os.chdir(old_cwd)

    assert "auth.py" in _search_paths(result)


def test_search_docs_after_index(tmp_db, mock_embedder, tmp_path: Path):
    import os
    (tmp_path / "guide.md").write_text("## Deployment\n\nDeploy to Railway with docker.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_docs("how to deploy")
    finally:
        os.chdir(old_cwd)

    assert "guide.md" in _search_paths(result)


def test_release_docs_queries_prefer_real_docs_over_eval_noise(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "README.md").write_text("## Release\n\nRelease notes explain the publish workflow.\n")
    (tmp_path / "CHANGELOG.md").write_text("## Release Notes\n\nPublish the tag and update release notes.\n")
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "local_repos.toml").write_text('query = "release publish tag workflow and release notes"\n')

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_docs("release publish tag workflow and release notes", limit=2)
    finally:
        os.chdir(old_cwd)

    assert "README.md" in _search_paths(result) or "CHANGELOG.md" in _search_paths(result)
    assert "evals/local_repos.toml" not in _search_paths(result)


def test_release_docs_queries_can_find_changelog_with_semantic_plus_lexical(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "CHANGELOG.md").write_text("## Release Notes\n\nCreate the release tag and publish workflow notes here.\n")
    (tmp_path / "AGENTS.md").write_text("General coding rules.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_docs("release publish tag workflow and release notes", limit=1)
    finally:
        os.chdir(old_cwd)

    assert "CHANGELOG.md" in _search_paths(result)


def test_release_automation_doc_queries_prefer_changelog_over_agents(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    (tmp_path / "CHANGELOG.md").write_text("Release notes for the publish workflow and release.published event.\n")
    (tmp_path / "AGENTS.md").write_text("Maintainer release guide.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "publish workflow for release.published event",
            memory_limit=0,
            code_limit=0,
            docs_limit=1,
        )
    finally:
        os.chdir(old_cwd)

    assert payload["docs"][0]["file_path"] == "CHANGELOG.md"


def test_release_procedure_queries_surface_agents_and_changelog(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "AGENTS.md").write_text("Release workflow: update the changelog, push main, create the GitHub release.\n")
    (tmp_path / "CHANGELOG.md").write_text("## v0.0.18\n\nRelease notes for the publish.\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "billing.py").write_text("def create_invoice(customer_id):\n    return customer_id\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_docs("release procedure and maintainer steps for changelog publish tag workflow", limit=2)
    finally:
        os.chdir(old_cwd)

    assert "AGENTS.md" in _search_paths(result)
    assert "CHANGELOG.md" in _search_paths(result)


def test_release_procedure_queries_prefer_repo_agents_over_template_agents(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    (tmp_path / "AGENTS.md").write_text("Release procedure for maintainers.\n")
    template_dir = tmp_path / "src" / "vibe_rag" / "templates"
    template_dir.mkdir(parents=True)
    (template_dir / "AGENTS.md").write_text("Generated project maintainer template.\n")
    (tmp_path / "CHANGELOG.md").write_text("Release notes.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "release procedure and maintainer steps for changelog publish tag workflow",
            memory_limit=0,
            code_limit=0,
            docs_limit=2,
        )
    finally:
        os.chdir(old_cwd)

    doc_paths = [item["file_path"] for item in payload["docs"]]
    assert "AGENTS.md" in doc_paths
    assert "src/vibe_rag/templates/AGENTS.md" not in doc_paths


def test_setup_doc_queries_prefer_readme_and_docs_over_operational_docs(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    (tmp_path / "README.md").write_text("ACP setup for editors and IDE integration.\n")
    (tmp_path / "AGENTS.md").write_text("Maintainer notes.\n")
    (tmp_path / "CHANGELOG.md").write_text("Release notes.\n")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "acp-setup.md").write_text("Set up vibe-acp for Zed, JetBrains, and Neovim.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "vibe acp setup in editors and ide integration",
            memory_limit=0,
            code_limit=0,
            docs_limit=2,
        )
    finally:
        os.chdir(old_cwd)

    doc_paths = [item["file_path"] for item in payload["docs"]]
    assert "README.md" in doc_paths or "docs/acp-setup.md" in doc_paths
    assert "AGENTS.md" not in doc_paths


def test_bootstrap_doc_queries_prefer_setup_docs_over_agents_and_changelog(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    (tmp_path / "AGENTS.md").write_text("Maintainer notes.\n")
    (tmp_path / "CHANGELOG.md").write_text("Release notes.\n")
    (tmp_path / "README.md").write_text("Session bootstrap hook for codex and MCP startup.\n")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "setup-guide.md").write_text("Codex hook bootstrap and MCP startup setup guide.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "session bootstrap hook for codex and mcp startup",
            memory_limit=0,
            code_limit=0,
            docs_limit=2,
        )
    finally:
        os.chdir(old_cwd)

    doc_paths = [item["file_path"] for item in payload["docs"]]
    assert "README.md" in doc_paths or "docs/setup-guide.md" in doc_paths
    assert "AGENTS.md" not in doc_paths


def test_resume_doc_queries_prefer_readme_over_changelog(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    (tmp_path / "README.md").write_text("Use --continue or --resume to load a saved session.\n")
    (tmp_path / "CHANGELOG.md").write_text("Release notes and version history.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "resume continue load session metadata and messages",
            memory_limit=0,
            code_limit=0,
            docs_limit=1,
        )
    finally:
        os.chdir(old_cwd)

    assert payload["docs"][0]["file_path"] == "README.md"


def test_procedural_docs_queries_prefer_operational_docs_over_finish_line_plan(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    docs_dir = tmp_path / "docs" / "docs"
    docs_dir.mkdir(parents=True)
    (tmp_path / "FINISH-LINE-PLAN.md").write_text("Action backlog and planning notes for signal bus cleanup.\n")
    (docs_dir / "signal-bus.md").write_text(
        "Signal bus architecture, immutable signals, decision lifecycle, and context injection.\n"
    )

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "signal bus architecture immutable signals decisions and context injection",
            memory_limit=0,
            code_limit=0,
            docs_limit=1,
        )
    finally:
        os.chdir(old_cwd)

    assert payload["docs"][0]["file_path"] == "docs/docs/signal-bus.md"


def test_mcp_doc_queries_prefer_mcp_tools_docs_over_generic_architecture(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    mcp_server_dir = tmp_path / "mcp-server"
    mcp_server_dir.mkdir()
    (docs_dir / "MCP-TOOLS.md").write_text("Available MCP tools and routes.\n")
    (docs_dir / "ARCHITECTURE.md").write_text("General architecture.\n")
    (mcp_server_dir / "README.md").write_text("MCP server overview.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        payload = load_session_context(
            "mcp tools server typescript route integrations and available tool docs",
            memory_limit=0,
            code_limit=0,
            docs_limit=1,
        )
    finally:
        os.chdir(old_cwd)

    assert payload["docs"][0]["file_path"] == "docs/MCP-TOOLS.md"


def test_release_automation_queries_surface_workflow_file(tmp_db, mock_embedder, tmp_path: Path):
    import os

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "publish.yml").write_text("name: publish\non:\n  release:\n    types: [published]\n")
    (tmp_path / "README.md").write_text("General project overview.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_code("release automation github workflow publish tag", limit=3)
    finally:
        os.chdir(old_cwd)

    assert ".github/workflows/publish.yml" in _search_paths(result)


def test_rrf_merge_boosts_multi_source_hits_and_tracks_sources():
    shared = {
        "file_path": "shared.py",
        "chunk_index": 0,
        "content": "def shared():\n    return 'shared'\n",
        "language": "python",
        "symbol": "shared",
        "start_line": 1,
        "end_line": 2,
        "indexed_at": "2026-03-23T00:00:00Z",
    }
    merged = _rrf_merge(
        (
            "vector",
            [
                {
                    "file_path": "vector_only.py",
                    "chunk_index": 0,
                    "content": "def vector_only():\n    return 1\n",
                    "language": "python",
                    "symbol": "vector_only",
                    "start_line": 1,
                    "end_line": 2,
                    "indexed_at": "2026-03-23T00:00:00Z",
                    "distance": 0.1,
                },
                {**shared, "distance": 0.4},
            ],
        ),
        (
            "lexical",
            [
                {
                    "file_path": "lexical_only.py",
                    "chunk_index": 0,
                    "content": "def lexical_only():\n    return 1\n",
                    "language": "python",
                    "symbol": "lexical_only",
                    "start_line": 1,
                    "end_line": 2,
                    "indexed_at": "2026-03-23T00:00:00Z",
                    "score": 1.0,
                },
                {**shared, "score": 0.5},
            ],
        ),
        (
            "workflow",
            [
                {**shared, "score": 1.0},
            ],
        ),
        limit=5,
    )

    assert merged[0]["file_path"] == "shared.py"
    assert merged[0]["match_sources"] == ["vector", "lexical", "workflow"]
    assert merged[0]["vector_distance"] == 0.4
    assert "score" not in merged[0]
    assert "distance" not in merged[0]
    assert merged[0]["rank_score"] > merged[1]["rank_score"]


def test_load_session_context_bundles_memory_code_and_docs(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "billing.py").write_text("def create_invoice(customer_id):\n    return customer_id\n")
    (tmp_path / "billing.md").write_text("## Billing\n\nInvoices are created in the billing flow.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        remember("billing uses invoices and customer ids")
        result = load_session_context("continue the billing flow", memory_limit=3, code_limit=3, docs_limit=2)
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert result["memories"][0]["content"] == "billing uses invoices and customer ids"
    assert result["memories"][0]["project_id"] == result["project_id"]
    assert result["code"][0]["file_path"] == "billing.py"
    assert result["code"][0]["start_line"] == 1
    assert result["code"][0]["rank_score"] > 0
    assert set(result["code"][0]["match_sources"]) >= {"vector", "lexical"}
    assert "score" not in result["code"][0]
    assert result["code"][0]["provenance"]["source"] == "project-index"
    assert result["docs"][0]["file_path"] == "billing.md"
    assert result["docs"][0]["rank_score"] > 0
    assert set(result["docs"][0]["match_sources"]) >= {"vector", "lexical"}
    assert "score" not in result["docs"][0]
    assert result["docs"][0]["provenance"]["source"] == "project-index"


def test_project_pulse_returns_branch_and_workspace(tmp_path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 1\n")

    from vibe_rag.tools import _project_pulse

    pulse = _project_pulse(tmp_path)

    assert pulse["branch"] is not None
    assert pulse["workspace"]["is_clean"] is False
    assert "file.py" in str(pulse["workspace"]["modified"]) or "file.py" in str(pulse["workspace"]["untracked"])
    assert len(pulse["recent_commits"]) >= 1
    assert pulse["recent_commits"][0]["message"] == "init"


def test_project_pulse_first_modified_file_is_not_misparsed_as_staged(tmp_path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "file.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 2\n")

    from vibe_rag.tools import _project_pulse

    pulse = _project_pulse(tmp_path)

    assert pulse["workspace"]["modified"] == ["file.py"]
    assert pulse["workspace"]["staged"] == []


def test_project_pulse_non_git_directory(tmp_path):
    from vibe_rag.tools import _project_pulse

    pulse = _project_pulse(tmp_path)

    assert pulse["branch"] is None
    assert pulse["workspace"] is None
    assert pulse["recent_commits"] == []


def test_project_pulse_ahead_behind_on_branch(tmp_path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "feature work"], cwd=tmp_path, check=True, capture_output=True)

    from vibe_rag.tools import _project_pulse

    pulse = _project_pulse(tmp_path)

    assert pulse["branch"] == "feature"
    assert pulse["default_branch"] is None
    assert pulse["is_default_branch"] is None


def test_session_narrative_with_enriched_memories(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    user_db = srv._get_user_db()
    project_id = srv._ensure_project_id()

    user_db.remember_structured(
        summary="Fixed auth token refresh",
        content="Fixed auth token refresh logic in gateway",
        embedding=[0.0] * 2560,
        project_id=project_id,
        memory_kind="summary",
        metadata={"capture_kind": "session_distillation", "topic": "auth", "outcome": "completed"},
    )

    from vibe_rag.tools import _session_narrative

    narrative = _session_narrative(user_db, project_id)

    assert narrative is not None
    assert "auth" in narrative.lower()


def test_session_narrative_no_session_memories(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    from vibe_rag.tools import _session_narrative

    narrative = _session_narrative(srv._get_user_db(), srv._ensure_project_id())
    assert narrative is None


def test_session_narrative_degrades_for_pre_v010_memories(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    user_db = srv._get_user_db()
    project_id = srv._ensure_project_id()

    user_db.remember_structured(
        summary="Session covered 3 turns about config loading",
        content="Session covered 3 turns.",
        embedding=[0.0] * 2560,
        project_id=project_id,
        memory_kind="summary",
        metadata={"capture_kind": "session_rollup"},
    )

    from vibe_rag.tools import _session_narrative

    narrative = _session_narrative(user_db, project_id)

    assert narrative is not None
    assert "last session:" in narrative.lower()
    assert "config loading" in narrative.lower() or "session" in narrative.lower()


def test_session_narrative_cleans_numbered_pre_v010_summary(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    user_db = srv._get_user_db()
    project_id = srv._ensure_project_id()

    user_db.remember_structured(
        summary="1. load session context for understanding this repo 2. check the current project status 3. search the code",
        content="Session covered 3 turns.",
        embedding=[0.0] * 2560,
        project_id=project_id,
        memory_kind="summary",
        metadata={"capture_kind": "session_rollup"},
    )

    from vibe_rag.tools import _session_narrative

    narrative = _session_narrative(user_db, project_id)

    assert narrative is not None
    assert "last session:" in narrative.lower()
    assert "1." not in narrative
    assert "; check the current project status" in narrative.lower()


def test_session_narrative_filters_by_project_id(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    user_db = srv._get_user_db()

    user_db.remember_structured(
        summary="Work on other project",
        content="Work on other project",
        embedding=[0.0] * 2560,
        project_id="other-project-abc",
        memory_kind="summary",
        metadata={"capture_kind": "session_distillation", "topic": "billing"},
    )

    from vibe_rag.tools import _session_narrative

    narrative = _session_narrative(user_db, srv._ensure_project_id())
    assert narrative is None


def test_hazard_scan_empty_index(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    hazards = _hazard_scan(srv._get_db(), Path.cwd(), srv._ensure_project_id(), {"recent_commits": [], "workspace": None})
    categories = [item["category"] for item in hazards]
    assert "no_index" in categories


def test_hazard_scan_provider_unavailable(tmp_db, mock_embedder, monkeypatch):
    import vibe_rag.server as srv
    import vibe_rag.tools as tools_mod

    monkeypatch.setattr(tools_mod, "embedding_provider_status", lambda: {"ok": False, "provider": "openrouter", "detail": "not reachable"})
    hazards = _hazard_scan(srv._get_db(), Path.cwd(), srv._ensure_project_id(), {"recent_commits": [], "workspace": None})
    categories = [item["category"] for item in hazards]
    assert "provider_unavailable" in categories


def test_hazard_scan_stale_index(tmp_db, mock_embedder, monkeypatch, tmp_path):
    import os
    import subprocess
    import vibe_rag.server as srv
    import vibe_rag.tools as tools_mod

    monkeypatch.setattr(tools_mod, "embedding_provider_status", lambda: {"ok": True, "provider": "mock"})

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init", "--no-verify"], cwd=tmp_path, check=True, capture_output=True)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        subprocess.run(["git", "commit", "--allow-empty", "-m", "new commit"], cwd=tmp_path, check=True, capture_output=True)
        new_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
        pulse = {
            "recent_commits": [{"sha": new_head, "message": "new commit"}],
            "workspace": {"modified": [], "staged": [], "untracked": [], "is_clean": True},
        }
        hazards = _hazard_scan(srv._get_db(), tmp_path, srv._ensure_project_id(), pulse)
    finally:
        os.chdir(old_cwd)

    categories = [item["category"] for item in hazards]
    assert "stale_index" in categories


def test_hazard_scan_uncommitted_work(tmp_db, mock_embedder, monkeypatch):
    import vibe_rag.server as srv
    import vibe_rag.tools as tools_mod

    monkeypatch.setattr(tools_mod, "embedding_provider_status", lambda: {"ok": True, "provider": "mock"})
    hazards = _hazard_scan(
        srv._get_db(),
        Path.cwd(),
        srv._ensure_project_id(),
        {
            "recent_commits": [],
            "workspace": {"modified": ["a.py", "b.py"], "staged": [], "untracked": [], "is_clean": False},
        },
    )
    categories = [item["category"] for item in hazards]
    assert "uncommitted_work" in categories
    msg = next(item["message"] for item in hazards if item["category"] == "uncommitted_work")
    assert "2 files" in msg


def test_hazard_scan_everything_healthy(tmp_db, mock_embedder, monkeypatch, tmp_path):
    import os
    import subprocess
    import vibe_rag.server as srv
    import vibe_rag.tools as tools_mod

    monkeypatch.setattr(tools_mod, "embedding_provider_status", lambda: {"ok": True, "provider": "mock"})

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init", "--no-verify"], cwd=tmp_path, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        pulse = {"recent_commits": [{"sha": head, "message": "init"}], "workspace": {"modified": [], "staged": [], "untracked": [], "is_clean": True}}
        hazards = _hazard_scan(srv._get_db(), tmp_path, srv._ensure_project_id(), pulse)
    finally:
        os.chdir(old_cwd)

    assert hazards == []


def test_live_decisions_returns_only_decisions_and_constraints(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    project_id = srv._ensure_project_id()
    db = srv._get_db()
    db.remember_structured(summary="gateway owns tokens", content="gateway owns tokens", embedding=[0.0] * 2560, project_id=project_id, memory_kind="decision")
    db.remember_structured(summary="max 100 retries", content="max 100 retries", embedding=[0.0] * 2560, project_id=project_id, memory_kind="constraint")
    db.remember_structured(summary="session note", content="session note", embedding=[0.0] * 2560, project_id=project_id, memory_kind="note")

    decisions = _live_decisions(db, srv._get_user_db(), project_id)
    kinds = [item["memory_kind"] for item in decisions]
    assert "decision" in kinds
    assert "constraint" in kinds
    assert "note" not in kinds
    assert all("score" not in item for item in decisions)


def test_live_decisions_excludes_superseded(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    project_id = srv._ensure_project_id()
    db = srv._get_db()
    old_id = db.remember_structured(summary="old decision", content="old decision", embedding=[0.0] * 2560, project_id=project_id, memory_kind="decision")
    db.remember_structured(
        summary="new decision",
        content="new decision",
        embedding=[0.0] * 2560,
        project_id=project_id,
        memory_kind="decision",
        supersedes=old_id,
    )

    decisions = _live_decisions(db, srv._get_user_db(), project_id)
    summaries = [item["summary"] for item in decisions]
    assert "new decision" in summaries
    assert "old decision" not in summaries


def test_format_briefing_full_output():
    pulse = {
        "branch": "main",
        "is_default_branch": True,
        "default_branch": "main",
        "workspace": {"modified": ["tools.py"], "staged": [], "untracked": [], "is_clean": False},
        "recent_commits": [{"sha": "abc1234", "message": "fix auth"}],
    }
    narrative = "You were last here 2 hours ago working on auth (completed)."
    hazards = [
        {"level": "error", "category": "no_index", "message": "No code index"},
        {"level": "warning", "category": "uncommitted_work", "message": "1 files modified"},
    ]
    decisions = [{"summary": "gateway owns tokens", "memory_kind": "decision", "updated_at": "2026-03-23 10:00:00"}]
    task_results = {
        "memories": [],
        "code": [
            {
                "file_path": "src/billing.py",
                "start_line": 12,
                "symbol": None,
                "content": "def issue_invoice(customer_id: str) -> str:",
            }
        ],
        "docs": [],
    }

    briefing = _format_briefing(pulse, narrative, hazards, decisions, task_results, "test-project")

    assert "test-project" in briefing
    assert "main" in briefing
    assert "1 modified file" in briefing
    assert "You were last here" in briefing
    assert "No code index" in briefing
    assert "gateway owns tokens" in briefing
    assert "src/billing.py:12 def issue_invoice" in briefing
    assert briefing.index("No code index") < briefing.index("1 files modified")


def test_format_briefing_new_project():
    pulse = {"branch": "main", "is_default_branch": True, "default_branch": "main", "workspace": {"modified": [], "staged": [], "untracked": [], "is_clean": True}, "recent_commits": []}
    briefing = _format_briefing(pulse, None, [], [], {"memories": [], "code": [], "docs": []}, "new-project")

    assert "new-project" in briefing
    assert "You were last here" not in briefing
    assert "!" not in briefing
    assert "Decisions" not in briefing


def test_format_briefing_excludes_low_signal_auto_memory():
    pulse = {"branch": "main", "is_default_branch": True, "default_branch": "main", "workspace": {"modified": [], "staged": [], "untracked": [], "is_clean": True}, "recent_commits": []}
    task_results = {
        "memories": [
            {
                "memory_kind": "summary",
                "summary": "hello",
                "content": "hello",
                "metadata": {"capture_kind": "session_rollup", "turn_count": 1, "task": "hello"},
                "provenance": {"capture_kind": "session_rollup", "is_current_project": True},
                "is_stale": False,
            }
        ],
        "code": [],
        "docs": [],
    }

    briefing = _format_briefing(pulse, None, [], [], task_results, "briefing-project")

    assert "Memory:" not in briefing
    assert "hello" not in briefing


def test_format_briefing_prefers_manual_memory_over_auto_capture():
    pulse = {"branch": "main", "is_default_branch": True, "default_branch": "main", "workspace": {"modified": [], "staged": [], "untracked": [], "is_clean": True}, "recent_commits": []}
    task_results = {
        "memories": [
            {
                "memory_kind": "summary",
                "summary": "Session summary for auth work and rollout status",
                "content": "Session covered auth validation ownership and rollout status across gateway and billing.",
                "metadata": {"capture_kind": "session_rollup", "turn_count": 3, "task": "continue auth work"},
                "provenance": {"capture_kind": "session_rollup", "is_current_project": True},
                "is_stale": False,
            },
            {
                "memory_kind": "decision",
                "summary": "Gateway owns auth token validation",
                "content": "Gateway owns auth token validation",
                "metadata": {"capture_kind": "manual"},
                "provenance": {"capture_kind": "manual", "is_current_project": True},
                "is_stale": False,
            },
        ],
        "code": [],
        "docs": [],
    }

    briefing = _format_briefing(pulse, None, [], [], task_results, "briefing-project")

    assert "Gateway owns auth token validation" in briefing
    assert "Session summary for auth work and rollout status" not in briefing


def test_format_briefing_budget():
    pulse = {
        "branch": "main",
        "is_default_branch": True,
        "default_branch": "main",
        "workspace": {"modified": [], "staged": [], "untracked": [], "is_clean": True},
        "recent_commits": [],
    }
    long_narrative = "x " * 4000
    briefing = _format_briefing(pulse, long_narrative, [], [], {"memories": [], "code": [], "docs": []}, "budget-test")

    assert len(briefing) <= 6000


def test_load_session_context_includes_briefing(tmp_db, mock_embedder, tmp_path):
    import os
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "notes.md").write_text("## Notes\n\nProject notes.\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init", "--no-verify"], cwd=tmp_path, check=True, capture_output=True)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        remember("billing uses invoices and customer ids")
        result = load_session_context("understand the codebase", code_limit=1, docs_limit=1, memory_limit=1)
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert isinstance(result.get("briefing"), str)
    assert result["pulse"]["branch"] is not None
    assert isinstance(result["hazards"], list)
    assert isinstance(result["live_decisions"], list)
    assert "memories" in result
    assert "code" in result
    assert "docs" in result


def test_load_session_context_briefing_on_fresh_project(tmp_db, mock_embedder):
    result = load_session_context("explore new project", code_limit=0, docs_limit=0, memory_limit=0)

    assert result["ok"] is True
    briefing = result["briefing"]
    assert "You were last here" not in briefing
    assert "Decisions" not in briefing


def test_load_session_context_backward_compat(tmp_db, mock_embedder, tmp_path):
    import os

    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = load_session_context("test compat", code_limit=0, docs_limit=0, memory_limit=0)
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert isinstance(result.get("memories"), list)
    assert isinstance(result.get("code"), list)
    assert isinstance(result.get("docs"), list)


def test_load_session_context_computes_pulse_before_db(monkeypatch):
    call_order = []

    monkeypatch.setattr(
        "vibe_rag.tools._project_pulse",
        lambda project_root: call_order.append("pulse") or {"branch": None, "workspace": None, "recent_commits": []},
    )
    monkeypatch.setattr("vibe_rag.tools._get_db", lambda: call_order.append("db") or object())
    monkeypatch.setattr("vibe_rag.tools._stale_state", lambda *args, **kwargs: {})
    monkeypatch.setattr("vibe_rag.tools._get_user_db", lambda: object())
    monkeypatch.setattr("vibe_rag.tools._session_narrative", lambda *args, **kwargs: None)
    monkeypatch.setattr("vibe_rag.tools._hazard_scan", lambda *args, **kwargs: [])
    monkeypatch.setattr("vibe_rag.tools._live_decisions", lambda *args, **kwargs: [])
    monkeypatch.setattr("vibe_rag.tools._search_memory_results", lambda *args, **kwargs: (None, []))
    monkeypatch.setattr("vibe_rag.tools._search_code_results", lambda *args, **kwargs: (None, []))
    monkeypatch.setattr("vibe_rag.tools._search_docs_results", lambda *args, **kwargs: (None, []))
    monkeypatch.setattr("vibe_rag.tools._format_briefing", lambda *args, **kwargs: "briefing")

    result = load_session_context("test pulse ordering", code_limit=0, docs_limit=0, memory_limit=0)

    assert result["ok"] is True
    assert call_order[:2] == ["pulse", "db"]


def test_current_file_counts_skips_cache_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    (tmp_path / ".mypy_cache").mkdir()
    (tmp_path / ".mypy_cache" / "state.json").write_text("{}\n")
    (tmp_path / ".ruff_cache").mkdir()
    (tmp_path / ".ruff_cache" / "state.json").write_text("{}\n")

    from vibe_rag.tools import _current_file_counts

    code_count, doc_count = _current_file_counts(tmp_path)

    assert code_count == 1
    assert doc_count == 1


def test_stale_state_skips_file_count_drift_when_git_head_changed(tmp_db, mock_embedder, monkeypatch, tmp_path):
    import os
    import subprocess
    import vibe_rag.server as srv

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init", "--no-verify"], cwd=tmp_path, check=True, capture_output=True)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        subprocess.run(["git", "commit", "--allow-empty", "-m", "next", "--no-verify"], cwd=tmp_path, check=True, capture_output=True)

        monkeypatch.setattr(
            "vibe_rag.tools._current_file_counts",
            lambda project_root: (_ for _ in ()).throw(AssertionError("should not count files when git head already changed")),
        )

        from vibe_rag.tools import _stale_state

        stale = _stale_state(srv._get_db(), tmp_path, srv._ensure_project_id())
    finally:
        os.chdir(old_cwd)

    warning_kinds = [warning["kind"] for warning in stale["warnings"]]
    assert "git_head_changed" in warning_kinds
    assert "file_count_drift" not in warning_kinds


# --- Edge case tests ---


def test_index_project_no_api_key(tmp_db, tmp_path: Path, monkeypatch):
    """index_project should return an error string when the default embedder is unavailable."""
    import os
    import vibe_rag.server as srv

    old_embedder = srv._embedder
    srv._embedder = None
    monkeypatch.delenv("RAG_OR_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_MOD", raising=False)
    monkeypatch.delenv("RAG_OR_EMBED_DIM", raising=False)
    monkeypatch.delenv("RAG_DB", raising=False)
    monkeypatch.delenv("RAG_USER_DB", raising=False)

    (tmp_path / "hello.py").write_text("x = 1\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)
        srv._embedder = old_embedder

    assert result["ok"] is False
    assert "RAG_OR_API_KEY not set" in _error_message(result)


def test_index_project_no_files(tmp_db, mock_embedder, tmp_path: Path):
    """index_project returns a message when directory has no indexable files."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is False
    assert _error_message(result) == "no files found to index"


def test_embed_sync_with_progress_propagates_internal_typeerror():
    def broken_embed(texts, *, progress_callback=None):
        raise TypeError("internal embed bug")

    with pytest.raises(TypeError, match="internal embed bug"):
        _embed_sync_with_progress(
            broken_embed,
            ["hello"],
            progress_callback=lambda event: None,
        )


def test_search_code_with_language_filter(tmp_db, mock_embedder, tmp_path: Path):
    """search_code with language filter should work after indexing."""
    import os
    (tmp_path / "app.py").write_text("def run():\n    pass\n")
    (tmp_path / "util.js").write_text("function help() { return 1; }\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_code("run", language="python")
    finally:
        os.chdir(old_cwd)

    # Should return results (may or may not filter depending on chunk metadata)
    assert result["ok"] is True
    assert "app.py" in _search_paths(result)


def test_remember_with_tags(tmp_db, mock_embedder):
    result = remember("auth uses JWT tokens", tags="architecture,security")
    assert result["ok"] is True
    assert result["memory"]["tags"] == ["architecture", "security"]

    result = search_memory("JWT")
    assert result["ok"] is True
    assert "JWT" in result["results"][0]["content"]


def test_remember_structured_returns_memory_payload(tmp_db, mock_embedder):
    result = remember_structured(
        summary="auth decisions live in the gateway",
        details="The API gateway validates JWT tokens before forwarding requests.",
        memory_kind="decision",
        tags="auth,architecture",
        metadata={"confidence": "high"},
        source_session_id="sess-1",
    )

    assert result["ok"] is True
    assert result["memory"]["summary"] == "auth decisions live in the gateway"
    assert result["memory"]["memory_kind"] == "decision"
    assert result["memory"]["metadata"]["capture_kind"] == "manual"
    assert result["memory"]["metadata"]["confidence"] == "high"


def test_remember_marks_freeform_memory_metadata(tmp_db, mock_embedder):
    remember("freeform note about auth")
    result = load_session_context("freeform note", memory_limit=1, code_limit=0, docs_limit=0)

    assert result["memories"][0]["memory_kind"] == "note"
    assert result["memories"][0]["metadata"]["capture_kind"] == "freeform"
    assert result["memories"][0]["provenance"]["capture_kind"] == "freeform"
    assert result["memories"][0]["provenance"]["source_type"] == "freeform"


def test_save_session_memory_distills_and_deduplicates(tmp_db, mock_embedder):
    first = save_session_memory(
        task="figure out auth ownership",
        response="The API gateway validates JWT tokens before requests reach downstream services.",
        source_session_id="sess-1",
        source_message_id="msg-1",
        user_message_id="user-1",
    )
    second = save_session_memory(
        task="figure out auth ownership",
        response="The API gateway validates JWT tokens before requests reach downstream services.",
        source_session_id="sess-1",
        source_message_id="msg-1",
        user_message_id="user-1",
    )

    assert first["ok"] is True
    assert first["deduplicated"] is False
    assert first["memory"]["memory_kind"] == "summary"
    assert first["memory"]["metadata"]["capture_kind"] == "session_distillation"
    assert first["memory"]["metadata"]["user_message_id"] == "user-1"
    assert first["memory"]["provenance"]["capture_kind"] == "session_distillation"
    assert first["memory"]["provenance"]["source_type"] == "session_distillation"
    assert "Task: figure out auth ownership" in first["memory"]["content"]
    assert second["ok"] is True
    assert second["deduplicated"] is True
    assert second["memory"]["id"] == first["memory"]["id"]


def test_save_session_memory_infers_enriched_metadata(tmp_db, mock_embedder):
    result = save_session_memory(
        task="Fix the auth token refresh bug",
        response="Fixed the token refresh logic. The gateway now validates tokens correctly and refresh passes.",
        source_session_id="sess-meta",
        source_message_id="msg-meta",
    )

    assert result["ok"] is True
    assert "topic" in result["memory"]["metadata"]
    assert "outcome" in result["memory"]["metadata"]
    assert "session_ended_at" in result["memory"]["metadata"]
    assert result["memory"]["metadata"]["topic"]
    assert result["memory"]["metadata"]["outcome"] in {"completed", "in_progress", "blocked"}


def test_save_session_memory_skips_low_signal_no_memory_response(tmp_db, mock_embedder):
    result = save_session_memory(
        task="What durable memory do you have?",
        response="I have no durable memory about this task.",
        source_session_id="sess-none",
        source_message_id="msg-none",
    )

    assert result["ok"] is True
    assert result["skipped"] is True


def test_save_session_memory_skips_low_signal_auto_memory(tmp_db, mock_embedder):
    result = save_session_memory(
        task="hi",
        response="hello",
        source_session_id="sess-hi",
        source_message_id="msg-hi",
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "low-signal auto memory"


def test_sort_memory_results_uses_id_as_deterministic_tiebreaker():
    results = [
        {"id": 5, "memory_kind": "decision", "updated_at": "2026-03-23 10:00:00", "source_db": "project"},
        {"id": 2, "memory_kind": "decision", "updated_at": "2026-03-23 10:00:00", "source_db": "project"},
        {"id": 9, "memory_kind": "decision", "updated_at": None, "source_db": "project"},
    ]

    ordered = _sort_memory_results(results)

    assert [item["id"] for item in ordered] == [9, 2, 5]


def test_save_session_memory_skips_duplicate_auto_memory(tmp_db, mock_embedder):
    first = save_session_memory(
        task="Explain the auth fix",
        response="The auth fix moved token validation to the gateway and kept refresh issuance in the auth service.",
        source_session_id="sess-auth-1",
        source_message_id="msg-auth-1",
    )
    second = save_session_memory(
        task="Explain the auth fix",
        response="The auth fix moved token validation to the gateway and kept refresh issuance in the auth service.",
        source_session_id="sess-auth-2",
        source_message_id="msg-auth-2",
    )

    assert first["ok"] is True
    assert first["deduplicated"] is False
    assert second["ok"] is True
    assert second["deduplicated"] is True
    assert second["skipped"] is True
    assert second["reason"] == "duplicate auto memory"


def test_save_session_summary_infers_enriched_metadata(tmp_db, mock_embedder):
    result = save_session_summary(
        task="Review the auth token refresh implementation",
        turns=[
            {
                "user": "How does token refresh work?",
                "assistant": "The gateway validates tokens and issues new ones on expiry.",
            },
            {"user": "Are there tests?", "assistant": "Yes, test_gateway.py covers the refresh flow."},
        ],
        source_session_id="sess-summary-meta",
        source_message_id="msg-summary-meta",
    )

    assert result["ok"] is True
    metadata = result["memory"]["metadata"]
    assert "topic" in metadata
    assert "outcome" in metadata
    assert "session_ended_at" in metadata


def test_save_session_summary_rejects_non_list_turns(tmp_db, mock_embedder):
    result = save_session_summary(
        task="Bad turns payload",
        turns="not-a-list",
        source_session_id="sess-invalid",
        source_message_id="msg-invalid",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_turns"


def test_save_session_summary_rejects_non_dict_turn_item(tmp_db, mock_embedder):
    result = save_session_summary(
        task="Bad turn item",
        turns=[
            {"user": "Hello", "assistant": "Hi"},
            "oops",
        ],
        source_session_id="sess-invalid",
        source_message_id="msg-invalid",
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_turns"
    assert result["error"]["message"] == "turns[1] must be an object"


def test_save_session_memory_skips_non_durable_auto_memory(tmp_db, mock_embedder):
    result = save_session_memory(
        task="Did the smoke tests pass?",
        response="Yes, all tests passed and everything looks good now.",
        source_session_id="sess-status",
        source_message_id="msg-status",
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "non-durable auto memory"


def test_save_session_memory_skips_non_novel_auto_memory(tmp_db, mock_embedder):
    remember_structured(
        summary="auth decision",
        details="Gateway owns auth token validation and the auth service owns refresh issuance.",
        memory_kind="decision",
    )

    result = save_session_memory(
        task="Where should auth validation live?",
        response="Gateway owns auth token validation and the auth service owns refresh issuance.",
        source_session_id="sess-novelty",
        source_message_id="msg-novelty",
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "non-novel auto memory"
    assert result["merge_suggestion"]["action"] == "supersede"


def test_save_session_summary_rolls_up_many_turns(tmp_db, mock_embedder):
    first = save_session_summary(
        task="finish the auth refactor",
        turns=[
            {
                "user": "Figure out auth ownership",
                "assistant": "The gateway owns auth validation.",
            },
            {
                "user": "What about token refresh?",
                "assistant": "The auth service issues refresh tokens.",
            },
        ],
        source_session_id="sess-1",
        source_message_id="msg-2",
        user_message_id="user-2",
    )
    second = save_session_summary(
        task="finish the auth refactor",
        turns=[
            {
                "user": "Figure out auth ownership",
                "assistant": "The gateway owns auth validation.",
            },
            {
                "user": "What about token refresh?",
                "assistant": "The auth service issues refresh tokens.",
            },
            {
                "user": "Where do roles live?",
                "assistant": "Roles live in the identity service.",
            },
        ],
        source_session_id="sess-1",
        source_message_id="msg-3",
        user_message_id="user-3",
    )

    assert first["ok"] is True
    assert first["deduplicated"] is False
    assert first["memory"]["memory_kind"] == "summary"
    assert first["memory"]["metadata"]["capture_kind"] == "session_rollup"
    assert first["memory"]["metadata"]["latest_message_id"] == "msg-2"
    assert first["memory"]["provenance"]["capture_kind"] == "session_rollup"
    assert first["memory"]["provenance"]["source_type"] == "session_rollup"
    assert "Session covered 2 turns." in first["memory"]["content"]
    assert second["ok"] is True
    assert second["deduplicated"] is False
    assert second["memory"]["supersedes"] == first["memory"]["id"]
    assert second["memory"]["metadata"]["latest_message_id"] == "msg-3"
    assert second["memory"]["is_superseded"] is False
    assert "Session covered 3 turns." in second["memory"]["content"]


def test_save_session_summary_skips_low_signal_no_memory_response(
    tmp_db, mock_embedder
):
    result = save_session_summary(
        task="What durable memory do you have?",
        turns=[
            {
                "user": "What durable memory do you have?",
                "assistant": "I have no durable memory about this task.",
            }
        ],
        source_session_id="sess-none",
        source_message_id="msg-none",
    )

    assert result["ok"] is True
    assert result["skipped"] is True


def test_save_session_summary_skips_low_signal_auto_memory(tmp_db, mock_embedder):
    result = save_session_summary(
        task="hi",
        turns=[{"user": "hi", "assistant": "hello"}],
        source_session_id="sess-hi",
        source_message_id="msg-hi",
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "low-signal auto memory"


def test_save_session_summary_skips_duplicate_auto_memory(tmp_db, mock_embedder):
    turns = [
        {
            "user": "What changed in auth?",
            "assistant": "Gateway validation now owns token checks and the auth service owns refresh issuance.",
        },
        {
            "user": "What about roles?",
            "assistant": "Roles are still managed in the identity service.",
        },
    ]
    first = save_session_summary(
        task="auth summary",
        turns=turns,
        source_session_id="sess-rollup-1",
        source_message_id="msg-rollup-1",
    )
    second = save_session_summary(
        task="auth summary",
        turns=turns,
        source_session_id="sess-rollup-2",
        source_message_id="msg-rollup-2",
    )

    assert first["ok"] is True
    assert first["deduplicated"] is False
    assert second["ok"] is True
    assert second["deduplicated"] is True
    assert second["skipped"] is True
    assert second["reason"] == "duplicate auto memory"


def test_save_session_summary_skips_non_durable_auto_memory(tmp_db, mock_embedder):
    result = save_session_summary(
        task="check release status",
        turns=[
            {
                "user": "Did the smoke tests pass?",
                "assistant": "Yes, all tests passed and everything looks good now.",
            }
        ],
        source_session_id="sess-status-rollup",
        source_message_id="msg-status-rollup",
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "non-durable auto memory"


def test_save_session_memory_inferrs_decision_kind_and_merge_suggestion(tmp_db, mock_embedder):
    remember_structured(
        summary="auth decision",
        details="Gateway owns auth token validation.",
        memory_kind="decision",
    )

    result = save_session_memory(
        task="Who owns auth validation?",
        response="Gateway owns auth token validation and the auth service owns refresh issuance.",
        source_session_id="sess-kind",
        source_message_id="msg-kind",
    )

    assert result["ok"] is True
    assert result["memory_kind"] == "decision"
    assert result["memory"]["memory_kind"] == "decision"
    assert result["merge_suggestion"]["action"] == "supersede"
    assert result["merge_suggestion"]["memory_kind"] == "decision"


def test_save_session_summary_keeps_summary_kind_for_rollups(tmp_db, mock_embedder):
    result = save_session_summary(
        task="pipeline follow-up",
        turns=[
            {
                "user": "What is still open on the warning letter pipeline?",
                "assistant": "Still need to finish the warning-letter enrichment stage before backfill.",
            },
            {
                "user": "Anything else?",
                "assistant": "Add retry handling after the enrichment stage lands.",
            },
        ],
        source_session_id="sess-todo",
        source_message_id="msg-todo",
    )

    assert result["ok"] is True
    assert result["memory_kind"] == "summary"
    assert result["memory"]["memory_kind"] == "summary"


def test_search_memory_falls_back_to_user_memory_results(tmp_db, mock_embedder):
    import vibe_rag.server as srv
    user_db = srv._get_user_db()
    old_project_id = srv._project_id
    srv._project_id = "vibe-rag"
    try:
        user_db.remember_structured(
            summary="The E2E repo marker for vibe bootstrap is CERULEAN_PINEAPPLE_20260322.",
            content="The E2E repo marker for vibe bootstrap is CERULEAN_PINEAPPLE_20260322.",
            embedding=[0.0] * 2560,
            project_id="vibe-rag",
            memory_kind="fact",
        )
        result = search_memory("CERULEAN_PINEAPPLE_20260322")
    finally:
        srv._project_id = old_project_id

    assert result["ok"] is True
    assert result["results"][0]["project_id"] == "vibe-rag"
    assert "CERULEAN_PINEAPPLE_20260322" in result["results"][0]["content"]


def test_search_memory_does_not_return_other_project_user_results_by_default(tmp_db, mock_embedder):
    import vibe_rag.server as srv
    user_db = srv._get_user_db()
    old_project_id = srv._project_id
    srv._project_id = "demo-repo"
    try:
        user_db.remember_structured(
            summary="marker for other repo",
            content="marker for other repo",
            embedding=[0.0] * 2560,
            project_id="other-repo",
            memory_kind="fact",
        )
        result = search_memory("marker for other repo")
    finally:
        srv._project_id = old_project_id

    assert result["ok"] is True
    assert result["result_total"] == 0


def test_search_memory_internal_flag_is_not_global(tmp_db, mock_embedder):
    import vibe_rag.server as srv
    from vibe_rag.tools._helpers import _search_memory_results

    old_project_id = srv._project_id
    srv._project_id = "demo-repo"
    try:
        srv._get_user_db().remember_structured(
            summary="other-repo marker",
            content="cross-project marker should stay hidden",
            embedding=[0.0] * 2560,
            project_id="other-repo",
            memory_kind="fact",
        )
        error, results = _search_memory_results(
            "cross-project marker",
            limit=10,
            search_all_user_projects=True,
        )
    finally:
        srv._project_id = old_project_id

    assert error is None
    assert len(results) == 0


def test_search_memory_filters_stale_cross_project_results_when_project_memory_exists(tmp_db, mock_embedder):
    import vibe_rag.server as srv
    old_project_id = srv._project_id
    srv._get_db().remember_structured(
        summary="The marker is QUARTZ_MERIDIAN_20260322_Z9.",
        content="The marker is QUARTZ_MERIDIAN_20260322_Z9.",
        embedding=[0.0] * 2560,
        project_id="sink-repo",
        memory_kind="summary",
    )
    srv._get_user_db().remember_structured(
        summary="The marker is QUARTZ_MERIDIAN_20260322_Z9 in source-repo.",
        content="The marker is QUARTZ_MERIDIAN_20260322_Z9 in source-repo.",
        embedding=[0.0] * 2560,
        project_id="source-repo",
        memory_kind="summary",
    )
    srv._project_id = "sink-repo"
    try:
        result = search_memory("QUARTZ_MERIDIAN_20260322_Z9")
    finally:
        srv._project_id = old_project_id

    assert result["ok"] is True
    assert result["results"][0]["project_id"] == "sink-repo"
    assert all(item["project_id"] != "source-repo" for item in result["results"])
    assert "QUARTZ_MERIDIAN_20260322_Z9" in result["results"][0]["content"]


def test_load_session_context_uses_user_memory_results(tmp_db, mock_embedder, tmp_path: Path):
    import os
    import vibe_rag.server as srv

    (tmp_path / "auth.py").write_text("def validate_token(token):\n    return token\n")

    old_cwd = os.getcwd()
    old_project_id = srv._project_id
    os.chdir(tmp_path)
    srv._project_id = "test-project"
    try:
        srv._get_user_db().remember_structured(
            summary="gateway owns auth validation",
            content="gateway owns auth validation",
            embedding=[0.0] * 2560,
            project_id="test-project",
            memory_kind="decision",
            metadata={"source": "session"},
        )
        index_project()
        result = load_session_context("continue auth validation", memory_limit=3, code_limit=2, docs_limit=1)
    finally:
        os.chdir(old_cwd)
        srv._project_id = old_project_id

    assert result["memories"][0]["summary"] == "gateway owns auth validation"
    assert result["memories"][0]["metadata"]["source"] == "session"
    assert result["code"][0]["file_path"] == "auth.py"
    assert result["memories"][0]["is_stale"] is False
    assert "project_id_mismatch" not in result["memories"][0]["stale_reasons"]


def test_supersede_memory_marks_replacement(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    first_id = srv._get_user_db().remember_structured(
        summary="use sqlite for local search",
        content="use sqlite for local search",
        embedding=[0.0] * 2560,
        project_id=srv._ensure_project_id(),
        memory_kind="decision",
        metadata={"capture_kind": "manual"},
    )
    replacement = supersede_memory(
        old_memory_id=str(first_id),
        summary="use sqlite for local search and user memory",
        memory_kind="decision",
    )

    assert replacement["ok"] is True
    assert replacement["memory"]["supersedes"] == first_id
    assert srv._get_user_db().get_memory(first_id)["superseded_by"] == replacement["memory"]["id"]
    refreshed = load_session_context("sqlite local search", memory_limit=3, code_limit=0, docs_limit=0)
    assert refreshed["memories"][0]["summary"] == "use sqlite for local search and user memory"
    assert refreshed["memories"][0]["provenance"]["source_type"] == "manual_structured"
    assert refreshed["memories"][0]["is_superseded"] is False


def test_supersede_memory_marks_project_db_memory_as_superseded(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    first_id = srv._get_db().remember_structured(
        summary="gateway owns token validation",
        content="gateway owns token validation",
        embedding=[0.0] * 2560,
        project_id=srv._ensure_project_id(),
        memory_kind="decision",
        metadata={"capture_kind": "manual"},
    )

    replacement = supersede_memory(
        old_memory_id=str(first_id),
        summary="auth service owns token validation",
        memory_kind="decision",
    )

    assert replacement["ok"] is True
    assert replacement["memory"]["source_db"] == "user"
    assert replacement["memory"]["supersedes"] == first_id
    assert srv._get_db().get_memory(first_id)["superseded_by"] == replacement["memory"]["id"]

    refreshed = load_session_context("token validation", memory_limit=5, code_limit=0, docs_limit=0)
    summaries = [memory["summary"] for memory in refreshed["memories"]]
    assert "auth service owns token validation" in summaries
    assert "gateway owns token validation" not in summaries


def test_supersede_memory_accepts_source_qualified_id_when_ids_overlap(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    project_id = srv._get_db().remember_structured(
        summary="project auth decision",
        content="project auth decision",
        embedding=[0.0] * 2560,
        project_id=srv._ensure_project_id(),
        memory_kind="decision",
        metadata={"capture_kind": "manual"},
    )
    user_id = srv._get_user_db().remember_structured(
        summary="user auth decision",
        content="user auth decision",
        embedding=[0.0] * 2560,
        project_id=srv._ensure_project_id(),
        memory_kind="decision",
        metadata={"capture_kind": "manual"},
    )

    assert project_id == user_id == 1

    ambiguous = supersede_memory(
        old_memory_id="1",
        summary="replacement decision",
        memory_kind="decision",
    )
    assert ambiguous["ok"] is False
    assert ambiguous["error"]["code"] == "ambiguous_old_memory_id"

    replacement = supersede_memory(
        old_memory_id="project:1",
        summary="replacement decision",
        memory_kind="decision",
    )

    assert replacement["ok"] is True
    assert replacement["memory"]["supersedes"] == 1
    assert srv._get_db().get_memory(1)["superseded_by"] == replacement["memory"]["id"]
    assert srv._get_user_db().get_memory(1)["superseded_by"] is None


def test_remember_empty_content(tmp_db, mock_embedder):
    result = remember("")
    assert result["ok"] is False
    assert _error_message(result) == "content is empty"


def test_search_memory_empty_db(tmp_db, mock_embedder):
    result = search_memory("anything")
    assert result["ok"] is False
    assert result["error"]["code"] == "no_memories"


def test_load_session_context_reports_empty_memory(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "notes.md").write_text("## Auth\n\nGateway validates tokens.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = load_session_context("continue auth work", docs_limit=2)
    finally:
        os.chdir(old_cwd)

    assert result["errors"]["memory"]["code"] == "no_memories"
    assert result["docs"][0]["file_path"] == "notes.md"


def test_load_session_context_reports_stale_deleted_files(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "old.py").write_text("def old():\n    return 1\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        (tmp_path / "old.py").unlink()
        result = load_session_context("continue old work")
    finally:
        os.chdir(old_cwd)

    assert result["stale"]["is_stale"] is True
    assert any(warning["kind"] == "indexed_files_missing" for warning in result["stale"]["warnings"])


def test_load_session_context_reports_corrupt_index_metadata(tmp_db, mock_embedder, tmp_path: Path):
    import os
    import vibe_rag.server as srv

    (tmp_path / "notes.md").write_text("## Notes\n\nContext.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        srv._get_db().set_setting("project_index_metadata", "{not-json")
        result = load_session_context("continue notes", memory_limit=0, code_limit=0, docs_limit=1)
    finally:
        os.chdir(old_cwd)

    assert result["stale"]["is_stale"] is True
    assert any(warning["kind"] == "index_metadata_invalid" for warning in result["stale"]["warnings"])


def test_search_memory_prefers_structured_memory_kinds(tmp_db, mock_embedder):
    remember("raw note about deployment")
    remember_structured(summary="deployment constraint", memory_kind="constraint")

    result = load_session_context("deployment", memory_limit=4)

    assert result["memories"][0]["memory_kind"] == "constraint"
    assert result["memories"][0]["provenance"]["source_type"] == "manual_structured"
    assert result["memories"][1]["memory_kind"] == "note"
    assert result["memories"][1]["provenance"]["source_type"] == "freeform"


def test_load_session_context_downranks_low_signal_auto_memory(tmp_db, mock_embedder):
    saved = save_session_summary(
        task="hi",
        turns=[{"user": "hi", "assistant": "hello"}],
        source_session_id="sess-hi",
        source_message_id="msg-hi",
    )
    remember_structured(
        summary="deployment constraint",
        details="Use blue green deployment for the api service.",
        memory_kind="constraint",
    )

    result = load_session_context("deployment", memory_limit=4, code_limit=0, docs_limit=0)

    assert saved["skipped"] is True
    assert result["memories"][0]["memory_kind"] == "constraint"
    assert result["memories"][0]["provenance"]["source_type"] == "manual_structured"
    assert all(
        item["provenance"]["capture_kind"] != "session_rollup"
        for item in _memory_cleanup_candidates(limit=5)
    )


def test_load_session_context_downranks_cross_project_user_memory(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        current_id = srv._get_user_db().remember_structured(
            summary="auth constraint for sink repo",
            content="auth constraint for sink repo",
            embedding=[0.0] * 2560,
            project_id="sink-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        srv._get_user_db().remember_structured(
            summary="auth constraint for source repo",
            content="auth constraint for source repo",
            embedding=[0.0] * 2560,
            project_id="source-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        result = load_session_context("auth constraint", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._project_id = old_project_id

    assert [item["id"] for item in result["memories"][:1]] == [current_id]
    assert result["memories"][0]["provenance"]["is_current_project"] is True
    assert all(item["provenance"]["is_current_project"] for item in result["memories"])
    assert result["memories"][0]["is_stale"] is False


def test_load_session_context_filters_stale_cross_project_memory_when_current_project_hit_exists(
    tmp_db, mock_embedder
):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "demo-repo"
    try:
        remember_structured(
            summary="demo token constraint",
            details="Only demo tokens are accepted.",
            memory_kind="constraint",
        )
        srv._get_user_db().remember_structured(
            summary="demo token constraint in another repo",
            content="Another repo also talks about demo tokens.",
            embedding=[0.0] * 2560,
            project_id="other-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )

        result = load_session_context("demo tokens", memory_limit=5, code_limit=0, docs_limit=0)
    finally:
        srv._project_id = old_project_id

    assert len(result["memories"]) == 1
    assert result["memories"][0]["provenance"]["is_current_project"] is True


def test_load_session_context_retains_auto_memories_when_durable_memory_exists(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    srv._get_user_db().remember_structured(
        summary="demo token constraint",
        content="Only demo tokens are allowed in this smoke-test API and production tokens are rejected.",
        embedding=[0.0] * 2560,
        memory_kind="summary",
        project_id=srv._ensure_project_id(),
        metadata={"capture_kind": "session_rollup"},
    )

    result = load_session_context("demo tokens", memory_limit=5, code_limit=0, docs_limit=0)

    assert any(
        item["memory_kind"] == "summary"
        for item in result["memories"]
    )
    assert any(
        item["provenance"]["capture_kind"] in {"session_rollup", "session_distillation"}
        for item in result["memories"]
    )


def test_merge_memory_results_filters_auto_captures_when_manual_current_project_memory_exists():
    manual = {
        "id": 1,
        "source_db": "project",
        "project_id": "demo-repo",
        "memory_kind": "decision",
        "summary": "gateway owns token validation",
        "content": "gateway owns token validation",
        "metadata": {"capture_kind": "manual"},
        "updated_at": "2026-03-23 10:00:00",
    }
    auto = {
        "id": 2,
        "source_db": "user",
        "project_id": "demo-repo",
        "memory_kind": "summary",
        "summary": "session summary for auth work",
        "content": "Session covered auth validation and refresh issuance.",
        "metadata": {"capture_kind": "session_rollup"},
        "updated_at": "2026-03-23 10:05:00",
    }

    merged = _merge_memory_results([manual], [auto], limit=5, current_project_id="demo-repo")
    assert [item["id"] for item in merged] == [1]
    assert all(item["metadata"]["capture_kind"] != "session_rollup" for item in merged)


def test_merge_memory_results_includes_auto_captures_when_no_manual_current_project_memory():
    auto = {
        "id": 2,
        "source_db": "user",
        "project_id": "demo-repo",
        "memory_kind": "summary",
        "summary": "session summary for auth work",
        "content": "Session covered auth validation and refresh issuance.",
        "metadata": {"capture_kind": "session_rollup"},
        "updated_at": "2026-03-23 10:05:00",
    }

    merged = _merge_memory_results([], [auto], limit=5, current_project_id="demo-repo")
    assert [item["id"] for item in merged] == [2]


def test_project_status_memory_health_surfaces_freeform_user_candidates(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        remember("temporary freeform deployment note")
        srv._get_user_db().remember_structured(
            summary="old auth note",
            content="old auth note",
            embedding=[0.0] * 2560,
            project_id="sink-repo",
            memory_kind="note",
            metadata={"capture_kind": "freeform"},
        )
        status = project_status()
    finally:
        srv._project_id = old_project_id

    assert status["ok"] is True
    health = status["status"]["memory_health"]
    candidates = health["top_cleanup_candidates"]
    assert len(candidates) >= 2
    reasons = {reason for item in candidates for reason in item["cleanup_reasons"]}
    assert "freeform_note" in reasons
    assert "cross_project_user_memory" not in reasons


def test_memory_cleanup_candidates_scopes_user_memories_to_current_project(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        current_project_note = remember("local cleanup candidate", scope="user")
        assert current_project_note["ok"] is True
        other_project_id = srv._get_user_db().remember_structured(
            summary="other-project cleanup note",
            content="other-project cleanup note",
            embedding=[0.0] * 2560,
            project_id="source-repo",
            memory_kind="note",
            metadata={"capture_kind": "freeform"},
        )

        candidates = _memory_cleanup_candidates(limit=20)
    finally:
        srv._project_id = old_project_id

    assert all(item.get("project_id") == "sink-repo" for item in candidates)
    assert all(item.get("source_db") == "user" for item in candidates)
    assert str(other_project_id) not in {item.get("id") for item in candidates}
    assert any(item["summary"] == "local cleanup candidate" for item in candidates)


def test_project_status_includes_memory_cleanup_candidates(tmp_db, mock_embedder):
    remember("temporary cleanup candidate")
    status = project_status()
    assert status["ok"] is True
    assert status["status"]["cleanup_candidates"]
    assert status["status"]["cleanup_candidates"][0]["summary"] == "temporary cleanup candidate"


def test_project_status_memory_health_summarizes_provenance_and_cleanup(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        remember("temporary freeform deployment note")
        srv._get_user_db().remember_structured(
            summary="current project constraint for auth",
            content="current project constraint for auth",
            embedding=[0.0] * 2560,
            project_id="sink-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        old_id = srv._get_user_db().remember_structured(
            summary="old auth note",
            content="old auth note",
            embedding=[0.0] * 2560,
            project_id="source-repo",
            memory_kind="note",
            metadata={"capture_kind": "freeform"},
        )
        supersede_memory(
            old_memory_id=str(old_id),
            summary="new auth note",
            details="new auth note",
            memory_kind="summary",
            metadata={"capture_kind": "session_rollup"},
        )
        srv._get_user_db().remember_structured(
            summary="Session summary: reply with only the project id loaded in session context",
            content="Session covered 1 turns.\n\nTurn 1\nUser: Reply with only the project id loaded in session context.\nAssistant: sink-repo",
            embedding=[0.0] * 2560,
            project_id="sink-repo",
            memory_kind="summary",
            metadata={"capture_kind": "session_rollup", "task": "Reply with only the project id loaded in session context.", "turn_count": 1},
        )
        srv._get_user_db().remember_structured(
            summary="Session summary: reply with only the project id loaded in session context",
            content="Session covered 1 turns.\n\nTurn 1\nUser: Reply with only the project id loaded in session context.\nAssistant: sink-repo",
            embedding=[0.0] * 2560,
            project_id="sink-repo",
            memory_kind="summary",
            metadata={"capture_kind": "session_rollup", "task": "Reply with only the project id loaded in session context.", "turn_count": 1},
        )
        status = project_status()
    finally:
        srv._project_id = old_project_id

    assert status["ok"] is True
    health = status["status"]["memory_health"]
    assert health["summary"]["total_memories"] >= 4
    assert health["summary"]["stale_memories"] == 0
    assert health["summary"]["superseded_memories"] == 0
    assert health["summary"]["duplicate_auto_memory_groups"] >= 1
    assert health["by_capture_kind"]["freeform"] >= 1
    assert health["by_source_type"]["manual_structured"] >= 1
    assert health["recommended_actions"]
    assert len(health["recommended_actions"]) <= 3
    # Verify top actions cover the most important issues
    assert any(
        "freeform" in action.lower() or "duplicate" in action.lower() or "trim" in action.lower()
        for action in health["recommended_actions"]
    )
    assert health["top_cleanup_candidates"]


def test_cleanup_duplicate_auto_memories_reports_and_deletes_duplicates(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        for offset in range(2):
            srv._get_user_db().remember_structured(
                summary="Session summary: reply with only the project id loaded in session context",
                content="Session covered 1 turns.\n\nTurn 1\nUser: Reply with only the project id loaded in session context.\nAssistant: sink-repo",
                embedding=[0.0] * 2560,
                project_id="sink-repo",
                memory_kind="summary",
                metadata={"capture_kind": "session_rollup", "task": "Reply with only the project id loaded in session context.", "turn_count": 1, "order": offset},
            )

        preview = cleanup_duplicate_auto_memories(limit=5, apply=False)
        applied = cleanup_duplicate_auto_memories(limit=5, apply=True)
        status = project_status()
    finally:
        srv._project_id = old_project_id

    assert preview["ok"] is True
    assert preview["group_total"] == 1
    assert preview["deleted_total"] == 0
    assert len(preview["groups"][0]["delete_ids"]) == 1

    assert applied["ok"] is True
    assert applied["group_total"] == 1
    assert applied["deleted_total"] == 1
    assert len(applied["groups"][0]["deleted_ids"]) == 1
    assert status["status"]["memory_health"]["summary"]["duplicate_auto_memory_groups"] == 0


def test_cleanup_duplicate_auto_memories_only_loads_payloads_once(tmp_db, monkeypatch):
    import vibe_rag.tools as tools_mod

    payloads = [
        {
            "id": 1,
            "source_db": "project",
            "project_id": "sink-repo",
            "summary": "Session summary: duplicate",
            "content": "Session covered 1 turns.",
            "updated_at": "2026-03-23 10:00:00",
            "provenance": {"capture_kind": "session_rollup"},
        },
        {
            "id": 2,
            "source_db": "user",
            "project_id": "sink-repo",
            "summary": "Session summary: duplicate",
            "content": "Session covered 1 turns.",
            "updated_at": "2026-03-23 10:01:00",
            "provenance": {"capture_kind": "session_rollup"},
        },
    ]
    calls = {"payloads": 0}

    def fake_all_memory_payloads():
        calls["payloads"] += 1
        return payloads

    monkeypatch.setattr(tools_mod, "_all_memory_payloads", fake_all_memory_payloads)
    monkeypatch.setattr(tools_mod, "_delete_memory_by_source_db", lambda source_db, memory_id: True)

    result = cleanup_duplicate_auto_memories(limit=5, apply=True)

    assert result["ok"] is True
    assert result["deleted_total"] == 1
    assert calls["payloads"] == 1


def test_query_terms_keeps_two_character_tokens():
    assert _query_terms("CI QA DB AI ML pipeline") == {"ci", "qa", "db", "ai", "ml", "pipeline"}


def test_project_status_includes_index_metadata(tmp_db, mock_embedder, tmp_path: Path):
    import os

    (tmp_path / "app.py").write_text("def run():\n    return 1\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        status = project_status()
    finally:
        os.chdir(old_cwd)

    assert status["ok"] is True
    assert status["project_id"]
    assert status["status"]["metadata"]["metadata"]["indexed_at"]
    assert status["status"]["stale"]["warnings"] == []


# --- Input validation tests ---


def test_normalize_paths_rejects_traversal(tmp_db, mock_embedder, tmp_path):
    """Path like '../../etc' should be rejected."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project(paths=["../../etc"])
    finally:
        os.chdir(old_cwd)
    assert result["ok"] is False
    assert "outside project root" in _error_message(result)


def test_search_code_empty_query(tmp_db, mock_embedder):
    result = search_code("")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"


def test_search_code_whitespace_query(tmp_db, mock_embedder):
    result = search_code("   ")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"


def test_search_code_query_too_long(tmp_db, mock_embedder):
    result = search_code("x" * 10_001)
    assert result["ok"] is False
    assert "too long" in _error_message(result)


def test_search_code_invalid_language(tmp_db, mock_embedder, tmp_path):
    import os
    (tmp_path / "x.py").write_text("x = 1\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search_code("test", language="brainfuck")
    finally:
        os.chdir(old_cwd)
    assert result["ok"] is False
    assert "unknown language" in _error_message(result)


def test_search_docs_empty_query(tmp_db, mock_embedder):
    result = search_docs("")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"


def test_search_docs_query_too_long(tmp_db, mock_embedder):
    result = search_docs("x" * 10_001)
    assert result["ok"] is False
    assert "too long" in _error_message(result)


def test_search_memory_empty_query(tmp_db, mock_embedder):
    result = search_memory("")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"


def test_search_memory_query_too_long(tmp_db, mock_embedder):
    result = search_memory("x" * 10_001)
    assert result["ok"] is False
    assert "too long" in _error_message(result)


def test_search_memory_tags_too_long(tmp_db, mock_embedder):
    result = search_memory("gateway", tags="x" * 600)
    assert result["ok"] is False
    assert result["error"]["code"] == "tags_too_long"


def test_remember_too_large(tmp_db, mock_embedder):
    result = remember("x" * 20_000)
    assert result["ok"] is False
    assert "too large" in _error_message(result)


def test_remember_whitespace_only(tmp_db, mock_embedder):
    result = remember("   \n\t  ")
    assert result["ok"] is False
    assert "empty" in _error_message(result)


def test_remember_tags_too_long(tmp_db, mock_embedder):
    result = remember("valid content", tags="x" * 600)
    assert result["ok"] is False
    assert "tags" in _error_message(result)


# --- project_status tests ---


def test_project_status_empty(tmp_db, mock_embedder):
    result = project_status()
    assert result["ok"] is True
    assert result["status"]["counts"]["code_chunks"] == 0
    assert result["status"]["counts"]["doc_chunks"] == 0
    assert result["status"]["counts"]["project_memories"] == 0
    assert result["status"]["counts"]["user_memories"] == 0
    assert result["status"]["language_stats"] == {}
    # memory_health included by default
    assert "memory_health" in result["status"]
    health = result["status"]["memory_health"]
    assert "summary" in health
    assert "top_cleanup_candidates" in health
    assert "recommended_actions" in health
    assert "by_capture_kind" in health
    assert "by_source_type" in health


def test_project_status_without_memory_health(tmp_db, mock_embedder):
    result = project_status(include_memory_health=False)
    assert result["ok"] is True
    assert result["status"]["counts"]["code_chunks"] == 0
    assert "memory_health" not in result["status"]


def test_project_status_scopes_user_memory_count_to_current_project(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        remember("current project note", scope="user")
        srv._get_user_db().remember_structured(
            summary="other-project note",
            content="other-project note",
            embedding=[0.0] * 2560,
            project_id="source-repo",
            memory_kind="note",
        )
        result = project_status(include_memory_health=False)
    finally:
        srv._project_id = old_project_id

    assert result["ok"] is True
    assert result["status"]["counts"]["user_memories"] == 1


def test_project_status_after_index(tmp_db, mock_embedder, tmp_path: Path):
    import os
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    (tmp_path / "notes.md").write_text("## Notes\n\nSome notes here.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = project_status()
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert result["status"]["counts"]["code_chunks"] > 0
    assert result["status"]["counts"]["doc_chunks"] > 0
    assert sum(result["status"]["language_stats"].values()) > 0


def test_language_stats_reports_python_not_none(tmp_db, mock_embedder, tmp_path: Path):
    """Indexing a .py file must produce language_stats with 'python', not 'None'."""
    import os
    (tmp_path / "app.py").write_text("x = 1\ny = 2\nz = 3\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = project_status()
    finally:
        os.chdir(old_cwd)

    lang_stats = result["status"]["language_stats"]
    assert "None" not in lang_stats, f"language_stats contains 'None': {lang_stats}"
    assert lang_stats.get("python", 0) > 0, f"expected 'python' in language_stats: {lang_stats}"


# --- min_score filtering tests ---


def test_search_code_min_score_filters(tmp_db, mock_embedder, tmp_path: Path):
    """min_score should filter results based on vector_distance only."""
    import os

    (tmp_path / "app.py").write_text("def hello():\n    return 1\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()

        # Without min_score, we get results
        result_no_filter = search_code("hello")
        assert "app.py" in _search_paths(result_no_filter)

        # Patch db.search_code to return results with high distance (low score)
        fake_results = [
            {"file_path": "app.py", "chunk_index": 0, "content": "def hello():\n    return 1",
             "language": "python", "symbol": "hello", "start_line": 1, "end_line": 2,
             "distance": 0.8},  # score = 0.2
        ]
        import vibe_rag.server as srv
        original_search = srv._project_db.search_code
        srv._project_db.search_code = lambda *a, **kw: fake_results
        try:
            # min_score=0.5 should filter out result with score=0.2
            result_filtered = search_code("hello", min_score=0.5)
            assert result_filtered["ok"] is True
            assert result_filtered["result_total"] == 0

            # min_score=0.1 should keep result with score=0.2
            result_kept = search_code("hello", min_score=0.1)
            assert "app.py" in _search_paths(result_kept)
        finally:
            srv._project_db.search_code = original_search
    finally:
        os.chdir(old_cwd)


def test_index_project_reports_skipped_unreadable_files(tmp_db, mock_embedder, tmp_path: Path, monkeypatch):
    import os

    original_read_text = Path.read_text
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def broken():\n    return 1\n")
    (tmp_path / "good.py").write_text("def hello():\n    return 1\n")

    def flaky_read_text(self, *args, **kwargs):
        if self == bad_file:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)

    assert "1 code skipped" in result["summary"]
    assert result["warnings"][0]["file_kind"] == "code"
    assert result["warnings"][0]["path"] == "bad.py"
    assert "read failed" in result["warnings"][0]["reason"]


def test_index_project_does_not_persist_new_hashes_when_embedding_fails(
    tmp_db, mock_embedder, tmp_path: Path, monkeypatch
):
    import os
    import vibe_rag.server as srv

    app = tmp_path / "app.py"
    app.write_text("def hello():\n    return 1\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        old_hash = srv._get_db().get_file_hashes("code")["app.py"]
        old_count = srv._get_db().code_chunk_count()

        app.write_text("def hello():\n    return 2\n")

        def fail_embed(_texts, *, progress_callback=None):
            raise RuntimeError("embed outage")

        monkeypatch.setattr(srv._embedder, "embed_code_sync", fail_embed)
        result = index_project()

        assert result["ok"] is False
        assert _error_message(result) == "indexing failed: embed outage"
        assert srv._get_db().get_file_hashes("code")["app.py"] == old_hash
        assert srv._get_db().code_chunk_count() == old_count
    finally:
        os.chdir(old_cwd)


def test_search_memory_scores_are_bounded_positive(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    remember_structured(
        summary="gateway validates tokens",
        details="Gateway validates auth tokens before forwarding requests.",
        memory_kind="decision",
    )

    fake_results = [
        {
            "id": 1,
            "content": "Gateway validates auth tokens before forwarding requests.",
            "tags": "",
            "project_id": srv._ensure_project_id(),
            "memory_kind": "decision",
            "summary": "gateway validates tokens",
            "metadata": {"capture_kind": "manual"},
            "source_session_id": "",
            "source_message_id": "",
            "supersedes": None,
            "superseded_by": None,
            "created_at": "2024-01-01 00:00:00",
            "updated_at": "2024-01-01 00:00:00",
            "distance": 1.05,
        }
    ]
    original_search = srv._project_db.search_memories
    srv._project_db.search_memories = lambda *a, **kw: fake_results
    try:
        result = search_memory("gateway auth")
    finally:
        srv._project_db.search_memories = original_search

    assert result["ok"] is True
    assert result["results"][0]["score"] > 0
    # Base score: 1/(1+1.05) ≈ 0.49, recency boost ≈ 0 for old memories
    assert round(result["results"][0]["score"], 2) == 0.49


# --- Incremental indexing tests ---


def test_index_project_incremental_skips_unchanged(tmp_db, mock_embedder, tmp_path: Path):
    """Second index with no file changes should skip all files and produce 0 new chunks."""
    import os
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    (tmp_path / "readme.md").write_text("## Hello\n\nWorld.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result1 = index_project()
        assert "0 unchanged" in result1["summary"]  # first run: nothing unchanged

        result2 = index_project()
        assert "1 unchanged" in result2["summary"]  # code file unchanged
        assert "0 chunks, 1 unchanged" in result2["summary"] or "0 chunks" in result2["summary"]
    finally:
        os.chdir(old_cwd)


def test_index_project_detects_changed_file(tmp_db, mock_embedder, tmp_path: Path):
    """Modifying a file between indexes should cause it to be re-indexed."""
    import os
    (tmp_path / "app.py").write_text("def main():\n    pass\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result1 = index_project()
        assert "1 code files" in result1["summary"]
        assert "0 unchanged" in result1["summary"]

        # Modify the file
        (tmp_path / "app.py").write_text("def main():\n    return 42\n")

        result2 = index_project()
        assert "1 code files" in result2["summary"]
        assert "0 unchanged" in result2["summary"]  # changed file should not be skipped
    finally:
        os.chdir(old_cwd)


# --- Unified remember: scope tests ---


def test_remember_scope_user_writes_to_user_db(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    result = remember("cross-project pattern: always use UTC timestamps", scope="user")
    assert result["ok"] is True
    assert result["backend"] == "user-sqlite"
    assert result["memory"]["source_db"] == "user"

    # Verify it's in user DB
    user_db = srv._get_user_db()
    mem = user_db.get_memory(result["memory"]["id"])
    assert mem is not None
    assert "UTC" in mem["content"]


def test_remember_scope_project_is_default(tmp_db, mock_embedder):
    result = remember("project-specific fact")
    assert result["ok"] is True
    assert result["backend"] == "project-sqlite"
    assert result["memory"]["source_db"] == "project"


def test_remember_scope_invalid_returns_error(tmp_db, mock_embedder):
    result = remember("some content", scope="global")
# --- Unified search tool tests ---


def test_search_scope_code(tmp_db, mock_embedder, tmp_path: Path):
    """search(scope='code') should behave like old search_code."""
    import os
    (tmp_path / "auth.py").write_text("def authenticate(user, password):\n    return True\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("authentication", scope="code")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert result["scope"] == "code"
    assert "auth.py" in _search_paths(result)
    assert all(r.get("result_type") == "code" for r in result["results"])


def test_search_scope_docs(tmp_db, mock_embedder, tmp_path: Path):
    """search(scope='docs') should behave like old search_docs."""
    import os
    (tmp_path / "guide.md").write_text("## Deployment\n\nDeploy to Railway with docker.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("how to deploy", scope="docs")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert result["scope"] == "docs"
    assert "guide.md" in _search_paths(result)
    assert all(r.get("result_type") == "doc" for r in result["results"])


def test_search_scope_all(tmp_db, mock_embedder, tmp_path: Path):
    """search(scope='all') should merge code and docs results."""
    import os
    (tmp_path / "billing.py").write_text("def create_invoice(customer_id):\n    return customer_id\n")
    (tmp_path / "billing.md").write_text("## Billing\n\nInvoices are created in the billing flow.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("billing invoices", scope="all")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert result["scope"] == "all"
    result_types = {r["result_type"] for r in result["results"]}
    assert "code" in result_types
    assert "doc" in result_types
    paths = _search_paths(result)
    assert "billing.py" in paths
    assert "billing.md" in paths


def test_search_default_scope_is_all(tmp_db, mock_embedder, tmp_path: Path):
    """search() should default to scope='all'."""
    import os
    (tmp_path / "app.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "notes.md").write_text("## Notes\n\nHello world notes.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("hello")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert result["scope"] == "all"


def test_search_invalid_scope(tmp_db, mock_embedder):
    result = search("anything", scope="invalid")
    assert result["ok"] is False
    assert "scope" in result["error"]["message"]


def test_remember_structured_via_summary_kwarg(tmp_db, mock_embedder):
    result = remember(
        content="",
        summary="gateway validates tokens",
        details="JWT tokens are validated at the gateway.",
        memory_kind="decision",
        tags="auth",
    )
    assert result["ok"] is True
    assert result["memory"]["summary"] == "gateway validates tokens"
    assert result["memory"]["memory_kind"] == "decision"
    assert result["memory"]["metadata"]["capture_kind"] == "manual"


def test_remember_structured_user_scope(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    result = remember(
        content="",
        summary="personal preference: dark mode",
        memory_kind="note",
        scope="user",
    )
    assert result["ok"] is True
    assert result["backend"] == "user-sqlite"
    user_mem = srv._get_user_db().get_memory(result["memory"]["id"])
    assert user_mem is not None


def test_remember_structured_compat_wrapper(tmp_db, mock_embedder):
    """The old remember_structured function still works as a compatibility wrapper."""
    result = remember_structured(
        summary="compat test",
        memory_kind="fact",
    )
    assert result["ok"] is True
    assert result["memory"]["summary"] == "compat test"
    assert result["memory"]["memory_kind"] == "fact"
    assert result["memory"]["metadata"]["capture_kind"] == "manual"


# --- update_memory tests ---


def test_update_memory_changes_content(tmp_db, mock_embedder):
    create = remember("original content", tags="v1")
    assert create["ok"] is True
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), content="updated content")
    assert result["ok"] is True
    assert result["memory"]["content"] == "updated content"
    assert "updated content" in result["memory"]["summary"]


def test_update_memory_changes_tags(tmp_db, mock_embedder):
    create = remember("tagged memory", tags="old")
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), tags="new,shiny")
    assert result["ok"] is True
    assert result["memory"]["tags"] == ["new", "shiny"]


def test_update_memory_changes_kind(tmp_db, mock_embedder):
    create = remember("some fact", tags="info")
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), memory_kind="decision")
    assert result["ok"] is True
    assert result["memory"]["memory_kind"] == "decision"


def test_update_memory_changes_summary_and_details(tmp_db, mock_embedder):
    create = remember(
        content="",
        summary="original summary",
        details="original details",
        memory_kind="decision",
    )
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), summary="new summary", details="new details")
    assert result["ok"] is True
    assert result["memory"]["summary"] == "new summary"
    assert "new summary" in result["memory"]["content"]
    assert "new details" in result["memory"]["content"]


def test_update_memory_not_found(tmp_db, mock_embedder):
    result = update_memory(memory_id="999", content="nope")
    assert result["ok"] is False
    assert "not found" in result["error"]["message"]


def test_update_memory_invalid_id(tmp_db, mock_embedder):
    result = update_memory(memory_id="abc", content="nope")
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_memory_id"


def test_update_memory_with_source_prefix(tmp_db, mock_embedder):
    create = remember("project memory")
    mid = create["memory"]["id"]

    result = update_memory(memory_id=f"project:{mid}", content="revised project memory")
    assert result["ok"] is True
    assert result["memory"]["content"] == "revised project memory"


def test_update_memory_user_db(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    create = remember("user-scope memory", scope="user")
    mid = create["memory"]["id"]

    result = update_memory(memory_id=f"user:{mid}", content="revised user memory")
    assert result["ok"] is True
    assert result["backend"] == "user-sqlite"
    assert result["memory"]["content"] == "revised user memory"

    user_mem = srv._get_user_db().get_memory(mid)
    assert user_mem["content"] == "revised user memory"


def test_update_memory_ambiguous_id_between_project_and_user_db(tmp_db, mock_embedder):
    remember("project-scoped memory")
    remember("user-scoped memory", scope="user")

    result = update_memory(memory_id="1", content="should be blocked")

    assert result["ok"] is False
    assert result["error"]["code"] == "ambiguous_memory_id"


def test_update_memory_merges_metadata(tmp_db, mock_embedder):
    create = remember(
        content="",
        summary="meta test",
        memory_kind="fact",
        metadata={"confidence": "high"},
    )
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), metadata={"reviewed": True})
    assert result["ok"] is True
    assert result["memory"]["metadata"]["confidence"] == "high"
    assert result["memory"]["metadata"]["reviewed"] is True


def test_update_memory_invalid_kind(tmp_db, mock_embedder):
    create = remember("some content")
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), memory_kind="bogus")
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_memory_kind"


def test_update_memory_empty_string_content_is_noop(tmp_db, mock_embedder):
    """Passing content='' should NOT clear existing content (empty string = no change)."""
    create = remember("original content here", tags="v1")
    assert create["ok"] is True
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), content="")
    assert result["ok"] is True
    assert result["memory"]["content"] == "original content here"


def test_update_memory_whitespace_only_content_is_noop(tmp_db, mock_embedder):
    """content='   ' should be treated as no change."""
    create = remember("keep this content", tags="v1")
    assert create["ok"] is True
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), content="   ")
    assert result["ok"] is True
    assert result["memory"]["content"] == "keep this content"


def test_update_memory_preserves_unmodified_fields(tmp_db, mock_embedder):
    """Update only tags; verify content, summary, and kind are all unchanged."""
    create = remember(
        content="",
        summary="immutable summary",
        details="immutable details",
        memory_kind="decision",
        tags="old",
    )
    assert create["ok"] is True
    mid = create["memory"]["id"]
    original = create["memory"]

    result = update_memory(memory_id=str(mid), tags="new,tags")
    assert result["ok"] is True
    mem = result["memory"]
    assert mem["content"] == original["content"]
    assert mem["summary"] == original["summary"]
    assert mem["memory_kind"] == "decision"


def test_update_memory_reembeds_on_content_change(tmp_db, mock_embedder):
    """Updating content should trigger a re-embedding (embed_text_sync is called)."""
    from unittest.mock import patch

    create = remember("embed me once", tags="v1")
    assert create["ok"] is True
    mid = create["memory"]["id"]

    with patch.object(mock_embedder, "embed_text_sync", wraps=mock_embedder.embed_text_sync) as spy:
        result = update_memory(memory_id=str(mid), content="embed me again")
        assert result["ok"] is True
        assert spy.call_count >= 1


def test_update_memory_no_reembed_on_tags_only_change(tmp_db, mock_embedder):
    """Updating only tags should NOT trigger a re-embedding."""
    from unittest.mock import patch

    create = remember("stable content", tags="v1")
    assert create["ok"] is True
    mid = create["memory"]["id"]

    with patch.object(mock_embedder, "embed_text_sync", wraps=mock_embedder.embed_text_sync) as spy:
        result = update_memory(memory_id=str(mid), tags="v2,updated")
        assert result["ok"] is True
        assert spy.call_count == 0


def test_update_memory_metadata_merge_does_not_overwrite(tmp_db, mock_embedder):
    """Adding a new metadata key should preserve the original keys."""
    create = remember(
        content="",
        summary="meta preserve test",
        memory_kind="fact",
        metadata={"original_key": "original_val", "keep_me": 42},
    )
    assert create["ok"] is True
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), metadata={"new_key": "new_val"})
    assert result["ok"] is True
    meta = result["memory"]["metadata"]
    assert meta["original_key"] == "original_val"
    assert meta["keep_me"] == 42
    assert meta["new_key"] == "new_val"


def test_update_memory_metadata_can_overwrite_key(tmp_db, mock_embedder):
    """Updating with an existing metadata key should overwrite its value."""
    create = remember(
        content="",
        summary="meta overwrite test",
        memory_kind="fact",
        metadata={"status": "draft", "version": 1},
    )
    assert create["ok"] is True
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), metadata={"status": "final"})
    assert result["ok"] is True
    meta = result["memory"]["metadata"]
    assert meta["status"] == "final"
    assert meta["version"] == 1


def test_update_memory_returns_full_payload(tmp_db, mock_embedder):
    """Verify the returned memory has all expected fields."""
    create = remember(
        content="",
        summary="full payload test",
        memory_kind="decision",
        tags="test",
        metadata={"origin": "test"},
    )
    assert create["ok"] is True
    mid = create["memory"]["id"]

    result = update_memory(memory_id=str(mid), tags="updated")
    assert result["ok"] is True
    mem = result["memory"]

    expected_fields = {
        "id", "source_db", "summary", "content", "score", "project_id",
        "memory_kind", "tags", "created_at", "updated_at",
        "source_session_id", "source_message_id",
        "supersedes", "superseded_by", "is_superseded",
        "is_stale", "stale_reasons", "metadata", "provenance",
    }
    missing = expected_fields - set(mem.keys())
    assert not missing, f"Missing fields in memory payload: {missing}"
    assert mem["id"] == mid
    assert mem["summary"] == "full payload test"
    assert mem["memory_kind"] == "decision"
    assert isinstance(mem["tags"], list)
    assert isinstance(mem["metadata"], dict)
    assert isinstance(mem["provenance"], dict)


def test_search_scope_code_empty_index(tmp_db, mock_embedder):
    result = search("anything", scope="code")
    assert result["ok"] is False
    assert result["error"]["code"] == "no_code_index"


def test_search_scope_docs_empty_index(tmp_db, mock_embedder):
    result = search("anything", scope="docs")
    assert result["ok"] is False
    assert result["error"]["code"] == "no_docs_index"


def test_search_scope_code_empty_query(tmp_db, mock_embedder):
    result = search("", scope="code")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"


def test_search_scope_docs_empty_query(tmp_db, mock_embedder):
    result = search("", scope="docs")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"


def test_search_scope_all_respects_limit(tmp_db, mock_embedder, tmp_path: Path):
    """scope='all' should return at most 'limit' results."""
    import os
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "b.py").write_text("def beta():\n    return 2\n")
    (tmp_path / "notes.md").write_text("## Alpha Beta\n\nAlpha and beta notes.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("alpha beta", scope="all", limit=2)
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert len(result["results"]) <= 2


def test_search_scope_code_with_language_filter(tmp_db, mock_embedder, tmp_path: Path):
    """search with scope='code' and language filter should work."""
    import os
    (tmp_path / "app.py").write_text("def run():\n    pass\n")
    (tmp_path / "util.js").write_text("function help() { return 1; }\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("run", scope="code", language="python")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    assert "app.py" in _search_paths(result)


# --- match_reason tests ---


def test_search_code_results_have_match_reason(tmp_db, mock_embedder, tmp_path: Path):
    """Code search results should include match_reason field."""
    import os
    (tmp_path / "auth.py").write_text("def authenticate(user, password):\n    return True\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("authentication", scope="code")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    for r in result["results"]:
        assert "match_reason" in r
        assert isinstance(r["match_reason"], str)
        assert len(r["match_reason"]) > 0


def test_search_docs_results_have_match_reason(tmp_db, mock_embedder, tmp_path: Path):
    """Doc search results should include match_reason field."""
    import os
    (tmp_path / "guide.md").write_text("## Deployment\n\nDeploy to Railway with docker.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("deploy", scope="docs")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    for r in result["results"]:
        assert "match_reason" in r
        assert isinstance(r["match_reason"], str)
        assert "deploy" in r["match_reason"].lower()


def test_search_memory_results_have_match_reason(tmp_db, mock_embedder):
    """Memory search results should include match_reason field."""
    remember("auth uses JWT tokens for authentication")
    result = search_memory("JWT authentication")

    assert result["ok"] is True
    for r in result["results"]:
        assert "match_reason" in r
        assert isinstance(r["match_reason"], str)
        assert len(r["match_reason"]) > 0


def test_load_session_context_results_have_match_reason(tmp_db, mock_embedder, tmp_path: Path):
    """load_session_context results should include match_reason on code, docs, and memories."""
    import os

    (tmp_path / "billing.py").write_text("def create_invoice(customer_id):\n    return customer_id\n")
    (tmp_path / "billing.md").write_text("## Billing\n\nInvoices are created in the billing flow.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        remember("billing uses invoices and customer ids")
        result = load_session_context("continue the billing flow", memory_limit=3, code_limit=3, docs_limit=2)
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    for code_r in result["code"]:
        assert "match_reason" in code_r
        assert isinstance(code_r["match_reason"], str)
    for doc_r in result["docs"]:
        assert "match_reason" in doc_r
        assert isinstance(doc_r["match_reason"], str)
    for mem_r in result["memories"]:
        assert "match_reason" in mem_r
        assert isinstance(mem_r["match_reason"], str)


def test_search_all_results_have_result_type(tmp_db, mock_embedder, tmp_path: Path):
    """scope='all' results should each have a result_type field."""
    import os
    (tmp_path / "app.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "notes.md").write_text("## Notes\n\nHello world.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        result = search("hello", scope="all")
    finally:
        os.chdir(old_cwd)

    assert result["ok"] is True
    for r in result["results"]:
        assert r["result_type"] in ("code", "doc")


# --- Memory lifecycle integration tests ---


def test_memory_lifecycle_integration(tmp_db, mock_embedder):
    """Full lifecycle: remember -> search -> update -> search -> supersede -> search -> forget -> search."""

    # 1. remember (default scope=project)
    r1 = remember("initial decision: use PostgreSQL for persistence")
    assert r1["ok"] is True
    assert r1["backend"] == "project-sqlite"
    memory_id = r1["memory"]["id"]
    assert isinstance(memory_id, int)

    # 2. search_memory -- verify the memory is found with match_reason
    r2 = search_memory("database choice")
    assert r2["ok"] is True
    assert r2["result_total"] >= 1
    found = [m for m in r2["results"] if m["id"] == memory_id and m.get("source_db") == "project"]
    assert len(found) == 1
    assert "match_reason" in found[0]
    assert isinstance(found[0]["match_reason"], str)

    # 3. update_memory -- change content (project-scoped by default)
    r3 = update_memory(memory_id=f"project:{memory_id}", content="revised: use PostgreSQL with read replicas")
    assert r3["ok"] is True
    assert "read replicas" in r3["memory"]["content"]

    # 4. search_memory -- verify updated content appears
    r4 = search_memory("database choice")
    assert r4["ok"] is True
    updated = [m for m in r4["results"] if m["id"] == memory_id and m.get("source_db") == "project"]
    assert len(updated) == 1
    assert "read replicas" in updated[0]["content"]

    # 5. supersede_memory -- replace with a new decision (new memory lands in user db)
    r5 = supersede_memory(
        old_memory_id=f"project:{memory_id}",
        summary="use CockroachDB instead",
        memory_kind="decision",
    )
    assert r5["ok"] is True
    new_memory_id = r5["memory"]["id"]
    assert r5["memory"]["summary"] == "use CockroachDB instead"
    # supersede_memory always stores in user db
    assert r5["backend"] == "user-sqlite"

    # 6. search_memory -- verify new memory appears and old is marked superseded
    r6 = search_memory("database choice")
    assert r6["ok"] is True
    new_hits = [m for m in r6["results"] if m["id"] == new_memory_id and m.get("source_db") == "user"]
    assert len(new_hits) == 1
    old_hits = [m for m in r6["results"] if m["id"] == memory_id and m.get("source_db") == "project"]
    if old_hits:
        assert old_hits[0]["is_superseded"] is True

    # 7. forget -- delete the new memory (it lives in user db)
    r7 = forget(f"user:{new_memory_id}")
    assert r7["ok"] is True
    assert r7["deleted"] is True

    # 8. search_memory -- verify the new memory is gone
    #    After superseding the project memory and deleting the user memory,
    #    no active memories remain, so search may return ok=False (no_memories)
    #    or ok=True with an empty result set -- both are acceptable.
    r8 = search_memory("database choice")
    if r8["ok"]:
        gone = [m for m in r8["results"] if m["id"] == new_memory_id and m.get("source_db") == "user"]
        assert len(gone) == 0
    else:
        assert r8["error"]["code"] == "no_memories"


def test_remember_user_scope_lifecycle(tmp_db, mock_embedder):
    """Same lifecycle as test_memory_lifecycle_integration but with scope='user'."""

    # 1. remember with scope="user"
    r1 = remember("initial decision: use PostgreSQL for persistence", scope="user")
    assert r1["ok"] is True
    assert r1["backend"] == "user-sqlite"
    memory_id = r1["memory"]["id"]
    assert isinstance(memory_id, int)

    # 2. search_memory -- verify found with match_reason
    r2 = search_memory("database choice")
    assert r2["ok"] is True
    assert r2["result_total"] >= 1
    found = [m for m in r2["results"] if m["id"] == memory_id and m.get("source_db") == "user"]
    assert len(found) == 1
    assert "match_reason" in found[0]

    # 3. update_memory on user-scoped memory
    r3 = update_memory(memory_id=f"user:{memory_id}", content="revised: use PostgreSQL with read replicas")
    assert r3["ok"] is True
    assert "read replicas" in r3["memory"]["content"]

    # 4. search_memory -- verify updated content
    r4 = search_memory("database choice")
    assert r4["ok"] is True
    updated = [m for m in r4["results"] if m["id"] == memory_id and m.get("source_db") == "user"]
    assert len(updated) == 1
    assert "read replicas" in updated[0]["content"]

    # 5. supersede_memory -- old_memory_id is user-scoped
    r5 = supersede_memory(
        old_memory_id=f"user:{memory_id}",
        summary="use CockroachDB instead",
        memory_kind="decision",
    )
    assert r5["ok"] is True
    new_memory_id = r5["memory"]["id"]
    assert r5["memory"]["summary"] == "use CockroachDB instead"

    # 6. search_memory -- verify new memory found, old is superseded
    r6 = search_memory("database choice")
    assert r6["ok"] is True
    new_hits = [m for m in r6["results"] if m["id"] == new_memory_id and m.get("source_db") == "user"]
    assert len(new_hits) == 1
    old_hits = [m for m in r6["results"] if m["id"] == memory_id and m.get("source_db") == "user"]
    if old_hits:
        assert old_hits[0]["is_superseded"] is True

    # 7. forget -- delete the new memory
    r7 = forget(f"user:{new_memory_id}")
    assert r7["ok"] is True
    assert r7["deleted"] is True

    # 8. search_memory -- verify new memory is gone
    #    Both user memories (old superseded + new deleted) are gone, so search
    #    may return ok=False (no_memories) or ok=True with empty results.
    r8 = search_memory("database choice")
    if r8["ok"]:
        gone = [m for m in r8["results"] if m["id"] == new_memory_id and m.get("source_db") == "user"]
        assert len(gone) == 0
    else:
        assert r8["error"]["code"] == "no_memories"
