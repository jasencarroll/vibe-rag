import pytest
from pathlib import Path

from vibe_rag.tools import (
    _embed_sync_with_progress,
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
    index_project,
    load_session_context,
    memory_cleanup_report,
    memory_quality_report,
    project_status,
    remember,
    remember_structured,
    save_session_memory,
    save_session_summary,
    search_code,
    search_docs,
    search_memory,
    supersede_memory,
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
        embedding=[0.0] * 1024,
        project_id=srv._ensure_project_id(),
        memory_kind="fact",
        metadata={"capture_kind": "manual"},
    )
    user_id = srv._get_user_db().remember_structured(
        summary="user fact",
        content="user fact",
        embedding=[0.0] * 1024,
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
        embedding=[0.0] * 1024,
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
        embedding=[0.0] * 1024,
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
        embedding=[0.0] * 1024,
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
        embedding=[0.0] * 1024,
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

    monkeypatch.setattr(tools_mod, "embedding_provider_status", lambda: {"ok": False, "provider": "ollama", "detail": "not reachable"})
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
    db.remember_structured(summary="gateway owns tokens", content="gateway owns tokens", embedding=[0.0] * 1024, project_id=project_id, memory_kind="decision")
    db.remember_structured(summary="max 100 retries", content="max 100 retries", embedding=[0.0] * 1024, project_id=project_id, memory_kind="constraint")
    db.remember_structured(summary="session note", content="session note", embedding=[0.0] * 1024, project_id=project_id, memory_kind="note")

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
    old_id = db.remember_structured(summary="old decision", content="old decision", embedding=[0.0] * 1024, project_id=project_id, memory_kind="decision")
    db.remember_structured(
        summary="new decision",
        content="new decision",
        embedding=[0.0] * 1024,
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
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VIBE_RAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(
        "vibe_rag.indexing.embedder._resolve_ollama_host",
        lambda: (_ for _ in ()).throw(RuntimeError("Ollama not reachable")),
    )

    (tmp_path / "hello.py").write_text("x = 1\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)
        srv._embedder = old_embedder

    assert result["ok"] is False
    assert "Ollama not reachable" in _error_message(result)


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
            summary="The E2E repo marker for mistral-vibe is CERULEAN_PINEAPPLE_20260322.",
            content="The E2E repo marker for mistral-vibe is CERULEAN_PINEAPPLE_20260322.",
            embedding=[0.0] * 1024,
            project_id="mistral-vibe",
            memory_kind="fact",
        )
        result = search_memory("CERULEAN_PINEAPPLE_20260322")
    finally:
        srv._project_id = old_project_id

    assert result["ok"] is True
    assert result["results"][0]["project_id"] == "mistral-vibe"
    assert "CERULEAN_PINEAPPLE_20260322" in result["results"][0]["content"]


def test_search_memory_filters_stale_cross_project_results_when_project_memory_exists(tmp_db, mock_embedder):
    import vibe_rag.server as srv
    old_project_id = srv._project_id
    srv._get_db().remember_structured(
        summary="The marker is QUARTZ_MERIDIAN_20260322_Z9.",
        content="The marker is QUARTZ_MERIDIAN_20260322_Z9.",
        embedding=[0.0] * 1024,
        project_id="sink-repo",
        memory_kind="summary",
    )
    srv._get_user_db().remember_structured(
        summary="The marker is QUARTZ_MERIDIAN_20260322_Z9 in source-repo.",
        content="The marker is QUARTZ_MERIDIAN_20260322_Z9 in source-repo.",
        embedding=[0.0] * 1024,
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
            embedding=[0.0] * 1024,
            project_id="shared",
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
    assert result["memories"][0]["is_stale"] is True
    assert "project_id_mismatch" in result["memories"][0]["stale_reasons"]


def test_supersede_memory_marks_replacement(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    first_id = srv._get_user_db().remember_structured(
        summary="use sqlite for local search",
        content="use sqlite for local search",
        embedding=[0.0] * 1024,
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
        embedding=[0.0] * 1024,
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
        embedding=[0.0] * 1024,
        project_id=srv._ensure_project_id(),
        memory_kind="decision",
        metadata={"capture_kind": "manual"},
    )
    user_id = srv._get_user_db().remember_structured(
        summary="user auth decision",
        content="user auth decision",
        embedding=[0.0] * 1024,
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
        for item in memory_cleanup_report(limit=5)["candidates"]
    )


def test_load_session_context_downranks_cross_project_user_memory(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        current_id = srv._get_user_db().remember_structured(
            summary="auth constraint for sink repo",
            content="auth constraint for sink repo",
            embedding=[0.0] * 1024,
            project_id="sink-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        srv._get_user_db().remember_structured(
            summary="auth constraint for source repo",
            content="auth constraint for source repo",
            embedding=[0.0] * 1024,
            project_id="source-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        result = load_session_context("auth constraint", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._project_id = old_project_id

    assert [item["id"] for item in result["memories"][:1]] == [current_id]
    assert result["memories"][0]["provenance"]["is_current_project"] is True
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
            embedding=[0.0] * 1024,
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
    save_session_memory(
        task="Where should demo tokens be used?",
        response="Only demo tokens are allowed in this smoke-test API and production tokens are rejected.",
        source_session_id="sess-demo-memory",
        source_message_id="msg-demo-memory",
    )
    remember("Only demo tokens are allowed in this smoke-test API.")

    result = load_session_context("demo tokens", memory_limit=5, code_limit=0, docs_limit=0)

    assert result["memories"][0]["memory_kind"] == "constraint"
    assert result["memories"][0]["provenance"]["capture_kind"] == "freeform"
    assert any(
        item["provenance"]["capture_kind"] in {"session_rollup", "session_distillation"}
        for item in result["memories"][1:]
    )


def test_merge_memory_results_keeps_auto_captures_visible_when_manual_memory_exists():
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

    assert [item["id"] for item in merged] == [1, 2]


def test_memory_cleanup_report_surfaces_freeform_and_cross_project_candidates(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        remember("temporary freeform deployment note")
        srv._get_user_db().remember_structured(
            summary="old auth note",
            content="old auth note",
            embedding=[0.0] * 1024,
            project_id="source-repo",
            memory_kind="note",
            metadata={"capture_kind": "freeform"},
        )
        report = memory_cleanup_report(limit=5)
    finally:
        srv._project_id = old_project_id

    assert report["ok"] is True
    assert report["candidate_total"] >= 2
    reasons = {reason for item in report["candidates"] for reason in item["cleanup_reasons"]}
    assert "freeform_note" in reasons
    assert "cross_project_user_memory" in reasons


def test_project_status_includes_memory_cleanup_candidates(tmp_db, mock_embedder):
    remember("temporary cleanup candidate")
    status = project_status()
    assert status["ok"] is True
    assert status["status"]["cleanup_candidates"]
    assert status["status"]["cleanup_candidates"][0]["summary"] == "temporary cleanup candidate"


def test_memory_quality_report_summarizes_provenance_and_cleanup(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        remember("temporary freeform deployment note")
        srv._get_user_db().remember_structured(
            summary="current project constraint for auth",
            content="current project constraint for auth",
            embedding=[0.0] * 1024,
            project_id="sink-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        old_id = srv._get_user_db().remember_structured(
            summary="old auth note",
            content="old auth note",
            embedding=[0.0] * 1024,
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
            embedding=[0.0] * 1024,
            project_id="sink-repo",
            memory_kind="summary",
            metadata={"capture_kind": "session_rollup", "task": "Reply with only the project id loaded in session context.", "turn_count": 1},
        )
        srv._get_user_db().remember_structured(
            summary="Session summary: reply with only the project id loaded in session context",
            content="Session covered 1 turns.\n\nTurn 1\nUser: Reply with only the project id loaded in session context.\nAssistant: sink-repo",
            embedding=[0.0] * 1024,
            project_id="sink-repo",
            memory_kind="summary",
            metadata={"capture_kind": "session_rollup", "task": "Reply with only the project id loaded in session context.", "turn_count": 1},
        )
        report = memory_quality_report(limit=5)
    finally:
        srv._project_id = old_project_id

    assert report["ok"] is True
    assert report["summary"]["total_memories"] >= 4
    assert report["summary"]["stale_memories"] >= 1
    assert report["summary"]["superseded_memories"] >= 1
    assert report["by_source_db"]["user"] >= 3
    assert report["by_capture_kind"]["freeform"] >= 2
    assert report["by_source_type"]["manual_structured"] >= 1
    assert report["stale_reasons"]["project_id_mismatch"] >= 1
    assert report["cleanup_reasons"]["cross_project_user_memory"] >= 1
    assert report["cleanup_reasons"]["low_signal_auto_memory"] >= 1
    assert report["summary"]["duplicate_auto_memory_groups"] >= 1
    assert report["recommended_actions"]
    assert any("low-signal auto session summaries" in action for action in report["recommended_actions"])
    assert any("duplicate auto-captured session memories" in action for action in report["recommended_actions"])
    assert report["duplicate_auto_memory_groups"]
    assert report["top_cleanup_candidates"]


def test_cleanup_duplicate_auto_memories_reports_and_deletes_duplicates(tmp_db, mock_embedder):
    import vibe_rag.server as srv

    old_project_id = srv._project_id
    srv._project_id = "sink-repo"
    try:
        for offset in range(2):
            srv._get_user_db().remember_structured(
                summary="Session summary: reply with only the project id loaded in session context",
                content="Session covered 1 turns.\n\nTurn 1\nUser: Reply with only the project id loaded in session context.\nAssistant: sink-repo",
                embedding=[0.0] * 1024,
                project_id="sink-repo",
                memory_kind="summary",
                metadata={"capture_kind": "session_rollup", "task": "Reply with only the project id loaded in session context.", "turn_count": 1, "order": offset},
            )

        preview = cleanup_duplicate_auto_memories(limit=5, apply=False)
        applied = cleanup_duplicate_auto_memories(limit=5, apply=True)
        report = memory_quality_report(limit=5)
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
    assert report["summary"]["duplicate_auto_memory_groups"] == 0


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
            "created_at": "2026-03-23 00:00:00",
            "updated_at": "2026-03-23 00:00:00",
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
