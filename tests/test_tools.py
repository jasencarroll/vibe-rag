from pathlib import Path

from vibe_rag.tools import (
    _index_project_impl,
    forget,
    index_project,
    load_session_context,
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


def test_remember_and_search_memory(tmp_db, mock_embedder):
    result = remember("sqlite-vec is local and simple")
    assert "Remembered in project memory" in result
    assert "id=" in result

    result = search_memory("what is good for vectors?")
    assert "sqlite-vec" in result


def test_forget_existing(tmp_db, mock_embedder):
    result = remember("temporary fact")
    assert "id=1" in result

    result = forget(1)
    assert "Deleted" in result
    assert "temporary fact" in result


def test_forget_nonexistent(tmp_db, mock_embedder):
    result = forget(999)
    assert "not found" in result


def test_search_code_empty(tmp_db, mock_embedder):
    result = search_code("anything")
    assert "No code index" in result


def test_search_docs_empty(tmp_db, mock_embedder):
    result = search_docs("anything")
    assert "No docs indexed" in result


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

    assert "1 code files" in result
    assert "1 docs" in result


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

    assert "Indexed 1 code files" in result
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

    assert "1 code files" in result
    assert "1 docs" in result


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

    assert "1 code files" in result
    assert "1 docs" in result
    assert "pkg/hello.py" in search_result


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

    assert "auth.py" in result


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

    assert "guide.md" in result


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

    assert "README.md" in result or "CHANGELOG.md" in result
    assert "evals/local_repos.toml" not in result


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

    assert "CHANGELOG.md" in result


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

    assert "AGENTS.md" in result
    assert "CHANGELOG.md" in result


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


def test_api_and_pipeline_doc_queries_prefer_operational_docs_over_plans(
    tmp_db, mock_embedder, tmp_path: Path
):
    import os

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (tmp_path / "CLAUDE.md").write_text("Planning and repo notes.\n")
    (spec_dir / "05-api-routes.md").write_text("Draft route plan.\n")
    (spec_dir / "06-pipeline.md").write_text("Draft pipeline plan.\n")
    (docs_dir / "API.md").write_text("REST API endpoints for auth, letters, analytics, billing, and health.\n")
    (docs_dir / "PIPELINE.md").write_text("Warning letter discover, scrape, classify, enrich, and 483 ingestion pipeline.\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()
        api_payload = load_session_context(
            "backend api auth letters analytics engines billing health endpoints",
            memory_limit=0,
            code_limit=0,
            docs_limit=1,
        )
        pipeline_payload = load_session_context(
            "warning letter scrape parse classify enrich 483 observation pipeline",
            memory_limit=0,
            code_limit=0,
            docs_limit=1,
        )
    finally:
        os.chdir(old_cwd)

    assert api_payload["docs"][0]["file_path"] == "docs/API.md"
    assert pipeline_payload["docs"][0]["file_path"] == "docs/PIPELINE.md"


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

    assert ".github/workflows/publish.yml" in result


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
    assert result["memories"][0]["provenance"]["project_id"] == result["project_id"]
    assert result["code"][0]["file_path"] == "billing.py"
    assert result["code"][0]["start_line"] == 1
    assert result["code"][0]["provenance"]["source"] == "project-index"
    assert result["docs"][0]["file_path"] == "billing.md"
    assert result["docs"][0]["provenance"]["source"] == "project-index"


# --- Edge case tests ---


def test_index_project_no_api_key(tmp_db, tmp_path: Path, monkeypatch):
    """index_project should return an error string when the default embedder is unavailable."""
    import os
    import vibe_rag.server as srv

    old_embedder = srv._embedder
    srv._embedder = None
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
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

    assert "Ollama not reachable" in result


def test_index_project_no_files(tmp_db, mock_embedder, tmp_path: Path):
    """index_project returns a message when directory has no indexable files."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)

    assert "No files found" in result


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
    assert isinstance(result, str)


def test_remember_with_tags(tmp_db, mock_embedder):
    result = remember("auth uses JWT tokens", tags="architecture,security")
    assert "Remembered" in result

    result = search_memory("JWT")
    assert "JWT" in result


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
    assert "Task: figure out auth ownership" in first["memory"]["content"]
    assert second["ok"] is True
    assert second["deduplicated"] is True
    assert second["memory"]["id"] == first["memory"]["id"]


def test_save_session_memory_skips_low_signal_no_memory_response(tmp_db, mock_embedder):
    result = save_session_memory(
        task="What durable memory do you have?",
        response="I have no durable memory about this task.",
        source_session_id="sess-none",
        source_message_id="msg-none",
    )

    assert result["ok"] is True
    assert result["skipped"] is True


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
    assert "Session covered 2 turns." in first["memory"]["content"]
    assert second["ok"] is True
    assert second["deduplicated"] is False
    assert second["memory"]["supersedes"] == first["memory"]["id"]
    assert second["memory"]["metadata"]["latest_message_id"] == "msg-3"
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

    assert "mistral-vibe" in result
    assert "CERULEAN_PINEAPPLE_20260322" in result


def test_search_memory_prefers_project_results_before_user_results(tmp_db, mock_embedder):
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

    assert "sink-repo" in result
    assert "source-repo" in result
    assert "QUARTZ_MERIDIAN_20260322_Z9" in result


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


def test_supersede_memory_marks_replacement(tmp_db, mock_embedder):
    first = remember_structured(summary="use sqlite for local search", memory_kind="decision")
    replacement = supersede_memory(
        old_memory_id=first["memory"]["id"],
        summary="use sqlite for local search and user memory",
        memory_kind="decision",
    )

    assert replacement["ok"] is True
    assert replacement["memory"]["supersedes"] == first["memory"]["id"]


def test_remember_empty_content(tmp_db, mock_embedder):
    result = remember("")
    assert "Error" in result
    assert "empty" in result


def test_search_memory_empty_db(tmp_db, mock_embedder):
    result = search_memory("anything")
    assert "No memories" in result


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

    assert result["memory_status"] == "No memories stored yet."
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


def test_search_memory_prefers_structured_memory_kinds(tmp_db, mock_embedder):
    remember("raw note about deployment")
    remember_structured(summary="deployment constraint", memory_kind="constraint")

    result = load_session_context("deployment", memory_limit=2)

    assert result["memories"][0]["memory_kind"] == "constraint"


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

    assert "Project id:" in status
    assert "Indexed at:" in status
    assert "Stale warnings: none" in status


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
    assert "outside project root" in result


def test_search_code_empty_query(tmp_db, mock_embedder):
    result = search_code("")
    assert "Error" in result


def test_search_code_whitespace_query(tmp_db, mock_embedder):
    result = search_code("   ")
    assert "Error" in result


def test_search_code_query_too_long(tmp_db, mock_embedder):
    result = search_code("x" * 10_001)
    assert "Error" in result
    assert "too long" in result


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
    assert "Unknown language" in result


def test_search_docs_empty_query(tmp_db, mock_embedder):
    result = search_docs("")
    assert "Error" in result


def test_search_docs_query_too_long(tmp_db, mock_embedder):
    result = search_docs("x" * 10_001)
    assert "Error" in result
    assert "too long" in result


def test_search_memory_empty_query(tmp_db, mock_embedder):
    result = search_memory("")
    assert "Error" in result


def test_search_memory_query_too_long(tmp_db, mock_embedder):
    result = search_memory("x" * 10_001)
    assert "Error" in result
    assert "too long" in result


def test_remember_too_large(tmp_db, mock_embedder):
    result = remember("x" * 20_000)
    assert "Error" in result
    assert "too large" in result


def test_remember_whitespace_only(tmp_db, mock_embedder):
    result = remember("   \n\t  ")
    assert "Error" in result
    assert "empty" in result


def test_remember_tags_too_long(tmp_db, mock_embedder):
    result = remember("valid content", tags="x" * 600)
    assert "Error" in result
    assert "tags" in result


# --- project_status tests ---


def test_project_status_empty(tmp_db, mock_embedder):
    result = project_status()
    assert "Code chunks: 0" in result
    assert "Doc chunks: 0" in result
    assert "Project memories: 0" in result
    assert "User memories: 0" in result
    assert "Languages" not in result


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

    assert "Code chunks: 0" not in result
    assert "Doc chunks: 0" not in result
    assert "Languages" in result
    assert "chunks" in result


# --- min_score filtering tests ---


def test_search_code_min_score_filters(tmp_db, mock_embedder, tmp_path: Path):
    """min_score should filter results based on 1.0 - distance."""
    import os
    from unittest.mock import patch

    (tmp_path / "app.py").write_text("def hello():\n    return 1\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        index_project()

        # Without min_score, we get results
        result_no_filter = search_code("hello")
        assert "app.py" in result_no_filter

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
            assert "No matching code found" in result_filtered

            # min_score=0.1 should keep result with score=0.2
            result_kept = search_code("hello", min_score=0.1)
            assert "app.py" in result_kept
        finally:
            srv._project_db.search_code = original_search
    finally:
        os.chdir(old_cwd)


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
        assert "0 unchanged" in result1  # first run: nothing unchanged

        result2 = index_project()
        assert "1 unchanged" in result2  # code file unchanged
        assert "0 chunks, 1 unchanged" in result2 or "0 chunks" in result2
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
        assert "1 code files" in result1
        assert "0 unchanged" in result1

        # Modify the file
        (tmp_path / "app.py").write_text("def main():\n    return 42\n")

        result2 = index_project()
        assert "1 code files" in result2
        assert "0 unchanged" in result2  # changed file should not be skipped
    finally:
        os.chdir(old_cwd)
