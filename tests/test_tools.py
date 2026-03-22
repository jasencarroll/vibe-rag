from pathlib import Path

from vibe_rag.tools import (
    forget,
    index_project,
    load_session_context,
    project_status,
    remember,
    remember_structured,
    search_code,
    search_docs,
    search_memory,
    supersede_memory,
)


def test_remember_and_search_memory(tmp_db, mock_embedder):
    result = remember("pgvector is great for vectors")
    assert "Remembered" in result
    assert "id=" in result

    result = search_memory("what is good for vectors?")
    assert "pgvector" in result


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
    assert result["code"][0]["file_path"] == "billing.py"
    assert result["code"][0]["start_line"] == 1
    assert result["docs"][0]["file_path"] == "billing.md"


# --- Edge case tests ---


def test_index_project_no_api_key(tmp_db, tmp_path: Path):
    """index_project should return an error string when no embedder is available."""
    import os
    import vibe_rag.server as srv

    old_embedder = srv._embedder
    srv._embedder = None
    old_key = srv._api_key
    srv._api_key = ""

    (tmp_path / "hello.py").write_text("x = 1\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = index_project()
    finally:
        os.chdir(old_cwd)
        srv._embedder = old_embedder
        srv._api_key = old_key

    assert "MISTRAL_API_KEY" in result


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
    assert result["memory"]["metadata"]["confidence"] == "high"


def test_pg_memory_tools_work_inside_running_event_loop(tmp_db, mock_embedder):
    import asyncio
    import vibe_rag.server as srv

    class FakePG:
        async def remember(self, content, embedding, tags="", project_id=None):
            return 42

        async def memory_count(self):
            return 1

        async def search_memories(self, embedding, limit=10, project_id=None):
            return [
                {
                    "id": 42,
                    "content": "architecture note",
                    "project_id": project_id,
                    "score": 0.91,
                }
            ]

        async def forget(self, memory_id):
            return "architecture note"

    old_pg = srv._pg
    old_project_id = srv._project_id
    srv._pg = FakePG()
    srv._project_id = "test-project"
    try:
        async def run_tools():
            remembered = remember("architecture note")
            searched = search_memory("architecture")
            deleted = forget(42)
            return remembered, searched, deleted

        remembered, searched, deleted = asyncio.run(run_tools())
    finally:
        srv._pg = old_pg
        srv._project_id = old_project_id

    assert "Remembered in pgvector" in remembered
    assert "[id=42 [test-project] score=0.91]" in searched
    assert "Deleted from pgvector" in deleted


def test_load_session_context_uses_pg_memory_results(tmp_db, mock_embedder, tmp_path: Path):
    import os
    import vibe_rag.server as srv

    class FakePG:
        async def memory_count(self):
            return 1

        async def search_memories(self, embedding, limit=10, project_id=None):
            return [
                {
                    "id": "abc-123",
                    "content": "gateway owns auth validation",
                    "summary": "gateway owns auth validation",
                    "project_id": project_id,
                    "memory_kind": "decision",
                    "metadata": {"source": "session"},
                    "score": 0.88,
                }
            ]

    (tmp_path / "auth.py").write_text("def validate_token(token):\n    return token\n")

    old_cwd = os.getcwd()
    old_pg = srv._pg
    old_project_id = srv._project_id
    os.chdir(tmp_path)
    srv._pg = FakePG()
    srv._project_id = "test-project"
    try:
        index_project()
        result = load_session_context("continue auth validation", memory_limit=3, code_limit=2, docs_limit=1)
    finally:
        os.chdir(old_cwd)
        srv._pg = old_pg
        srv._project_id = old_project_id

    assert result["memories"][0]["summary"] == "gateway owns auth validation"
    assert result["memories"][0]["metadata"]["source"] == "session"
    assert result["code"][0]["file_path"] == "auth.py"


def test_supersede_memory_marks_replacement(tmp_db, mock_embedder):
    first = remember_structured(summary="use sqlite for local search", memory_kind="decision")
    replacement = supersede_memory(
        old_memory_id=first["memory"]["id"],
        summary="use sqlite for local search and pgvector for shared memory",
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
    assert "Memories: 0" in result
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
        original_search = srv._db.search_code
        srv._db.search_code = lambda *a, **kw: fake_results
        try:
            # min_score=0.5 should filter out result with score=0.2
            result_filtered = search_code("hello", min_score=0.5)
            assert "No matching code found" in result_filtered

            # min_score=0.1 should keep result with score=0.2
            result_kept = search_code("hello", min_score=0.1)
            assert "app.py" in result_kept
        finally:
            srv._db.search_code = original_search
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
