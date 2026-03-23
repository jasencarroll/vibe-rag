"""Tests that exercise the new composite conftest fixtures."""

import pytest
from pathlib import Path

from vibe_rag.tools import (
    search_code,
    search_docs,
    search_memory,
    forget,
    load_session_context,
)


# ---------------------------------------------------------------------------
# indexed_project fixture
# ---------------------------------------------------------------------------


def test_indexed_project_has_code_index(indexed_project):
    result = search_code("authenticate")
    assert result["ok"] is True
    paths = [item["file_path"] for item in result["results"]]
    assert "auth.py" in paths


def test_indexed_project_has_doc_index(indexed_project):
    result = search_docs("deploy")
    assert result["ok"] is True
    paths = [item["file_path"] for item in result["results"]]
    assert "guide.md" in paths


def test_indexed_project_search_code_billing(indexed_project):
    result = search_code("invoice customer")
    assert result["ok"] is True
    paths = [item["file_path"] for item in result["results"]]
    assert "billing.py" in paths


def test_indexed_project_load_session_context(indexed_project):
    result = load_session_context(
        "billing invoice",
        memory_limit=0,
        code_limit=3,
        docs_limit=2,
    )
    assert result["ok"] is True
    assert len(result["code"]) > 0
    code_paths = [item["file_path"] for item in result["code"]]
    assert "billing.py" in code_paths


# ---------------------------------------------------------------------------
# populated_memory fixture
# ---------------------------------------------------------------------------


def test_populated_memory_ids_are_assigned(populated_memory):
    assert "decision" in populated_memory
    assert "fact" in populated_memory
    assert "note" in populated_memory
    assert all(isinstance(v, int) for v in populated_memory.values())


def test_populated_memory_searchable(populated_memory):
    result = search_memory("auth tokens gateway")
    assert result["ok"] is True
    assert len(result["results"]) > 0
    summaries = [item["summary"] for item in result["results"]]
    assert "gateway owns auth tokens" in summaries


def test_populated_memory_forget(populated_memory):
    fact_id = populated_memory["fact"]
    result = forget(fact_id)
    assert result["ok"] is True
    assert result["deleted"] is True


# ---------------------------------------------------------------------------
# clean_env fixture
# ---------------------------------------------------------------------------

_ALL_PROVIDER_VARS = (
    "VIBE_RAG_EMBEDDING_PROVIDER",
    "VIBE_RAG_EMBEDDING_MODEL",
    "VIBE_RAG_EMBEDDING_DIMENSIONS",
    "VIBE_RAG_OLLAMA_HOST",
    "VIBE_RAG_CODE_EMBEDDING_MODEL",
    "VIBE_RAG_DB",
    "VIBE_RAG_USER_DB",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "VOYAGE_API_KEY",
    "OLLAMA_HOST",
)


def test_clean_env_removes_all_provider_vars(clean_env):
    import os

    for var in _ALL_PROVIDER_VARS:
        assert var not in os.environ, f"{var} should be removed by clean_env"


def test_clean_env_allows_setting_vars(clean_env, monkeypatch):
    import os

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")
    assert os.environ["MISTRAL_API_KEY"] == "test-key-123"

    # Other provider vars should still be absent
    assert "OPENAI_API_KEY" not in os.environ
    assert "VOYAGE_API_KEY" not in os.environ
