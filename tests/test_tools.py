from pathlib import Path

from vibe_rag.tools import (
    remember,
    search_memory,
    forget,
    search_code,
    search_docs,
    index_project,
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
