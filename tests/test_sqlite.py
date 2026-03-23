import pytest
from pathlib import Path
from vibe_rag.db.sqlite import SqliteVecDB


@pytest.fixture
def db(tmp_path: Path):
    db_path = tmp_path / "index.db"
    db = SqliteVecDB(db_path)
    db.initialize()
    return db


def test_initialize_creates_tables(db):
    tables = db.list_tables()
    assert "code_chunks" in tables
    assert "code_chunks_vec" in tables


def test_insert_and_search(db):
    chunks = [{"file_path": "src/main.py", "chunk_index": 0, "content": "def hello(): print('hi')",
               "language": "python", "symbol": "hello", "start_line": 1, "end_line": 1}]
    embeddings = [[0.1] * 1024]
    db.upsert_chunks(chunks, embeddings)
    results = db.search([0.1] * 1024, limit=5)
    assert len(results) == 1
    assert results[0]["file_path"] == "src/main.py"
    assert results[0]["symbol"] == "hello"


def test_clear_and_reindex(db):
    chunks = [{"file_path": "src/a.py", "chunk_index": 0, "content": "x = 1",
               "language": "python", "symbol": None, "start_line": 1, "end_line": 1}]
    db.upsert_chunks(chunks, [[0.1] * 1024])
    assert db.chunk_count() == 1
    db.clear()
    assert db.chunk_count() == 0


def test_search_empty_db(db):
    results = db.search([0.1] * 1024, limit=5)
    assert results == []


def test_search_with_language_filter(db):
    chunks = [
        {"file_path": "a.py", "chunk_index": 0, "content": "x", "language": "python", "symbol": None, "start_line": 1, "end_line": 1},
        {"file_path": "b.js", "chunk_index": 0, "content": "y", "language": "javascript", "symbol": None, "start_line": 1, "end_line": 1},
    ]
    embeddings = [[0.1] * 1024, [0.1] * 1024]
    db.upsert_chunks(chunks, embeddings)
    results = db.search([0.1] * 1024, limit=10, language="python")
    assert len(results) == 1
    assert results[0]["file_path"] == "a.py"


def test_initialize_respects_custom_dimensions(tmp_path: Path):
    db = SqliteVecDB(tmp_path / "custom.db", embedding_dimensions=1024)
    db.initialize()
    chunks = [{"file_path": "src/main.py", "chunk_index": 0, "content": "def hello(): pass",
               "language": "python", "symbol": "hello", "start_line": 1, "end_line": 1}]
    embeddings = [[0.1] * 1024]
    db.upsert_chunks(chunks, embeddings)
    results = db.search([0.1] * 1024, limit=5)
    assert len(results) == 1


def test_upsert_chunks_rejects_embedding_count_mismatch(db):
    chunks = [{"file_path": "src/main.py", "chunk_index": 0, "content": "def hello(): pass",
               "language": "python", "symbol": "hello", "start_line": 1, "end_line": 1}]

    with pytest.raises(RuntimeError, match="Embedding count mismatch"):
        db.upsert_chunks(chunks, [])


def test_get_setting_json_status_reports_invalid_json(db):
    db.set_setting("project_index_metadata", "{not-json")

    parsed, error = db.get_setting_json_status("project_index_metadata")

    assert parsed is None
    assert "invalid JSON" in error
