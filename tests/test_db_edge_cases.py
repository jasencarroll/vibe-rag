"""Edge case tests for vibe_rag.db.sqlite.SqliteVecDB."""
from __future__ import annotations

from pathlib import Path

import pytest
from vibe_rag.db.sqlite import SqliteVecDB


@pytest.fixture
def db(tmp_path: Path):
    d = SqliteVecDB(tmp_path / "edge.db")
    d.initialize()
    yield d
    d.close()


def _fake_embedding(val: float = 0.1) -> list[float]:
    return [val] * 2560


class TestUpsertChunks:
    def test_upsert_then_search(self, db: SqliteVecDB):
        chunks = [
            {"file_path": "a.py", "chunk_index": 0, "content": "def foo(): pass",
             "language": "python", "symbol": "foo", "start_line": 1, "end_line": 1},
        ]
        db.upsert_chunks(chunks, [_fake_embedding(0.5)])
        assert db.code_chunk_count() == 1
        results = db.search_code(_fake_embedding(0.5), limit=5)
        assert len(results) >= 1
        assert results[0]["file_path"] == "a.py"

    def test_upsert_replaces_on_conflict(self, db: SqliteVecDB):
        chunk = {"file_path": "a.py", "chunk_index": 0, "content": "v1",
                 "language": "python", "symbol": "f", "start_line": 1, "end_line": 1}
        db.upsert_chunks([chunk], [_fake_embedding(0.1)])
        chunk["content"] = "v2"
        db.upsert_chunks([chunk], [_fake_embedding(0.2)])
        assert db.code_chunk_count() == 1

    def test_upsert_empty_list(self, db: SqliteVecDB):
        db.upsert_chunks([], [])
        assert db.code_chunk_count() == 0


class TestSearchWithLanguageFilter:
    def test_filter_by_language(self, db: SqliteVecDB):
        py_chunk = {"file_path": "a.py", "chunk_index": 0, "content": "x=1",
                    "language": "python", "symbol": None, "start_line": 1, "end_line": 1}
        js_chunk = {"file_path": "b.js", "chunk_index": 0, "content": "x=1",
                    "language": "javascript", "symbol": None, "start_line": 1, "end_line": 1}
        db.upsert_chunks([py_chunk, js_chunk],
                         [_fake_embedding(0.1), _fake_embedding(0.2)])
        results = db.search_code(_fake_embedding(0.1), limit=10, language="python")
        languages = {r["language"] for r in results}
        assert languages <= {"python"}

    def test_filter_no_match(self, db: SqliteVecDB):
        chunk = {"file_path": "a.py", "chunk_index": 0, "content": "x=1",
                 "language": "python", "symbol": None, "start_line": 1, "end_line": 1}
        db.upsert_chunks([chunk], [_fake_embedding(0.1)])
        results = db.search_code(_fake_embedding(0.1), limit=10, language="rust")
        assert results == []


class TestClearOperations:
    def test_clear_code(self, db: SqliteVecDB):
        chunk = {"file_path": "a.py", "chunk_index": 0, "content": "x",
                 "language": "python", "symbol": None, "start_line": 1, "end_line": 1}
        db.upsert_chunks([chunk], [_fake_embedding()])
        assert db.code_chunk_count() == 1
        db.clear_code()
        assert db.code_chunk_count() == 0

    def test_clear_docs(self, db: SqliteVecDB):
        doc = {"file_path": "r.md", "chunk_index": 0, "content": "hello"}
        db.upsert_docs([doc], [_fake_embedding()])
        assert db.doc_count() == 1
        db.clear_docs()
        assert db.doc_count() == 0

    def test_compat_clear_alias(self, db: SqliteVecDB):
        chunk = {"file_path": "a.py", "chunk_index": 0, "content": "x",
                 "language": None, "symbol": None, "start_line": 1, "end_line": 1}
        db.upsert_chunks([chunk], [_fake_embedding()])
        db.clear()  # compat alias
        assert db.chunk_count() == 0


class TestListTables:
    def test_has_expected_tables(self, db: SqliteVecDB):
        tables = db.list_tables()
        for expected in ("code_chunks", "memories", "docs"):
            assert expected in tables


class TestCloseReopen:
    def test_close_and_reopen(self, tmp_path: Path):
        db = SqliteVecDB(tmp_path / "reopen.db")
        db.initialize()
        db.remember("fact", _fake_embedding(), tags="test")
        assert db.memory_count() == 1
        db.close()
        # Reopen
        db2 = SqliteVecDB(tmp_path / "reopen.db")
        db2.initialize()
        assert db2.memory_count() == 1
        db2.close()


class TestMemories:
    def test_remember_and_forget(self, db: SqliteVecDB):
        mid = db.remember("note", _fake_embedding(), tags="t1")
        assert db.memory_count() == 1
        content = db.forget(mid)
        assert content == "note"
        assert db.memory_count() == 0

    def test_forget_nonexistent(self, db: SqliteVecDB):
        assert db.forget(9999) is None

    def test_search_memories(self, db: SqliteVecDB):
        db.remember("alpha", _fake_embedding(0.1))
        db.remember("beta", _fake_embedding(0.2))
        results = db.search_memories(_fake_embedding(0.1), limit=5)
        assert len(results) >= 1
