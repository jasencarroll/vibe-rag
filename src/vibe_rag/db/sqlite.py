from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


class SqliteVecDB:
    def __init__(self, path: Path):
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        return self._conn

    def initialize(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                language TEXT,
                symbol TEXT,
                start_line INTEGER,
                end_line INTEGER,
                indexed_at TEXT DEFAULT (datetime('now')),
                UNIQUE(file_path, chunk_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[1536]
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                tags TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[1536]
            );

            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                indexed_at TEXT DEFAULT (datetime('now')),
                UNIQUE(file_path, chunk_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[1536]
            );

            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                kind TEXT NOT NULL
            );
        """)
        conn.commit()

    # --- Code chunks ---

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        conn = self._get_conn()
        for chunk, embedding in zip(chunks, embeddings):
            cursor = conn.execute(
                """INSERT OR REPLACE INTO code_chunks
                   (file_path, chunk_index, content, language, symbol, start_line, end_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk["file_path"], chunk["chunk_index"], chunk["content"],
                 chunk["language"], chunk["symbol"], chunk["start_line"], chunk["end_line"]),
            )
            row_id = cursor.lastrowid
            conn.execute(
                "INSERT OR REPLACE INTO code_chunks_vec (id, embedding) VALUES (?, ?)",
                (row_id, sqlite_vec.serialize_float32(embedding)),
            )
        conn.commit()

    def search_code(self, query_embedding: list[float], limit: int = 10, language: str | None = None) -> list[dict]:
        conn = self._get_conn()
        serialized = sqlite_vec.serialize_float32(query_embedding)

        if language:
            rows = conn.execute(
                """SELECT c.file_path, c.chunk_index, c.content, c.language,
                          c.symbol, c.start_line, c.end_line, v.distance
                   FROM code_chunks_vec v
                   JOIN code_chunks c ON c.id = v.id
                   WHERE v.embedding MATCH ? AND k = ?
                     AND c.language = ?
                   ORDER BY v.distance""",
                (serialized, limit * 3, language),
            ).fetchall()
            rows = rows[:limit]
        else:
            rows = conn.execute(
                """SELECT c.file_path, c.chunk_index, c.content, c.language,
                          c.symbol, c.start_line, c.end_line, v.distance
                   FROM code_chunks_vec v
                   JOIN code_chunks c ON c.id = v.id
                   WHERE v.embedding MATCH ? AND k = ?
                   ORDER BY v.distance""",
                (serialized, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def clear_code(self) -> None:
        conn = self._get_conn()
        conn.executescript("DELETE FROM code_chunks_vec; DELETE FROM code_chunks;")
        conn.commit()

    def code_chunk_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]

    # --- Memories ---

    def remember(self, content: str, embedding: list[float], tags: str = "") -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO memories (content, tags) VALUES (?, ?)",
            (content, tags),
        )
        row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
            (row_id, sqlite_vec.serialize_float32(embedding)),
        )
        conn.commit()
        return row_id

    def search_memories(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        serialized = sqlite_vec.serialize_float32(query_embedding)
        rows = conn.execute(
            """SELECT m.id, m.content, m.tags, m.created_at, v.distance
               FROM memories_vec v
               JOIN memories m ON m.id = v.id
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (serialized, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, memory_id: int) -> str | None:
        conn = self._get_conn()
        row = conn.execute("SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        content = row["content"]
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
        conn.commit()
        return content

    def memory_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # --- Docs ---

    def upsert_docs(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        conn = self._get_conn()
        for chunk, embedding in zip(chunks, embeddings):
            cursor = conn.execute(
                """INSERT OR REPLACE INTO docs (file_path, chunk_index, content)
                   VALUES (?, ?, ?)""",
                (chunk["file_path"], chunk["chunk_index"], chunk["content"]),
            )
            row_id = cursor.lastrowid
            conn.execute(
                "INSERT OR REPLACE INTO docs_vec (id, embedding) VALUES (?, ?)",
                (row_id, sqlite_vec.serialize_float32(embedding)),
            )
        conn.commit()

    def search_docs(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        serialized = sqlite_vec.serialize_float32(query_embedding)
        rows = conn.execute(
            """SELECT d.file_path, d.chunk_index, d.content, v.distance
               FROM docs_vec v
               JOIN docs d ON d.id = v.id
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (serialized, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_docs(self) -> None:
        conn = self._get_conn()
        conn.executescript("DELETE FROM docs_vec; DELETE FROM docs;")
        conn.commit()

    def doc_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def language_stats(self) -> dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT language, COUNT(*) as cnt FROM code_chunks GROUP BY language"
        ).fetchall()
        return {row["language"]: row["cnt"] for row in rows}

    # --- File hashes (incremental indexing) ---

    def get_file_hashes(self, kind: str) -> dict[str, str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_path, content_hash FROM file_hashes WHERE kind = ?",
            (kind,),
        ).fetchall()
        return {row["file_path"]: row["content_hash"] for row in rows}

    def set_file_hash(self, file_path: str, content_hash: str, kind: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, content_hash, kind) VALUES (?, ?, ?)",
            (file_path, content_hash, kind),
        )
        conn.commit()

    def delete_file_chunks(self, file_path: str, kind: str = "code") -> None:
        conn = self._get_conn()
        if kind == "code":
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM code_chunks WHERE file_path = ?", (file_path,)
            ).fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM code_chunks_vec WHERE id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM code_chunks WHERE id IN ({placeholders})", ids)
        else:
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM docs WHERE file_path = ?", (file_path,)
            ).fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM docs_vec WHERE id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM docs WHERE id IN ({placeholders})", ids)
        conn.execute("DELETE FROM file_hashes WHERE file_path = ? AND kind = ?", (file_path, kind))
        conn.commit()

    def delete_file_hashes(self, file_paths: list[str], kind: str) -> None:
        conn = self._get_conn()
        for fp in file_paths:
            conn.execute(
                "DELETE FROM file_hashes WHERE file_path = ? AND kind = ?",
                (fp, kind),
            )
        conn.commit()

    # --- Compat aliases ---

    def search(self, query_embedding: list[float], limit: int = 10, language: str | None = None) -> list[dict]:
        return self.search_code(query_embedding, limit, language)

    def clear(self) -> None:
        self.clear_code()

    def chunk_count(self) -> int:
        return self.code_chunk_count()

    def list_tables(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
        return [row["name"] for row in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
