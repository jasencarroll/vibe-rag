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
            self._conn = sqlite3.connect(str(self._path))
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
                embedding float[1024]
            );
        """)
        conn.commit()

    def list_tables(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
        return [row["name"] for row in rows]

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

    def search(self, query_embedding: list[float], limit: int = 10, language: str | None = None) -> list[dict]:
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

    def chunk_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]

    def clear(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            DELETE FROM code_chunks_vec;
            DELETE FROM code_chunks;
        """)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
