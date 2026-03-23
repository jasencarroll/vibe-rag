from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec


class SqliteVecDB:
    def __init__(self, path: Path, embedding_dimensions: int = 1024):
        self._path = path
        self._embedding_dimensions = embedding_dimensions
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
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

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
                embedding float[{self._embedding_dimensions}]
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                tags TEXT,
                project_id TEXT,
                memory_kind TEXT DEFAULT 'note',
                summary TEXT,
                metadata_json TEXT DEFAULT '{{}}',
                source_session_id TEXT,
                source_message_id TEXT,
                supersedes TEXT,
                superseded_by TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{self._embedding_dimensions}]
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
                embedding float[{self._embedding_dimensions}]
            );

            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                kind TEXT NOT NULL
            );
            """
        )
        self._ensure_dimensions(conn)
        self._ensure_memory_columns(conn)
        conn.commit()

    def get_setting(self, key: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def get_setting_json(self, key: str) -> dict | None:
        parsed, _ = self.get_setting_json_status(key)
        return parsed

    def get_setting_json_status(self, key: str) -> tuple[dict | None, str | None]:
        raw = self.get_setting(key)
        if raw is None:
            return None, None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None, f"setting '{key}' contains invalid JSON"
        if not isinstance(parsed, dict):
            return None, f"setting '{key}' must contain a JSON object"
        return parsed, None

    def set_setting_json(self, key: str, value: dict) -> None:
        self.set_setting(key, json.dumps(value))

    def _ensure_dimensions(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'embedding_dimensions'"
        ).fetchone()
        if row is None:
            schema_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'code_chunks_vec'"
            ).fetchone()
            if schema_row and schema_row["sql"]:
                match = re.search(r"float\[(\d+)\]", schema_row["sql"])
                if not match:
                    raise RuntimeError("Unable to determine embedding dimensions from existing sqlite-vec schema")
                existing_dimensions = int(match.group(1))
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES ('embedding_dimensions', ?)",
                    (str(existing_dimensions),),
                )
                if self._embedding_dimensions != existing_dimensions:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: db={existing_dimensions}, requested={self._embedding_dimensions}"
                    )
                return
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('embedding_dimensions', ?)",
                (str(self._embedding_dimensions),),
            )
            return

        existing_dimensions = int(row["value"])
        if existing_dimensions != self._embedding_dimensions:
            raise RuntimeError(
                f"Embedding dimension mismatch: db={existing_dimensions}, requested={self._embedding_dimensions}"
            )

    def _ensure_memory_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(memories)").fetchall()
        columns = {row["name"] for row in rows}
        additions = {
            "project_id": "ALTER TABLE memories ADD COLUMN project_id TEXT",
            "memory_kind": "ALTER TABLE memories ADD COLUMN memory_kind TEXT DEFAULT 'note'",
            "summary": "ALTER TABLE memories ADD COLUMN summary TEXT",
            "metadata_json": "ALTER TABLE memories ADD COLUMN metadata_json TEXT DEFAULT '{}'",
            "source_session_id": "ALTER TABLE memories ADD COLUMN source_session_id TEXT",
            "source_message_id": "ALTER TABLE memories ADD COLUMN source_message_id TEXT",
            "supersedes": "ALTER TABLE memories ADD COLUMN supersedes TEXT",
            "superseded_by": "ALTER TABLE memories ADD COLUMN superseded_by TEXT",
            "updated_at": "ALTER TABLE memories ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))",
        }
        for name, statement in additions.items():
            if name not in columns:
                conn.execute(statement)

    # --- Code chunks ---

    def _decode_metadata_json(self, raw: str | None) -> dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _row_to_memory(self, row) -> dict:
        result = dict(row)
        result["metadata"] = self._decode_metadata_json(result.pop("metadata_json", None))
        return result

    def _validate_embedding_count(
        self, items: list[dict], embeddings: list[list[float]], *, kind: str
    ) -> None:
        if len(items) != len(embeddings):
            raise RuntimeError(
                f"Embedding count mismatch for {kind}: {len(items)} items but {len(embeddings)} embeddings"
            )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        self._validate_embedding_count(chunks, embeddings, kind="code chunks")
        conn = self._get_conn()
        for chunk, embedding in zip(chunks, embeddings):
            cursor = conn.execute(
                """INSERT OR REPLACE INTO code_chunks
                   (file_path, chunk_index, content, language, symbol, start_line, end_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk["file_path"],
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk["language"],
                    chunk["symbol"],
                    chunk["start_line"],
                    chunk["end_line"],
                ),
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
                          c.symbol, c.start_line, c.end_line, c.indexed_at, v.distance
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
                          c.symbol, c.start_line, c.end_line, c.indexed_at, v.distance
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

    def lexical_search_code(self, terms: list[str], limit: int = 10) -> list[dict]:
        if not terms:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT file_path, chunk_index, content, language, symbol, start_line, end_line, indexed_at
            FROM code_chunks
            """
        ).fetchall()
        results: list[dict] = []
        for row in rows:
            item = dict(row)
            haystack = f"{item['file_path']} {item['content']}".lower()
            matches = sum(1 for term in terms if term in haystack)
            if matches == 0:
                continue
            item["score"] = matches / len(terms)
            results.append(item)
        results.sort(key=lambda item: (-float(item["score"]), str(item["file_path"]), int(item["chunk_index"])))
        return results[:limit]

    # --- Memories ---

    def remember(
        self,
        content: str,
        embedding: list[float],
        tags: str = "",
        project_id: str | None = None,
    ) -> int:
        return self.remember_structured(
            summary=content[:200],
            content=content,
            embedding=embedding,
            tags=tags,
            memory_kind="note",
            project_id=project_id,
        )

    def remember_structured(
        self,
        summary: str,
        content: str,
        embedding: list[float],
        tags: str = "",
        project_id: str | None = None,
        memory_kind: str = "note",
        metadata: dict | None = None,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        supersedes: int | None = None,
    ) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO memories (
                content, tags, project_id, memory_kind, summary, metadata_json,
                source_session_id, source_message_id, supersedes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                content,
                tags,
                project_id,
                memory_kind,
                summary,
                json.dumps(metadata or {}),
                source_session_id,
                source_message_id,
                supersedes,
            ),
        )
        row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
            (row_id, sqlite_vec.serialize_float32(embedding)),
        )
        if supersedes is not None:
            conn.execute(
                "UPDATE memories SET superseded_by = ?, updated_at = datetime('now') WHERE id = ?",
                (row_id, supersedes),
            )
        conn.commit()
        return row_id

    def search_memories(
        self,
        query_embedding: list[float],
        limit: int = 10,
        include_superseded: bool = False,
        project_id: str | None = None,
    ) -> list[dict]:
        conn = self._get_conn()
        serialized = sqlite_vec.serialize_float32(query_embedding)
        superseded_filter = "" if include_superseded else "AND m.superseded_by IS NULL"
        project_filter = "" if project_id is None else "AND m.project_id = ?"
        params: tuple[object, ...] = (serialized, limit)
        if project_id is not None:
            params += (project_id,)
        rows = conn.execute(
            f"""SELECT m.id, m.content, m.tags, m.project_id, m.memory_kind, m.summary, m.metadata_json,
                       m.source_session_id, m.source_message_id, m.supersedes, m.superseded_by,
                       m.created_at, m.updated_at, v.distance
                FROM memories_vec v
                JOIN memories m ON m.id = v.id
                WHERE v.embedding MATCH ? AND k = ? {superseded_filter} {project_filter}
                ORDER BY v.distance""",
            params,
        ).fetchall()
        results = []
        for row in rows:
            results.append(self._row_to_memory(row))
        return results

    def get_memory(self, memory_id: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def get_memory_by_source(
        self, source_session_id: str, source_message_id: str
    ) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT *
            FROM memories
            WHERE source_session_id = ? AND source_message_id = ?
            LIMIT 1
            """,
            (source_session_id, source_message_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

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

    def memory_count(self, include_superseded: bool = False) -> int:
        conn = self._get_conn()
        if include_superseded:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL").fetchone()[0]

    def list_memories(
        self,
        limit: int = 20,
        include_superseded: bool = False,
        project_id: str | None = None,
    ) -> list[dict]:
        conn = self._get_conn()
        superseded_filter = "" if include_superseded else "AND superseded_by IS NULL"
        project_filter = "" if project_id is None else "AND project_id = ?"
        params: tuple[object, ...] = ()
        if project_id is not None:
            params += (project_id,)
        params += (limit,)
        rows = conn.execute(
            f"""
            SELECT id, content, tags, project_id, memory_kind, summary, metadata_json,
                   source_session_id, source_message_id, supersedes, superseded_by,
                   created_at, updated_at
            FROM memories
            WHERE 1=1 {superseded_filter} {project_filter}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        results = []
        for row in rows:
            results.append(self._row_to_memory(row))
        return results

    # --- Docs ---

    def upsert_docs(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        self._validate_embedding_count(chunks, embeddings, kind="doc chunks")
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
            """SELECT d.file_path, d.chunk_index, d.content, d.indexed_at, v.distance
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

    def lexical_search_docs(self, terms: list[str], limit: int = 10) -> list[dict]:
        if not terms:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT file_path, chunk_index, content, indexed_at
            FROM docs
            """
        ).fetchall()
        results: list[dict] = []
        for row in rows:
            item = dict(row)
            haystack = f"{item['file_path']} {item['content']}".lower()
            matches = sum(1 for term in terms if term in haystack)
            if matches == 0:
                continue
            item["score"] = matches / len(terms)
            results.append(item)
        results.sort(key=lambda item: (-float(item["score"]), str(item["file_path"]), int(item["chunk_index"])))
        return results[:limit]

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
