"""SQLite + sqlite-vec vector database wrapper for vibe-rag.

Provides :class:`SqliteVecDB`, the single storage abstraction used by
both the **project index** (code chunks, doc chunks, file hashes -- stored
at ``.vibe-rag/index.db``) and the **user memory** store (memories -- stored
at ``~/.vibe-rag/memory.db``).  The wrapper combines regular SQLite tables for
metadata with ``sqlite-vec`` virtual tables for cosine-distance vector
search.

Consumed by:
    - ``server.py`` (lazy-initialised singletons ``_project_db`` /
      ``_user_db``)
    - ``tools.py``  (every MCP tool that touches storage)
    - ``cli.py``    (``status``, ``doctor``, ``reindex``)

Thread-safety model:
    A single :class:`sqlite3.Connection` is created lazily on first use
    and reused for the lifetime of the instance.  ``server.py`` protects
    the singletons with a :class:`threading.Lock`, so concurrent MCP
    requests are serialised at the server layer -- the DB object itself
    is **not** thread-safe.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec
from vibe_rag.constants import EXT_TO_LANG
from vibe_rag.types import CodeChunk, DocChunk, MemoryRow, RankedCodeResult, RankedDocResult


class SqliteVecDB:
    """SQLite + sqlite-vec storage backend for code, docs, and memories.

    Wraps a single ``.db`` file that contains:

    * **code_chunks / code_chunks_vec** -- indexed source-code fragments
    * **docs / docs_vec** -- indexed documentation fragments
    * **memories / memories_vec** -- user/project memories with
      supersede-chain support
    * **file_hashes** -- content hashes for incremental re-indexing
    * **settings** -- key/value pairs (embedding dimensions, etc.)

    The same class backs both the per-project DB and the global user-memory
    DB; which tables are actually populated depends on the caller.

    Connection lifecycle:
        Created lazily by :meth:`_get_conn` on the first call that needs
        the database.  WAL journal mode and ``NORMAL`` synchronous are set
        once.  The ``sqlite-vec`` extension is loaded at connection time.

    Parameters
    ----------
    path : Path
        Filesystem path to the ``.db`` file (created if absent).
    embedding_dimensions : int
        Vector width for all ``vec0`` virtual tables.  Must match the
        dimensionality of the embedding provider in use; a mismatch
        against an existing DB raises :class:`RuntimeError` during
        :meth:`initialize`.
    """

    def __init__(self, path: Path, embedding_dimensions: int = 2560):
        self._path = path
        self._embedding_dimensions = embedding_dimensions
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Return the shared connection, creating it on first call.

        On creation: enables WAL mode, loads the ``sqlite-vec`` extension,
        and sets ``row_factory = sqlite3.Row`` for dict-style access.
        """
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
        """Create all tables/indexes and validate embedding dimensions.

        Idempotent -- safe to call on every startup.  Runs a single
        ``executescript`` that creates (if not exists):

        * ``settings`` -- key/value config store
        * ``code_chunks`` + ``code_chunks_vec`` -- code index
        * ``memories`` + ``memories_vec`` -- memory store
        * ``docs`` + ``docs_vec`` -- documentation index
        * ``file_hashes`` -- incremental-indexing content hashes
        * Indexes on ``memories(source_session_id, source_message_id)``,
          ``memories(project_id)``, ``memories(superseded_by)``

        After schema creation:

        1. ``_ensure_dimensions`` -- records or validates embedding
           dimensions in the ``settings`` table.  Raises
           :class:`RuntimeError` if the DB was created with a different
           dimension than the one requested.
        2. ``_ensure_memory_columns`` -- migrates the ``memories`` table
           by adding any columns introduced after the initial schema
           (e.g. ``project_id``, ``memory_kind``, ``supersedes``).

        Raises
        ------
        RuntimeError
            If ``embedding_dimensions`` does not match the value stored
            in an existing database (dimension mismatch).
        """
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

            CREATE INDEX IF NOT EXISTS idx_memories_source
                ON memories(source_session_id, source_message_id);
            CREATE INDEX IF NOT EXISTS idx_memories_project
                ON memories(project_id);
            CREATE INDEX IF NOT EXISTS idx_memories_superseded
                ON memories(superseded_by);
            """
        )
        self._ensure_dimensions(conn)
        self._ensure_memory_columns(conn)
        conn.commit()

    def get_setting(self, key: str) -> str | None:
        """Fetch a single value from the ``settings`` table.

        Executes ``SELECT value FROM settings WHERE key = ?``.

        Returns ``None`` if the key does not exist.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        """Store a key/value pair in the ``settings`` table.

        Executes ``INSERT OR REPLACE INTO settings``.  Commits immediately.
        """
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def get_setting_json(self, key: str) -> dict | None:
        """Retrieve a JSON-object setting, discarding any error status.

        Convenience wrapper around :meth:`get_setting_json_status`.
        Returns the parsed ``dict``, or ``None`` on missing/invalid data.
        """
        parsed, _ = self.get_setting_json_status(key)
        return parsed

    def get_setting_json_status(self, key: str) -> tuple[dict | None, str | None]:
        """Retrieve a JSON-object setting with an error description.

        Returns ``(parsed_dict, None)`` on success, or
        ``(None, error_message)`` if the stored value is missing, not
        valid JSON, or not a JSON object.
        """
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
        """Serialize *value* as JSON and store it via :meth:`set_setting`."""
        self.set_setting(key, json.dumps(value))

    def _ensure_dimensions(self, conn: sqlite3.Connection) -> None:
        """Record or validate embedding dimensions in the settings table.

        On a fresh DB, stores ``self._embedding_dimensions``.  On an
        existing DB, compares the stored value (or introspects the
        ``code_chunks_vec`` schema) and raises :class:`RuntimeError` on
        mismatch.
        """
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
        """Migrate the ``memories`` table by adding any missing columns.

        Uses ``PRAGMA table_info(memories)`` to discover the current
        schema, then runs ``ALTER TABLE ADD COLUMN`` for each column not
        yet present (e.g. ``project_id``, ``memory_kind``, ``supersedes``).
        """
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
        """Parse a ``metadata_json`` column value, defaulting to ``{}``."""
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _int_or_none(self, value: object) -> int | None:
        """Coerce *value* to ``int``, returning ``None`` on failure."""
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _row_to_memory(self, row) -> MemoryRow:
        """Convert a raw ``sqlite3.Row`` into a :class:`MemoryRow` dict.

        Decodes ``metadata_json`` into ``metadata``, and coerces
        ``supersedes`` / ``superseded_by`` to ``int | None``.
        """
        result: MemoryRow = dict(row)
        result["metadata"] = self._decode_metadata_json(result.pop("metadata_json", None))
        result["id"] = int(result["id"])
        result["supersedes"] = self._int_or_none(result.get("supersedes"))
        result["superseded_by"] = self._int_or_none(result.get("superseded_by"))
        return result

    def _validate_embedding_count(
        self, items: list[CodeChunk] | list[DocChunk], embeddings: list[list[float]], *, kind: str
    ) -> None:
        """Raise :class:`RuntimeError` if *items* and *embeddings* differ in length."""
        if len(items) != len(embeddings):
            raise RuntimeError(
                f"Embedding count mismatch for {kind}: {len(items)} items but {len(embeddings)} embeddings"
            )

    def upsert_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Insert or replace code chunks and their vector embeddings.

        For each chunk, executes:
        * ``INSERT OR REPLACE INTO code_chunks`` (keyed on
          ``file_path, chunk_index``)
        * ``INSERT OR REPLACE INTO code_chunks_vec`` with the serialised
          embedding

        Parameters
        ----------
        chunks : list[CodeChunk]
            Code fragments with file_path, chunk_index, content, language,
            symbol, start_line, end_line.
        embeddings : list[list[float]]
            One embedding vector per chunk (must match length of *chunks*).

        Raises
        ------
        RuntimeError
            If ``len(chunks) != len(embeddings)``.
        """
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

    def search_code(
        self, query_embedding: list[float], limit: int = 10, language: str | None = None
    ) -> list[RankedCodeResult]:
        """Vector-similarity search over code chunks.

        Queries ``code_chunks_vec`` with ``MATCH`` / ``k`` and joins back
        to ``code_chunks`` for metadata.  When *language* is provided, the
        query over-fetches by 3x and filters post-join (sqlite-vec does
        not support pre-filter on joined columns).

        Parameters
        ----------
        query_embedding : list[float]
            The query vector (same dimensionality as stored embeddings).
        limit : int
            Maximum results to return (default 10).
        language : str | None
            Optional language filter (e.g. ``"python"``).

        Returns
        -------
        list[RankedCodeResult]
            Dicts with code chunk fields plus ``distance`` (lower is
            more similar).
        """
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
        """Delete all rows from ``code_chunks`` and ``code_chunks_vec``."""
        conn = self._get_conn()
        conn.executescript("DELETE FROM code_chunks_vec; DELETE FROM code_chunks;")
        conn.commit()

    def code_chunk_count(self) -> int:
        """Return the total number of rows in ``code_chunks``."""
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]

    def lexical_search_code(self, terms: list[str], limit: int = 10) -> list[RankedCodeResult]:
        """Brute-force keyword search over all code chunks.

        Loads every row from ``code_chunks``, scores each by the fraction
        of *terms* found (case-insensitive) in ``file_path + content``,
        and returns the top *limit* results sorted by score descending,
        then by ``file_path`` and ``chunk_index`` for deterministic order.

        Parameters
        ----------
        terms : list[str]
            Lowercased search terms.  Empty list returns ``[]``.
        limit : int
            Maximum results (default 10).

        Returns
        -------
        list[RankedCodeResult]
            Dicts with code chunk fields plus ``score`` (0.0 -- 1.0).
        """
        if not terms:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT file_path, chunk_index, content, language, symbol, start_line, end_line, indexed_at
            FROM code_chunks
            """
        ).fetchall()
        results: list[RankedCodeResult] = []
        for row in rows:
            item: RankedCodeResult = dict(row)
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
        """Store a simple memory (convenience wrapper).

        Delegates to :meth:`remember_structured` with ``memory_kind="note"``
        and ``summary`` set to the first 200 characters of *content*.

        Returns the new memory row id.
        """
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
        update_superseded: bool = True,
    ) -> int:
        """Insert a structured memory with its embedding vector.

        Executes ``INSERT INTO memories`` followed by
        ``INSERT INTO memories_vec``.  If *supersedes* is set and
        *update_superseded* is ``True``, also marks the old memory's
        ``superseded_by`` to point to the new row.

        Parameters
        ----------
        summary : str
            Short description of the memory.
        content : str
            Full memory text.
        embedding : list[float]
            Vector embedding of the content.
        tags : str
            Comma-separated tag string.
        project_id : str | None
            Scopes the memory to a project (None = global/user).
        memory_kind : str
            Classification (note/decision/constraint/todo/fact/summary).
        metadata : dict | None
            Arbitrary JSON-serialisable metadata.
        source_session_id / source_message_id : str | None
            Provenance tracking for auto-saved memories.
        supersedes : int | None
            Row id of a previous memory this one replaces.
        update_superseded : bool
            If True and *supersedes* is set, updates the old row's
            ``superseded_by`` column.

        Returns
        -------
        int
            The ``id`` (``lastrowid``) of the newly inserted memory.
        """
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
        if supersedes is not None and update_superseded:
            conn.execute(
                "UPDATE memories SET superseded_by = ?, updated_at = datetime('now') WHERE id = ?",
                (row_id, supersedes),
            )
        conn.commit()
        return row_id

    def set_memory_superseded_by(self, memory_id: int, superseded_by: int) -> bool:
        """Mark a memory as superseded by another.

        Executes ``UPDATE memories SET superseded_by = ?, updated_at = ...
        WHERE id = ?``.

        Returns ``True`` if the row was found and updated, ``False``
        otherwise.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = datetime('now') WHERE id = ?",
            (superseded_by, memory_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def search_memories(
        self,
        query_embedding: list[float],
        limit: int = 10,
        include_superseded: bool = False,
        project_id: str | None = None,
    ) -> list[MemoryRow]:
        """Vector-similarity search over memories.

        Queries ``memories_vec`` with ``MATCH`` / ``k`` and joins to
        ``memories``.  By default excludes rows where
        ``superseded_by IS NOT NULL`` (i.e. only returns current
        memories).

        Parameters
        ----------
        query_embedding : list[float]
            The query vector.
        limit : int
            Maximum results (default 10).
        include_superseded : bool
            If True, includes memories that have been superseded.
        project_id : str | None
            Filter to a specific project (None = no filter).

        Returns
        -------
        list[MemoryRow]
            Dicts with memory fields plus ``distance``.
        """
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

    def get_memory(self, memory_id: int) -> MemoryRow | None:
        """Fetch a single memory by primary key.

        Executes ``SELECT * FROM memories WHERE id = ?``.
        Returns ``None`` if no row matches.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def get_memory_by_source(
        self, source_session_id: str, source_message_id: str
    ) -> MemoryRow | None:
        """Look up a memory by its session/message provenance.

        Executes ``SELECT * FROM memories WHERE source_session_id = ?
        AND source_message_id = ? LIMIT 1``.

        Returns ``None`` if no match is found.
        """
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
        """Delete a memory and its vector embedding.

        Executes ``DELETE FROM memories WHERE id = ?`` and
        ``DELETE FROM memories_vec WHERE id = ?``.

        Returns the deleted memory's ``content``, or ``None`` if
        *memory_id* did not exist.
        """
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
        """Return the number of memories.

        By default counts only current (non-superseded) rows via
        ``SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL``.
        Set *include_superseded* to count all rows.
        """
        conn = self._get_conn()
        if include_superseded:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL").fetchone()[0]

    def list_memories(
        self,
        limit: int = 20,
        include_superseded: bool = False,
        project_id: str | None = None,
        updated_since: str | None = None,
    ) -> list[MemoryRow]:
        """List memories ordered by ``updated_at DESC, id DESC``.

        Queries ``SELECT ... FROM memories`` with optional filters for
        superseded status, project scope, and recency.

        Parameters
        ----------
        limit : int
            Maximum rows (default 20).
        include_superseded : bool
            Include memories that have been superseded (default False).
        project_id : str | None
            Filter to a specific project.
        updated_since : str | None
            ISO-8601 datetime cutoff for ``updated_at >= ?``.

        Returns
        -------
        list[MemoryRow]
            Memory dicts (no ``distance`` field -- this is not a vector
            search).
        """
        conn = self._get_conn()
        superseded_filter = "" if include_superseded else "AND superseded_by IS NULL"
        project_filter = "" if project_id is None else "AND project_id = ?"
        updated_since_filter = "" if updated_since is None else "AND updated_at >= ?"
        params: tuple[object, ...] = ()
        if project_id is not None:
            params += (project_id,)
        if updated_since is not None:
            params += (updated_since,)
        params += (limit,)
        rows = conn.execute(
            f"""
            SELECT id, content, tags, project_id, memory_kind, summary, metadata_json,
                   source_session_id, source_message_id, supersedes, superseded_by,
                   created_at, updated_at
            FROM memories
            WHERE 1=1 {superseded_filter} {project_filter} {updated_since_filter}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        results = []
        for row in rows:
            results.append(self._row_to_memory(row))
        return results

    def update_memory(
        self,
        memory_id: int,
        embedding: list[float] | None = None,
        content: str | None = None,
        summary: str | None = None,
        tags: str | None = None,
        memory_kind: str | None = None,
        metadata: dict | None = None,
        source_session_id: str | None = ...,  # type: ignore[assignment]
        source_message_id: str | None = ...,  # type: ignore[assignment]
    ) -> bool:
        """Update an existing memory in place.

        Builds a dynamic ``UPDATE memories SET ... WHERE id = ?`` from
        the non-sentinel keyword arguments.  Uses ``Ellipsis`` (``...``)
        as the default sentinel for ``source_session_id`` and
        ``source_message_id`` so that ``None`` can be passed to
        explicitly clear those fields.

        If *embedding* is provided, replaces the vector in
        ``memories_vec`` (delete + insert).

        Returns ``False`` if *memory_id* does not exist, ``True``
        otherwise.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return False

        sets: list[str] = []
        params: list[object] = []

        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if summary is not None:
            sets.append("summary = ?")
            params.append(summary)
        if tags is not None:
            sets.append("tags = ?")
            params.append(tags)
        if memory_kind is not None:
            sets.append("memory_kind = ?")
            params.append(memory_kind)
        if metadata is not None:
            sets.append("metadata_json = ?")
            params.append(json.dumps(metadata))
        if source_session_id is not ...:
            sets.append("source_session_id = ?")
            params.append(source_session_id)
        if source_message_id is not ...:
            sets.append("source_message_id = ?")
            params.append(source_message_id)

        if sets:
            sets.append("updated_at = datetime('now')")
            params.append(memory_id)
            conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
                params,
            )

        if embedding is not None:
            conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                (memory_id, sqlite_vec.serialize_float32(embedding)),
            )

        conn.commit()
        return True

    # --- Docs ---

    def upsert_docs(self, chunks: list[DocChunk], embeddings: list[list[float]]) -> None:
        """Insert or replace documentation chunks and their embeddings.

        For each chunk, executes ``INSERT OR REPLACE INTO docs`` (keyed
        on ``file_path, chunk_index``) then
        ``INSERT OR REPLACE INTO docs_vec``.

        Parameters
        ----------
        chunks : list[DocChunk]
            Doc fragments with file_path, chunk_index, content.
        embeddings : list[list[float]]
            One embedding per chunk (must match length).

        Raises
        ------
        RuntimeError
            If ``len(chunks) != len(embeddings)``.
        """
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

    def search_docs(self, query_embedding: list[float], limit: int = 10) -> list[RankedDocResult]:
        """Vector-similarity search over documentation chunks.

        Queries ``docs_vec`` with ``MATCH`` / ``k`` and joins to ``docs``.

        Returns up to *limit* :class:`RankedDocResult` dicts with a
        ``distance`` field (lower is more similar).
        """
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
        """Delete all rows from ``docs`` and ``docs_vec``."""
        conn = self._get_conn()
        conn.executescript("DELETE FROM docs_vec; DELETE FROM docs;")
        conn.commit()

    def doc_count(self) -> int:
        """Return the total number of rows in ``docs``."""
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def lexical_search_docs(self, terms: list[str], limit: int = 10) -> list[RankedDocResult]:
        """Brute-force keyword search over all doc chunks.

        Identical approach to :meth:`lexical_search_code`: loads every
        ``docs`` row, scores by fraction of *terms* matched in
        ``file_path + content`` (case-insensitive), returns top *limit*
        sorted by score descending then deterministically by path/index.

        Returns ``[]`` when *terms* is empty.
        """
        if not terms:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT file_path, chunk_index, content, indexed_at
            FROM docs
            """
        ).fetchall()
        results: list[RankedDocResult] = []
        for row in rows:
            item: RankedDocResult = dict(row)
            haystack = f"{item['file_path']} {item['content']}".lower()
            matches = sum(1 for term in terms if term in haystack)
            if matches == 0:
                continue
            item["score"] = matches / len(terms)
            results.append(item)
        results.sort(key=lambda item: (-float(item["score"]), str(item["file_path"]), int(item["chunk_index"])))
        return results[:limit]

    def language_stats(self) -> dict[str, int]:
        """Return ``{language: chunk_count}`` across all code chunks.

        Reads ``file_path`` and ``language`` from ``code_chunks``.  When
        ``language`` is NULL/empty, infers it from the file extension
        via :data:`EXT_TO_LANG` (falls back to ``"unknown"``).
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_path, language FROM code_chunks"
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            language = row["language"]
            if not language:
                language = EXT_TO_LANG.get(Path(str(row["file_path"])).suffix.lower(), "unknown")
            counts[str(language)] = counts.get(str(language), 0) + 1
        return counts

    def backfill_code_chunk_language(self, file_path: str, language: str) -> int:
        """Set ``language`` on code chunks that lack one for a given file.

        Executes ``UPDATE code_chunks SET language = ? WHERE file_path = ?
        AND (language IS NULL OR language = '')``.

        Returns the number of rows updated.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            UPDATE code_chunks
            SET language = ?
            WHERE file_path = ?
              AND (language IS NULL OR language = '')
            """,
            (language, file_path),
        )
        conn.commit()
        return int(cursor.rowcount)

    # --- File hashes (incremental indexing) ---

    def get_file_hashes(self, kind: str) -> dict[str, str]:
        """Return ``{file_path: content_hash}`` for all files of *kind*.

        Executes ``SELECT file_path, content_hash FROM file_hashes
        WHERE kind = ?``.  *kind* is typically ``"code"`` or ``"doc"``.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_path, content_hash FROM file_hashes WHERE kind = ?",
            (kind,),
        ).fetchall()
        return {row["file_path"]: row["content_hash"] for row in rows}

    def set_file_hash(self, file_path: str, content_hash: str, kind: str) -> None:
        """Record or update a file's content hash for incremental indexing.

        Executes ``INSERT OR REPLACE INTO file_hashes``.  Commits
        immediately.
        """
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, content_hash, kind) VALUES (?, ?, ?)",
            (file_path, content_hash, kind),
        )
        conn.commit()

    def delete_file_chunks(self, file_path: str, kind: str = "code") -> None:
        """Remove all chunks (and their vectors) for a single file.

        When ``kind="code"``, deletes from ``code_chunks`` +
        ``code_chunks_vec``.  When ``kind="doc"`` (or anything else),
        deletes from ``docs`` + ``docs_vec``.  Also removes the
        corresponding ``file_hashes`` row.

        Parameters
        ----------
        file_path : str
            The indexed file path to purge.
        kind : str
            ``"code"`` or ``"doc"`` (default ``"code"``).
        """
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
        """Remove file-hash entries for the given paths and *kind*.

        Executes ``DELETE FROM file_hashes WHERE file_path = ? AND
        kind = ?`` for each path.  Does **not** delete the associated
        chunks -- use :meth:`delete_file_chunks` for that.
        """
        conn = self._get_conn()
        for fp in file_paths:
            conn.execute(
                "DELETE FROM file_hashes WHERE file_path = ? AND kind = ?",
                (fp, kind),
            )
        conn.commit()

    # --- Compat aliases ---

    def search(
        self, query_embedding: list[float], limit: int = 10, language: str | None = None
    ) -> list[RankedCodeResult]:
        """Compat alias for :meth:`search_code`."""
        return self.search_code(query_embedding, limit, language)

    def clear(self) -> None:
        """Compat alias for :meth:`clear_code`."""
        self.clear_code()

    def chunk_count(self) -> int:
        """Compat alias for :meth:`code_chunk_count`."""
        return self.code_chunk_count()

    def list_tables(self) -> list[str]:
        """Return the names of all tables (including virtual) in the DB.

        Queries ``sqlite_master`` for entries of type ``table`` or
        ``virtual table``.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
        return [row["name"] for row in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection if open.

        Safe to call multiple times.  After closing, the next call to
        any data method will re-create the connection via
        :meth:`_get_conn`.
        """
        if self._conn:
            self._conn.close()
            self._conn = None
