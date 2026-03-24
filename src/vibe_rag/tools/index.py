"""MCP tool and implementation for project indexing.

Provides ``index_project`` (the MCP-exposed tool) and its internal
workhorse ``_index_project_impl``.  The implementation walks the project
tree, chunks code and documentation files, generates embeddings, and
upserts the results into the project SQLite-vec database.

Incremental indexing is the default: each file's content hash is stored
alongside its chunks, so only files that have actually changed since the
last run are re-chunked and re-embedded.  A full rebuild is triggered
automatically when the embedding profile (provider/model/dimensions)
changes, or on explicit request via ``force_full_rebuild``.
"""

from __future__ import annotations

import time
from pathlib import Path

from vibe_rag.chunking import chunk_doc, collect_files_with_skips
from vibe_rag.constants import EXT_TO_LANG
from vibe_rag.indexing.code_chunker import chunk_code
from vibe_rag.indexing.embedder import ProgressCallback, resolve_embedding_profile
from vibe_rag.server import _ensure_project_id, _get_db, _get_embedder, mcp
from vibe_rag.tools._helpers import (
    _content_hash,
    _current_git_head,
    _embedding_profile_state,
    _embed_sync_with_progress,
    _emit_progress,
    _failure,
    _format_embedding_profile,
    _index_metadata,
    _index_skip_summary,
    _normalize_paths,
    _relative_to_project,
    _success,
    _validate_embedding_count,
    INDEX_METADATA_KEY,
)


def _index_project_impl(
    paths: list[str] | str | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    force_full_rebuild: bool = False,
    rebuild_reason: str | None = None,
) -> dict:
    """Index (or re-index) the current project's code and documentation.

    This is the internal workhorse behind the ``index_project`` MCP tool.
    The pipeline proceeds through five stages:

    1. **Path resolution & file discovery** -- ``paths`` is normalised via
       ``_normalize_paths`` and ``collect_files_with_skips`` walks the
       resulting directory tree, separating code files from documentation
       files and recording any skipped paths.
    2. **Incremental change detection** -- For every discovered file the
       SHA-256 content hash is compared against the hash stored from the
       previous indexing run.  Files whose hash has not changed are counted
       as *unchanged* and skipped.  Stale entries (files that no longer
       exist on disk) are removed from the database.
    3. **Chunking** -- Changed code files are chunked with tree-sitter
       aware ``chunk_code``; changed doc files are chunked with
       ``chunk_doc`` (markdown-section-aware).
    4. **Embedding** -- Chunk text is sent to the configured embedding
       provider in batches via ``_embed_sync_with_progress``.  If
       embedding fails for any batch the entire indexing run is aborted
       and a failure dict is returned -- no partial writes occur because
       the DB upsert stage has not yet started.
    5. **DB upsert** -- Old chunks for each changed file are deleted and
       replaced with the freshly embedded chunks.  Index metadata
       (timestamp, git HEAD, embedding profile, file/chunk counts) is
       persisted so future runs can detect profile changes.

    A *full rebuild* (clearing all existing chunks and hashes first) is
    triggered when ``force_full_rebuild`` is ``True`` **or** when the
    current embedding profile differs from the one recorded in the index
    metadata, since vectors from different models are not comparable.

    Args:
        paths: Controls which paths to index.

            * ``None`` -- index the entire project root (auto-detected via
              git or cwd).
            * A single ``str`` -- index that one directory or file.
            * A ``list[str]`` -- index each listed path.

            Invalid paths cause an early ``"invalid_path"`` failure return.
        progress_callback: Optional callable invoked at each major phase
            transition (file discovery, chunking start/complete, embedding
            start, index complete) with a ``dict`` of phase-specific
            metrics.  Conforms to the ``ProgressCallback`` protocol from
            ``vibe_rag.indexing.embedder``.
        force_full_rebuild: When ``True``, wipe all existing chunks and
            file hashes before re-indexing.  Also set automatically when
            the embedding profile has changed.
        rebuild_reason: Human-readable reason attached to the rebuild
            warning when ``force_full_rebuild`` is ``True``.

    Returns:
        A dict with ``"status": "ok"`` on success or
        ``"status": "error"`` on failure.

        On success the dict contains:

        * ``summary`` -- human-readable one-line description of work done.
        * ``full_rebuild`` -- whether a full rebuild was performed.
        * ``project_id`` / ``project_root`` -- identifiers for the project.
        * ``elapsed_seconds`` -- wall-clock time for the run.
        * ``counts`` -- nested dict with ``code_files``, ``doc_files``,
          ``code_chunks``, ``doc_chunks``, ``code_unchanged``,
          ``doc_unchanged``, ``code_skipped``, ``doc_skipped``,
          ``indexed_code_chunks``, ``indexed_doc_chunks``.
        * ``warnings`` -- list of dicts for skipped files and rebuild
          notices.

        On failure the dict contains ``code`` (e.g.
        ``"invalid_path"``, ``"embedding_provider_unavailable"``,
        ``"no_files_found"``, ``"indexing_failed"``) and ``message``.
    """
    start = time.time()

    normalized = _normalize_paths(paths)
    if isinstance(normalized, str):
        return _failure("invalid_path", normalized.removeprefix("Error: ").strip(), paths=paths)
    root_paths, project_root = normalized

    try:
        embedder = _get_embedder()
    except RuntimeError as e:
        return _failure("embedding_provider_unavailable", str(e))

    code_files, doc_files, discovered_skips = collect_files_with_skips(root_paths)
    if not code_files and not doc_files:
        return _failure("no_files_found", "no files found to index", paths=paths)
    _emit_progress(
        progress_callback,
        phase="file_discovery_complete",
        code_file_total=len(code_files),
        doc_file_total=len(doc_files),
        code_skipped_total=sum(1 for skip in discovered_skips if skip["kind"] == "code"),
        doc_skipped_total=sum(1 for skip in discovered_skips if skip["kind"] == "doc"),
        project_root=str(project_root),
    )

    db = _get_db()
    warnings: list[dict] = []
    current_profile = resolve_embedding_profile()
    metadata_state = _index_metadata(db)
    profile_state = _embedding_profile_state(db, metadata_state.get("metadata"))
    profile_requires_rebuild = bool(profile_state.get("is_incompatible"))
    force_full_rebuild = force_full_rebuild or profile_requires_rebuild
    code_hashes = db.get_file_hashes("code")
    doc_hashes = db.get_file_hashes("doc")
    if force_full_rebuild:
        if db.code_chunk_count() > 0:
            db.clear_code()
        if db.doc_count() > 0:
            db.clear_docs()
        if code_hashes:
            db.delete_file_hashes(list(code_hashes), kind="code")
        if doc_hashes:
            db.delete_file_hashes(list(doc_hashes), kind="doc")
        warnings.append(
            {
                "kind": "full_rebuild_required" if profile_requires_rebuild else "full_rebuild_requested",
                "detail": (
                    "full rebuild required due to embedding profile change "
                    f"({_format_embedding_profile(profile_state.get('indexed_profile'))}"
                    f" -> {_format_embedding_profile(profile_state.get('current_profile'))})"
                )
                if profile_requires_rebuild
                else "full rebuild requested by operator",
                "reason": rebuild_reason or ("embedding_profile_changed" if profile_requires_rebuild else "manual_request"),
            }
        )
        code_hashes = {}
        doc_hashes = {}
    if code_hashes and db.code_chunk_count() == 0:
        db.delete_file_hashes(list(code_hashes), kind="code")
        code_hashes = {}
    if doc_hashes and db.doc_count() == 0:
        db.delete_file_hashes(list(doc_hashes), kind="doc")
        doc_hashes = {}
    current_code_paths = {_relative_to_project(path, project_root) for path in code_files}
    current_doc_paths = {_relative_to_project(path, project_root) for path in doc_files}

    for stale in sorted(set(code_hashes) - current_code_paths):
        db.delete_file_chunks(stale, kind="code")
    for stale in sorted(set(doc_hashes) - current_doc_paths):
        db.delete_file_chunks(stale, kind="doc")

    warnings.extend(
        [
        {
            "kind": "file_skipped",
            "path": _relative_to_project(Path(skip["path"]), project_root),
            "file_kind": skip["kind"],
            "reason": skip["reason"],
        }
        for skip in discovered_skips
        ]
    )
    code_chunks: list[dict] = []
    code_embeddings_input: list[str] = []
    code_updates: list[tuple[str, str, list[dict]]] = []
    doc_chunks: list[dict] = []
    doc_embeddings_input: list[str] = []
    doc_updates: list[tuple[str, str, list[dict]]] = []
    code_unchanged = 0
    doc_unchanged = 0
    code_skipped = sum(1 for skip in discovered_skips if skip["kind"] == "code")
    doc_skipped = sum(1 for skip in discovered_skips if skip["kind"] == "doc")
    _emit_progress(progress_callback, phase="code_chunking_start", file_total=len(code_files))

    for path in code_files:
        rel_path = _relative_to_project(path, project_root)
        try:
            content = path.read_text(errors="replace")
        except Exception as exc:
            code_skipped += 1
            warnings.append(
                {
                    "kind": "file_skipped",
                    "path": rel_path,
                    "file_kind": "code",
                    "reason": f"read failed: {exc}",
                }
            )
            continue
        digest = _content_hash(content)
        if code_hashes.get(rel_path) == digest:
            if language := EXT_TO_LANG.get(path.suffix.lower()):
                db.backfill_code_chunk_language(rel_path, language)
            code_unchanged += 1
            continue

        language = EXT_TO_LANG.get(path.suffix.lower())
        file_chunks = chunk_code(content, rel_path, language)
        code_updates.append((rel_path, digest, file_chunks))
        code_chunks.extend(file_chunks)
        code_embeddings_input.extend(chunk["content"] for chunk in file_chunks)

    _emit_progress(
        progress_callback,
        phase="code_chunking_complete",
        file_total=len(code_files),
        chunk_total=len(code_chunks),
        unchanged_total=code_unchanged,
        skipped_total=code_skipped,
    )
    _emit_progress(progress_callback, phase="doc_chunking_start", file_total=len(doc_files))

    for path in doc_files:
        rel_path = _relative_to_project(path, project_root)
        try:
            content = path.read_text(errors="replace")
        except Exception as exc:
            doc_skipped += 1
            warnings.append(
                {
                    "kind": "file_skipped",
                    "path": rel_path,
                    "file_kind": "doc",
                    "reason": f"read failed: {exc}",
                }
            )
            continue
        digest = _content_hash(content)
        if doc_hashes.get(rel_path) == digest:
            doc_unchanged += 1
            continue

        file_chunks = chunk_doc(content, rel_path)
        doc_updates.append((rel_path, digest, file_chunks))
        doc_chunks.extend(file_chunks)
        doc_embeddings_input.extend(chunk["content"] for chunk in file_chunks)

    _emit_progress(
        progress_callback,
        phase="doc_chunking_complete",
        file_total=len(doc_files),
        chunk_total=len(doc_chunks),
        unchanged_total=doc_unchanged,
        skipped_total=doc_skipped,
    )

    try:
        code_embeddings: list[list[float]] = []
        if code_chunks:
            _emit_progress(
                progress_callback,
                phase="code_embedding_start",
                chunk_total=len(code_chunks),
            )
            code_embeddings = _embed_sync_with_progress(
                embedder.embed_code_sync,
                code_embeddings_input,
                progress_callback=progress_callback,
            )
            _validate_embedding_count(code_chunks, code_embeddings, kind="code chunks")
        doc_embeddings: list[list[float]] = []
        if doc_chunks:
            _emit_progress(
                progress_callback,
                phase="doc_embedding_start",
                chunk_total=len(doc_chunks),
            )
            doc_embeddings = _embed_sync_with_progress(
                embedder.embed_text_sync,
                doc_embeddings_input,
                progress_callback=progress_callback,
            )
            _validate_embedding_count(doc_chunks, doc_embeddings, kind="doc chunks")
    except Exception as e:
        return _failure("indexing_failed", f"indexing failed: {e}")

    code_offset = 0
    for rel_path, digest, file_chunks in code_updates:
        db.delete_file_chunks(rel_path, kind="code")
        file_embedding_count = len(file_chunks)
        if file_embedding_count:
            db.upsert_chunks(
                file_chunks,
                code_embeddings[code_offset:code_offset + file_embedding_count],
            )
        db.set_file_hash(rel_path, digest, "code")
        code_offset += file_embedding_count

    doc_offset = 0
    for rel_path, digest, file_chunks in doc_updates:
        db.delete_file_chunks(rel_path, kind="doc")
        file_embedding_count = len(file_chunks)
        if file_embedding_count:
            db.upsert_docs(
                file_chunks,
                doc_embeddings[doc_offset:doc_offset + file_embedding_count],
            )
        db.set_file_hash(rel_path, digest, "doc")
        doc_offset += file_embedding_count

    db.set_setting_json(
        INDEX_METADATA_KEY,
        {
            "project_id": _ensure_project_id(),
            "project_root": str(project_root),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": _current_git_head(project_root),
            "embedding_profile": current_profile,
            "code_file_count": len(current_code_paths),
            "doc_file_count": len(current_doc_paths),
            "code_chunk_count": db.code_chunk_count(),
            "doc_chunk_count": db.doc_count(),
        },
    )

    elapsed = time.time() - start
    _emit_progress(
        progress_callback,
        phase="index_complete",
        elapsed_seconds=round(elapsed, 1),
        code_file_total=len(code_files),
        doc_file_total=len(doc_files),
        code_chunk_total=db.code_chunk_count(),
        doc_chunk_total=db.doc_count(),
    )
    summary = (
        f"Indexed {len(code_files)} code files ({len(code_chunks)} chunks, {code_unchanged} unchanged), "
        f"{len(doc_files)} docs ({len(doc_chunks)} chunks, {doc_unchanged} unchanged) in {elapsed:.1f}s"
        f"{_index_skip_summary(code_skipped, doc_skipped)}"
    )
    if force_full_rebuild:
        summary = f"Rebuilt index with {_format_embedding_profile(current_profile)}. {summary}"
    return _success(
        summary=summary,
        full_rebuild=force_full_rebuild,
        project_id=_ensure_project_id(),
        project_root=str(project_root),
        elapsed_seconds=round(elapsed, 1),
        counts={
            "code_files": len(code_files),
            "doc_files": len(doc_files),
            "code_chunks": len(code_chunks),
            "doc_chunks": len(doc_chunks),
            "code_unchanged": code_unchanged,
            "doc_unchanged": doc_unchanged,
            "code_skipped": code_skipped,
            "doc_skipped": doc_skipped,
            "indexed_code_chunks": db.code_chunk_count(),
            "indexed_doc_chunks": db.doc_count(),
        },
        warnings=warnings,
    )


@mcp.tool()
def index_project(paths: list[str] | str | None = None) -> dict:
    """Index or re-index code and docs in the current project for semantic search.

    Run after major file changes or when search returns stale results.
    Indexing is incremental by default: only files whose content has changed
    (detected via SHA-256 hash) are re-chunked and re-embedded.  A full
    rebuild happens automatically when the embedding profile changes.

    Args:
        paths: What to index.

            * ``None`` (default) -- index the entire project root.
            * A single path string -- index that directory or file.
            * A list of path strings -- index each listed path.

    Returns:
        A dict containing ``"status"`` (``"ok"`` or ``"error"``),
        ``summary``, ``elapsed_seconds``, ``counts`` (code/doc files,
        chunks, unchanged, skipped), and ``warnings`` (skipped files,
        rebuild notices).  See ``_index_project_impl`` for full detail.
    """
    return _index_project_impl(paths)
