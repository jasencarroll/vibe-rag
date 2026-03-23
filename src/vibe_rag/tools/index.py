from __future__ import annotations

import time
from pathlib import Path

from vibe_rag.chunking import chunk_doc, collect_files_with_skips
from vibe_rag.constants import EXT_TO_LANG
from vibe_rag.indexing.code_chunker import chunk_code
from vibe_rag.indexing.embedder import ProgressCallback
from vibe_rag.server import _ensure_project_id, _get_db, _get_embedder, mcp
from vibe_rag.tools._helpers import (
    _content_hash,
    _current_git_head,
    _embed_sync_with_progress,
    _emit_progress,
    _failure,
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
) -> dict:
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
    code_hashes = db.get_file_hashes("code")
    doc_hashes = db.get_file_hashes("doc")
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

    warnings = [
        {
            "kind": "file_skipped",
            "path": _relative_to_project(Path(skip["path"]), project_root),
            "file_kind": skip["kind"],
            "reason": skip["reason"],
        }
        for skip in discovered_skips
    ]
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
            code_unchanged += 1
            continue

        language = EXT_TO_LANG.get(path.suffix)
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
    return _success(
        summary=summary,
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
    """Index project source files and docs for semantic search. Prefer this before grep when exploring a repo."""
    return _index_project_impl(paths)
