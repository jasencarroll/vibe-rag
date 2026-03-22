from __future__ import annotations
from pathlib import Path
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def ingest_doc(path: str, project_id: str | None = None) -> str:
        """Index a local document (markdown or text) into memory for search."""
        from vibe_memory.server import _pg, _embedder, _config
        from vibe_memory.indexing.doc_chunker import chunk_markdown, chunk_plain_text

        if not _pg:
            return "Memory database unavailable."
        if not _embedder:
            return "Embedding API unavailable."

        file_path = Path(path)
        if not file_path.exists():
            return f"File not found: {path}"
        content = file_path.read_text(errors="replace")
        if not content.strip():
            return f"File is empty: {path}"

        if file_path.suffix in (".md", ".markdown"):
            chunks = chunk_markdown(content)
        else:
            chunks = chunk_plain_text(content)

        try:
            embeddings = await _embedder.embed_text(chunks)
        except Exception as e:
            return f"Embedding API unavailable: {e}"

        pid = project_id or (_config.project_id if _config else None)
        await _pg.upsert_doc_chunks(
            source=str(file_path.resolve()), chunks=chunks,
            embeddings=embeddings, project_id=pid,
        )
        return f"Ingested {file_path.name}: {len(chunks)} chunks"
