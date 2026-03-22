from __future__ import annotations
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def forget(query: str | None = None, memory_id: str | None = None, session_id: str | None = None) -> str:
        """Remove a memory or session. Provide query (finds closest match), memory_id, or session_id."""
        from vibe_rag.server import _pg, _embedder

        if not _pg:
            return "Memory database unavailable."
        if not query and not memory_id and not session_id:
            return "Error: provide at least one of: query, memory_id, session_id."

        if session_id:
            count = await _pg.forget_session(session_id)
            return f"Deleted {count} chunks from session {session_id}."

        if memory_id:
            deleted = await _pg.forget_memory(memory_id=memory_id)
            if deleted:
                return f"Deleted memory: {deleted['content'][:200]}"
            return f"Memory {memory_id} not found."

        if query:
            if not _embedder:
                return "Embedding API unavailable."
            try:
                embeddings = await _embedder.embed_text([query])
            except Exception as e:
                return f"Embedding API unavailable: {e}"
            match = await _pg.find_closest_memory(query_embedding=embeddings[0])
            if match and match["score"] >= 0.7:
                return (
                    f"Found match (score={match['score']:.2f}, id={match['id']}): "
                    f"{match['content'][:300]}\n\n"
                    f"To delete, call forget with memory_id='{match['id']}'"
                )
            return "No matching memory found above similarity threshold (0.7)."
