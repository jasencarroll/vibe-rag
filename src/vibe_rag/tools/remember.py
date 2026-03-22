from __future__ import annotations
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def remember(content: str, tags: list[str] | None = None, global_memory: bool = False) -> str:
        """Store a memory. Set global_memory=True to make it available across all projects. (Named global_memory instead of global to avoid Python keyword conflict.)"""
        from vibe_rag.server import _pg, _embedder, _config

        if not _pg:
            return "Memory database unavailable."
        if not _embedder:
            return "Embedding API unavailable."

        try:
            embeddings = await _embedder.embed_text([content])
        except Exception as e:
            return f"Embedding API unavailable: {e}"

        project_id = None if global_memory else (_config.project_id if _config else None)
        memory_id = await _pg.remember(
            content=content, embedding=embeddings[0], project_id=project_id, tags=tags or [],
        )
        scope = "global" if global_memory else f"project:{project_id}"
        return f"Remembered ({scope}): {content[:100]}... [id={memory_id}]"
