from __future__ import annotations
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_memory(query: str, scope: str = "all", project_only: bool = False, limit: int = 10) -> str:
        """Search across memories, past sessions, and docs. Scope: all, memories, sessions, docs."""
        from vibe_memory.server import _pg, _embedder, _config

        if not _pg:
            return "Memory database unavailable."
        if not _embedder:
            return "Embedding API unavailable."

        try:
            embeddings = await _embedder.embed_text([query])
        except Exception as e:
            return f"Embedding API unavailable: {e}"

        try:
            results = await _pg.search_memories(
                query_embedding=embeddings[0], scope=scope,
                project_id=_config.project_id if _config else None,
                project_only=project_only, limit=limit,
            )
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return f"No matching memories found. (debug: scope={scope}, project_id={_config.project_id if _config else None}, project_only={project_only}, embedding_dim={len(embeddings[0])}, pg_connected={_pg is not None})"

        output = []
        for r in results:
            score = f"{r.get('score', 0):.2f}"
            source = r.get("source_type", "unknown")
            content = r["content"][:500]
            line = f"[{source} | score={score}] {content}"
            if r.get("summary"):
                line = f"[{source} | {r['summary']}] {content}"
            output.append(line)
        return "\n\n---\n\n".join(output)
