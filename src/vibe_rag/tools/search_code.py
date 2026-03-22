from __future__ import annotations
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_code(query: str, limit: int = 10, language: str | None = None) -> str:
        """Search project code by semantic meaning. Returns matching code chunks with file paths and line numbers."""
        from vibe_rag.server import _sqlite, _embedder

        if not _sqlite:
            return "No code index found. Run index_project first."
        if not _embedder:
            return "Embedding API unavailable."

        try:
            embeddings = await _embedder.embed_code([query])
        except Exception as e:
            return f"Embedding API unavailable: {e}"

        results = _sqlite.search(embeddings[0], limit=limit, language=language)
        if not results:
            return "No matching code found."

        output = []
        for r in results:
            header = f"**{r['file_path']}:{r['start_line']}-{r['end_line']}**"
            if r.get("symbol"):
                header += f" (`{r['symbol']}`)"
            output.append(f"{header}\n```\n{r['content']}\n```")
        return "\n\n".join(output)
