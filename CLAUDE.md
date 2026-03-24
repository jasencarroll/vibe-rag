# CLAUDE.md

## Commands

```bash
uv run pytest                              # run all tests
uv run pytest tests/test_tools.py          # test core MCP tools
uv run pytest tests/test_cli.py            # test CLI commands
uv run pytest tests/test_tools.py -k test_remember  # single test
uv build                                   # build wheel
vibe-rag status                            # check index/memory counts
vibe-rag doctor                            # full health check
vibe-rag reindex                           # re-index current project
vibe-rag hook-session-start vibe            # render session-start hook
```

## Architecture

```
src/vibe_rag/
  server.py        — FastMCP server, lazy-init for DBs + embedder
  tools/           — 15 MCP tool definitions split across submodules (index, memory, search, session, status); unified search/remember with scope params, update_memory for in-place edits
  cli.py           — Click CLI: init, status, doctor, reindex, serve, hook-session-start
  hook_bridge.py   — session-start hook renderer for codex/claude/gemini/vibe formats
  chunking.py      — doc chunking (markdown section-aware + plain text)
  constants.py     — file extensions, skip dirs, chunk sizes
  types.py         — shared TypedDict and type definitions
  db/sqlite.py     — SqliteVecDB wrapper (sqlite-vec for vector search)
  indexing/
    embedder.py    — OpenRouter embedding (default model: pplx-embed-v1-4b)
    code_chunker.py — tree-sitter syntax-aware code chunking
  templates/       — AGENTS.md template for `vibe-rag init`
  template_bundle/ — vibe/codex/claude/gemini config scaffolding for `vibe-rag init`
```

## Key Patterns

- **Dual-DB design**: project index at `.vibe/index.db`, user memory at `~/.vibe/memory.db`
- **Lazy init**: server.py uses thread-locked singletons for `_project_db`, `_user_db`, `_embedder`
- **Import side-effect**: `server.py` line 112 does `import vibe_rag.tools` to register all `@mcp.tool()` decorators -- this must stay after `mcp` and helper definitions
- **Embedding provider**: OpenRouter only (default model: `perplexity/pplx-embed-v1-4b`); override model/dimensions with `RAG_OR_EMBED_MOD`/`RAG_OR_EMBED_DIM`
- **Unified search**: `search(query, scope="all|code|docs")` replaces old `search_code`/`search_docs` tools
- **Unified remember**: `remember(content, scope="project|user")` replaces old `remember`/`remember_structured`; optional structured fields (`summary`, `details`, `memory_kind`)
- **Match transparency**: all search results include `match_reason` field
- **In-place editing**: `update_memory(memory_id, ...)` for lightweight corrections without supersede/forget
- **Memory health in status**: `project_status(include_memory_health=True)` replaces separate `memory_cleanup_report`/`memory_quality_report` tools
- **Memory classification**: `save_session_memory` auto-classifies into kinds (decision/constraint/todo/fact/note) using term-matching heuristics in tools/

## Environment Variables

- `RAG_OR_API_KEY` — OpenRouter API key (required for embedding)
- `RAG_OR_EMBED_MOD` — override embedding model (default: `perplexity/pplx-embed-v1-4b`)
- `RAG_OR_EMBED_DIM` — override embedding dimensions (default: 2560)
- `VIBE_RAG_DB` — override project DB path
- `VIBE_RAG_USER_DB` — override user memory DB path

## Testing

- Tests use pytest-httpx to mock embedding API calls
- `conftest.py` provides shared fixtures
- No test requires a live embedding provider or OpenRouter API key
- Python 3.12 required (`tree-sitter-languages` lacks cp313 wheels)

## Gotchas

- `tools/` is the largest package -- MCP tool logic is split across submodules (index, memory, search, session, status)
- Version is tracked in three places: `pyproject.toml`, `src/vibe_rag/__init__.py`, `tests/test_cli.py` -- use `scripts/prepare_release.py` to update all
- `vibe-rag init` must work from an installed wheel, not just source -- always verify with `uv build` after template changes
- This repo now tracks maintainer bootstrap in `.vibe/`, `.codex/`, `.claude/`, and `.mcp.json`; only runtime DB state remains gitignored
- Release publishes via GitHub Actions `release.yml` which dispatches `publish.yml` -- do not push tags without creating a GitHub release
