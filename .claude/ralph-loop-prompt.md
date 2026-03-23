You are continuing the vibe-rag MCP overhaul on branch streamline/mcp-overhaul. The goal is to make the MCP tool surface feel effortless to use as an AI consumer. Three consolidation waves already landed: remember merge, search merge, status fold.

Each iteration you MUST launch exactly 6 agents in parallel using worktree isolation, each on its own branch. After agents complete, merge their branches into streamline/mcp-overhaul, run tests, and loop.

Here is the backlog. Work top to bottom, 6 at a time:

1. Fix language_stats None bug - tree-sitter code chunker not populating language into DB rows. Trace the flow from file extension through EXT_TO_LANG to DB insert.

2. Split tools.py into a tools/ package with focused modules. Create __init__.py, _helpers.py, search.py, memory.py, index.py, status.py, session.py. All existing imports must remain backward-compatible.

3. Rewrite all 13 MCP tool docstrings to be clear, actionable, and help AI consumers pick the right tool on first call. Each docstring should explain when to use the tool, what it returns, and how it differs from similar tools.

4. Add embedding LRU cache in the embedder layer so repeated queries hit cache. load_session_context embeds the task 3 times in one call. Only cache single-text embeds, not batch indexing.

5. Improve CLI help text and vibe-rag doctor output - add memory health summary, language coverage warnings, embedding provider health check.

6. Update CLAUDE.md to reflect the new tool surface: 13 tools, unified search/remember with scope param, update_memory, match_reason on all search results.

7. Add integration test that exercises the full remember -> search_memory -> update_memory -> forget lifecycle in one test.

8. Improve search result ranking - add recency boost for memories, file-path relevance boost for code results.

9. Add a compact mode to load_session_context with a compact=True param that returns just the briefing text without full result objects, for lighter context windows.

10. Harden update_memory - add tests for partial field updates with empty strings vs None, metadata merge edge cases, re-embedding behavior.

11. Add tag-based filtering to search_memory with an optional tags param that pre-filters before vector search.

12. Clean up conftest.py test fixtures - deduplicate setup patterns, add a fixture for pre-populated index with code and docs.

Run uv run pytest -x -q --ignore=tests/test_embedder.py after each merge to verify. Never push to remote.
