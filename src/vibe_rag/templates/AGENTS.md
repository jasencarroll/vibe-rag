# AGENTS.md

Use the memory MCP tools first when the task is about:

- understanding the repo
- finding code by meaning
- searching docs
- recalling prior decisions

For a fresh repo session, start with:

- `load session context for understanding this repo`
- `index this project`

## Tool order

1. `memory_load_session_context`
2. `memory_index_project`
3. `memory_search`
4. `memory_search_memory`
5. `memory_project_status`
6. `memory_remember`
7. `read_file`
8. `grep`

The MCP server itself exposes bare tool names like `load_session_context`, `index_project`, `search`, `remember`, and `project_status`.
In generated Vibe configs the server is named `memory`, so client-visible tool names are prefixed as `memory_*`.
Use `memory_search` with `scope="code"` or `scope="docs"` instead of looking for separate code/doc search tools.
OpenRouter is the only supported embedding backend in this generated workflow.

## Rules

- Prefer memory tools over `grep` when the user does not know the exact identifier.
- Put OpenRouter credentials in `~/.vibe-rag/config.toml`; use env overrides only when you need a repo-specific override.
- If the MCP tools are missing, hooks do not fire, or indexing/search fails unexpectedly, run `vibe-rag doctor`.
- Re-run `vibe-rag init` from the repo root when you need to refresh the generated scaffold. `vibe-rag init --here` remains an alias.
- Re-index after pulling changes or after large edits.
- If the index is stale or incompatible, run `vibe-rag reindex` or `vibe-rag reindex --full`.
- If memory and source disagree, trust the source and re-index.
- If the durable user memory DB is unreadable, run `vibe-rag reset-user-memory`.
- Never store secrets.

## Storage

- project index lives in `.vibe-rag/index.db`
- durable user memory lives in `~/.vibe-rag/memory.db`
- OpenRouter is the embedding backend

## Client Scaffolding

- All four agent CLIs are supported: Claude Code, Codex, Gemini CLI, and Vibe.
- `vibe-rag init` initializes a git repo when one does not already exist so repo-scoped client behavior works.
- `vibe-rag init` pins the resolved `vibe-rag` binary path in generated client configs so startup does not depend on `PATH` ordering.
- Generated configs are expected to work from the installed binary, not only from a source checkout.
- If hook-loaded context disagrees with the source tree, trust the source and re-index.

## Good prompts

- "load session context for understanding this repo"
- "search the code for authentication handling"
- "search docs for release steps"
- "remember that auth tokens are validated in the API gateway"
