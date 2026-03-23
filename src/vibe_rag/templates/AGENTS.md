# AGENTS.md

Use the memory MCP tools first when the task is about:

- understanding the repo
- finding code by meaning
- searching docs
- recalling prior decisions

## Tool order

1. `memory_load_session_context`
2. `memory_index_project`
3. `memory_search_code`
4. `memory_search_docs`
5. `memory_search_memory`
6. `read_file`
7. `grep`

## Rules

- Prefer memory tools over `grep` when the user does not know the exact identifier.
- Re-index after pulling changes or after large edits.
- If memory and source disagree, trust the source and re-index.
- Never store secrets.

## Storage

- project index lives in `.vibe/index.db`
- durable user memory lives in `~/.vibe/memory.db`
- Ollama is the default embedding provider

## Client Scaffolding

- Vibe is the first-class client.
- Generated repos also include experimental Codex and Claude Code session-start scaffolding.
- `vibe-rag init` initializes a git repo when one does not already exist so repo-scoped client behavior works.
- `vibe-rag init` pins the resolved `vibe-rag` binary path in generated client configs so startup does not depend on `PATH` ordering.
- If hook-loaded context disagrees with the source tree, trust the source and re-index.

## Good prompts

- "load session context for understanding this repo"
- "search the code for authentication handling"
- "search docs for release steps"
- "remember that auth tokens are validated in the API gateway"
