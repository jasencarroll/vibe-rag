---
name: semantic-repo-search
description: Search code, docs, and prior project context with vibe-rag memory tools before grep or shell.
license: MIT
user-invocable: true
allowed-tools:
  - memory_load_session_context
  - memory_index_project
  - memory_search
  - memory_search_memory
  - memory_remember
  - memory_project_status
  - memory_supersede_memory
  - read_file
  - grep
---

# Semantic Repo Search

Use this skill for repo-understanding tasks where the user describes behavior, intent, architecture, or prior decisions.

## Use This For

- "where is auth handled?"
- "search the code for config writing"
- "tell me about this repo"
- "find the builder install logic"
- "what did we decide about persistence?"

## Workflow

1. For a new task or resumed session, run `memory_load_session_context` with the task in plain English.
2. If the project may be new or stale, either pass `refresh_index: true` to `memory_load_session_context` or run `memory_index_project` with `paths: ["."]`.
3. For deeper code understanding, run `memory_search` with `scope: "code"`.
4. For README, plans, specs, or notes, run `memory_search` with `scope: "docs"`.
5. For direct prior decisions or cross-session context, run `memory_search_memory`.
6. If results look stale or noisy, run `memory_project_status` to inspect staleness and memory health.
7. After memory narrows the target, use `read_file` on the specific files you need.
8. Use `grep` only for exact strings, symbols, filenames, or post-edit verification.

## Rules

- Prefer memory tools over `grep` when the user does not know the exact identifier.
- Prefer `memory_load_session_context` as the first retrieval step when you need both prior memory and likely code/doc context.
- The underlying MCP tool names are `load_session_context`, `index_project`, `search`, `search_memory`, `remember`, and `project_status`; Vibe prefixes them because the server is named `memory`.
- Refresh the index after major edits, after pulling changes, or when search results look stale.
- If memory and source disagree, trust the source and re-index.
- Never store secrets with `memory_remember`.
- If memory tools seem unavailable or low-quality, check whether the repo is trusted and whether OpenRouter is configured and reachable.
