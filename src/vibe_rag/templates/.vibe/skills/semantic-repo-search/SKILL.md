---
name: semantic-repo-search
description: Search code, docs, and prior project context with vibe-rag memory tools before grep or shell.
license: MIT
user-invocable: true
allowed-tools:
  - memory_index_project
  - memory_search_code
  - memory_search_docs
  - memory_search_memory
  - memory_remember
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

1. If the project may be new or stale, run `memory_index_project` with `paths: ["."]`.
2. For code understanding, run `memory_search_code`.
3. For README, plans, specs, or notes, run `memory_search_docs`.
4. For prior decisions or cross-session context, run `memory_search_memory`.
5. After memory narrows the target, use `read_file` on the specific files you need.
6. Use `grep` only for exact strings, symbols, filenames, or post-edit verification.

## Rules

- Prefer memory tools over `grep` when the user does not know the exact identifier.
- Refresh the index after major edits, after pulling changes, or when search results look stale.
- If memory and source disagree, trust the source and re-index.
- Never store secrets with `memory_remember`.
