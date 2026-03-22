# AGENTS.md

## Purpose

This project is configured for Vibe with `vibe-rag`.

Use the memory MCP tools first when the task is about:

- understanding the project
- finding relevant code by meaning
- searching docs
- recalling decisions from earlier sessions

## Tool preference

Use these tools in this order:

1. `memory_load_session_context`
   Use first for a new task or a resumed session.
   Pass the current task in plain English.
   Use `refresh_index: true` if the repo may be stale.

2. `memory_index_project`
   Re-index after pulling changes, after large edits, or any time the project index may be stale.
   Use `paths: ["."]` for the current project root.

3. `memory_search_code`
   Use for semantic questions like:
   - "where do we handle auth?"
   - "show me the builder install logic"
   - "find the part that writes config"

4. `memory_search_docs`
   Use for README, plans, specs, and markdown/text docs when the question is conceptual or process-oriented.

5. `memory_search_memory`
   Use before asking the user to repeat prior decisions, architecture notes, or cross-session context.

6. `read_file`
   After memory search narrows the target, read the specific files you need.

7. `grep`
   Use only when you already know the exact string, symbol, filename, or pattern to match.

## Default workflow

For repo understanding:

1. Run `memory_load_session_context` with the current task.
2. Re-index if needed with `refresh_index: true` or `memory_index_project`.
3. Run `memory_search_code` or `memory_search_docs` for deeper follow-up.
4. Read the most relevant files.
5. Make changes only after reading the target files.

For remembered context:

1. Run `memory_load_session_context` or `memory_search_memory` first.
2. If a new decision is made, store it with `memory_remember`.

For a brand-new repo session:

1. Run `memory_load_session_context`.
2. Run `memory_index_project`.
3. Use `memory_search_code` and `memory_search_docs`.
4. If results are stale or empty, re-check repo trust and MCP config.

## When to prefer memory tools over grep

Prefer memory tools when:

- the user describes behavior instead of exact identifiers
- the code could be spread across multiple files
- you need conceptual matches, not literal string matches
- you want prior project decisions or notes

Prefer `grep` when:

- you know the exact symbol or filename
- you are checking whether a literal string still exists
- you are doing fast exact-match verification after editing

## Memory rules

Store durable information with `memory_remember`:

- architecture decisions
- naming decisions
- chosen libraries or services
- migration notes
- repo-specific gotchas

Prefer `memory_remember_structured` when the host or workflow wants:

- a short summary plus supporting details
- a memory kind like `decision`, `constraint`, or `todo`
- source session metadata
- later supersession with `memory_supersede_memory`

Do not store:

- secrets
- temporary debugging noise
- information already obvious from a single current file

## Prompt patterns that work well

- "index this project"
- "load session context for continuing the auth refactor"
- "load session context for understanding this repo"
- "search the code for authentication handling"
- "search docs for release steps"
- "remember that we use pgvector for cross-repo memory"
- "what did we decide about config layout?"

## Setup reminders

- This repo should have a `.vibe/config.toml` that points the `memory` MCP server at `vibe-rag serve`.
- If durable memory is expected, that MCP config must include `DATABASE_URL`.
- `MISTRAL_API_KEY` is required for embeddings and semantic search.
- If background session bootstrap is expected, the Vibe config should contain:
  - `[background_mcp_hook]`
  - `enabled = true`
  - `tool_name = "memory_load_session_context"`
  - `task_arg = "task"`
- If the repo is not trusted, project-local config and skills may be ignored.

## Editing rules

- Read before editing.
- Keep changes narrow.
- Verify with tests or a direct command when possible.
- If memory results and source files disagree, trust the source files and refresh the index.
