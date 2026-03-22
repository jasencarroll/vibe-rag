# AGENTS.md

## Purpose

This project is configured for Mistral Vibe with `vibe-rag`.

Use the memory MCP tools first when the task is about understanding the project, finding relevant code by meaning, searching docs, or recalling decisions from earlier sessions.

## Tool preference

Use these tools in this order:

1. `memory_index_project`
   Re-index after pulling changes, after large edits, or any time the project index may be stale.
   Use `paths: ["."]` for the current project root.

2. `memory_search_code`
   Use for semantic questions like:
   - "where do we handle auth?"
   - "show me the builder install logic"
   - "find the part that writes config"

3. `memory_search_docs`
   Use for README, plans, specs, and markdown/text docs when the question is conceptual or process-oriented.

4. `memory_search_memory`
   Use before asking the user to repeat prior decisions, architecture notes, or cross-session context.

5. `read_file`
   After memory search narrows the target, read the specific files you need.

6. `grep`
   Use only when you already know the exact string, symbol, filename, or pattern to match.

## Default workflow

For repo understanding:

1. Run `memory_index_project` if the index is missing or stale.
2. Run `memory_search_code` or `memory_search_docs`.
3. Read the most relevant files.
4. Make changes only after reading the target files.

For remembered context:

1. Run `memory_search_memory` first.
2. If a new decision is made, store it with `memory_remember`.

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

Do not store:

- secrets
- temporary debugging noise
- information already obvious from a single current file

## Prompt patterns that work well

- "index this project"
- "search the code for authentication handling"
- "search docs for release steps"
- "remember that we use pgvector for cross-repo memory"
- "what did we decide about config layout?"

## Editing rules

- Read before editing.
- Keep changes narrow.
- Verify with tests or a direct command when possible.
- If memory results and source files disagree, trust the source files and refresh the index.
