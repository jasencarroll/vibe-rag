# vibe-rag User Guide

This is the day-to-day guide for using `vibe-rag` with Mistral Vibe.

## Mental Model

`vibe-rag` gives Vibe two different kinds of memory:

- Project index in local sqlite
  - Used for `index_project`, `search_code`, and `search_docs`
  - Lives in `.vibe/index.db`
  - Per-project
- Cross-session memory in pgvector
  - Used for `remember`, `search_memory`, and `forget`
  - Lives in PostgreSQL when `DATABASE_URL` is set
  - Shared across sessions
  - Scoped by `project_id`, so the current repo can find its own memories plus global ones

In practice:

- Use `index this project` when you want Vibe to understand the current repo
- Use `remember ...` when you want Vibe to carry something forward to later sessions
- Use `load session context for ...` when you want one retrieval step that bundles prior memories plus likely code and doc context

## Where To Put Keys

You need:

- `MISTRAL_API_KEY` for embeddings
- `DATABASE_URL` for cross-session pgvector memory

Put them in either:

1. shell env before launching `vibe`
2. Vibe MCP config `env` block

Project-local example:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
env = {
  MISTRAL_API_KEY = "your_api_key_here",
  DATABASE_URL = "postgresql://user:pass@localhost:5432/vibe_rag"
}
```

Use `.vibe/config.toml` for repo-specific setup and `~/.vibe/config.toml` for global setup.

## First Session In A Repo

When you open a repo for the first time:

```text
index this project
search the code for authentication handling
search docs for deployment instructions
```

That builds the project index and lets Vibe search by meaning instead of exact strings.

## How To Carry Context Across Sessions

The rule is simple:

- store durable decisions with `remember`
- retrieve them at the start of a later session with `load session context for ...` or `search_memory`

If you are building automation around memory, prefer:

- `remember_structured` for canonical stored facts
- `supersede_memory` when a newer decision replaces an older one

Good things to remember:

- architecture decisions
- chosen libraries
- naming decisions
- migration notes
- repo-specific conventions
- known gotchas

Bad things to remember:

- secrets
- temporary debugging notes
- information already obvious from current files

## Recommended Session Openers

If you are resuming work in the same repo:

```text
load session context for continuing the auth work
index this project
```

If you are starting a new task:

```text
load session context for investigating billing flow
```

If you want Vibe to pick up a decision for tomorrow:

```text
remember that we use pgvector for cross-session memory
remember that auth tokens are validated in the API gateway
remember that invoice numbers must remain human-readable
```

If the host is storing structured memory instead of plain notes, the equivalent record should contain:

- a short summary
- optional details
- a memory kind like `decision` or `constraint`
- optional source session/message metadata

## What “Across Sessions” Actually Means

If `DATABASE_URL` is configured:

- a memory stored today can be found tomorrow
- a memory stored in one Vibe session can be found in another
- the current repo will search its own project memories first and can also see global memories
- `load_session_context` can pull memories, likely code hits, and likely docs hits in one step

If `DATABASE_URL` is not configured:

- `remember` falls back to local sqlite memory
- that is not the same cross-repo persistent setup

## Practical Prompt Patterns

Good prompts:

```text
load session context for continuing the release automation work
index this project
search memory for our release process
search the code for where config is written
remember that we are standardizing on UUID primary keys
search memory for UUID decisions
forget memory id 123
```

Also good:

```text
before you start, search memory for prior architecture decisions and then index this project
```

## How To Think About IDs

When Vibe stores a pgvector memory, it returns a UUID-like ID, for example:

```text
Remembered in pgvector (id=ce630541-633a-411e-9b7d-3e79835cb59a): ...
```

You can use that exact ID with `forget`.

Example:

```text
forget memory ce630541-633a-411e-9b7d-3e79835cb59a
```

If a decision changes later, use `supersede_memory` instead of blindly piling on contradictory notes.

## Normal Workflow

A good default workflow is:

1. `load session context for ...`
2. `index this project` if needed
3. `search the code for ...`
4. make changes
5. `remember ...` if you made a durable decision

Example:

```text
load session context for working on database connections
remember that we now require UUID ids in postgres memory storage
```

## Troubleshooting

If Vibe seems to forget prior decisions:

- make sure `DATABASE_URL` is set in the MCP server env
- run `vibe-rag status`
- ask Vibe to `search memory for ...` explicitly

If code search seems stale:

- run `index this project` again

If Vibe uses grep instead of memory tools:

- prompt directly with `search memory for ...` or `search the code for ...`
- make sure the repo is trusted so project-local Vibe config and skills load

## Quick Reference

Use these prompts directly:

```text
load session context for continuing auth work
index this project
search memory for auth decisions
search the code for where config is written
search docs for release steps
remember that we deploy from main after green tests
forget memory <id>
```
