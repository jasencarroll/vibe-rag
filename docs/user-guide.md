# vibe-rag User Guide

This is the day-to-day workflow guide after setup is done.

If you have not installed Vibe, `vibe-rag`, PostgreSQL, and the MCP config yet, start here first:

- [Setup Guide](setup-guide.md)

## Mental Model

Use `vibe-rag` as two systems:

| System | What it is for | Where it lives |
| --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` |
| Durable memory | cross-session decisions, conventions, constraints, TODO context | PostgreSQL via `pgvector` |

In practice:

- `index this project` teaches Vibe the current repo
- `remember ...` stores something worth keeping
- `load session context for ...` gives Vibe one retrieval step that bundles likely memory, code, and docs context

Important distinction:

- `index this project` helps Vibe understand the current repo now
- `remember ...` helps Vibe remember something later
- these are not interchangeable

## First Session in a Repo

Use this order:

```text
load session context for understanding this repo
index this project
search the code for authentication handling
search docs for deployment instructions
```

If the repo is large or changed recently, `index this project` is not optional.

If you skip indexing, Vibe may still have durable memory, but code/docs retrieval quality will be worse.

## What to Remember

Good durable memories:

- architecture decisions
- boundaries between services
- repo-specific naming rules
- chosen libraries
- migration rules
- deployment constraints
- known gotchas

Do not remember:

- secrets
- transient debugging notes
- things already obvious from a single file

Rule of thumb:

If you would want the same fact available next week without rereading the whole repo, store it.

## Prompt Patterns That Work

Good openers:

```text
load session context for continuing the auth refactor
load session context for investigating billing failures
load session context for release automation work
```

Good follow-ups:

```text
search the code for where config is written
search docs for release steps
search memory for invoice rules
remember that auth tokens are validated in the API gateway
```

Good “resume later” loop:

```text
remember that invoice ids must stay human-readable
remember that webhook retries are capped at 5
```

Next session:

```text
load session context for working on invoice webhooks
```

## Structured Memory

If your host or automation wants cleaner records, prefer:

- `remember_structured`
- `supersede_memory`

Use structured memory for things like:

- `decision`
- `constraint`
- `todo`
- `summary`
- `fact`

Use supersession when an old decision is no longer valid. That is better than keeping contradictory notes alive forever.

## How to Think About Search

Use `search_code` when:

- you know behavior but not the exact symbol
- the relevant code may be spread across files
- you want semantic matches, not exact string matches

Use `search_docs` when:

- the answer is likely in README or markdown
- you are looking for process, setup, release, or architecture notes

Use `search_memory` when:

- you want prior decisions
- you are resuming a task from another day
- you want cross-repo context from durable memory

Use `grep` only when you already know the literal string or identifier.

If Vibe keeps defaulting to `grep`, nudge it with a more semantic prompt:

```text
search the code for where auth tokens are validated
```

## Recommended Session Openers

### Resume a task

```text
load session context for continuing the auth work
index this project
```

### Start a new task

```text
load session context for investigating the billing flow
index this project
```

### Confirm a prior decision

```text
search memory for UUID decisions
search memory for auth architecture
```

### Save a new decision

```text
remember that we deploy from main after green tests
remember that invoice numbers must remain human-readable
```

## Working with IDs

When `remember` stores a pgvector memory, it returns something like:

```text
Remembered in pgvector (id=ce630541-633a-411e-9b7d-3e79835cb59a): ...
```

Use that exact id with:

```text
forget memory ce630541-633a-411e-9b7d-3e79835cb59a
```

If durable memory is backed by pgvector, ids are UUID-like. If you only have local fallback memory, the behavior may be different.

## Default Workflow

A solid default loop is:

1. `load session context for ...`
2. `index this project`
3. `search the code for ...`
4. make changes
5. `remember ...` if you created a durable rule or decision

Example:

```text
load session context for working on database connections
index this project
search the code for where database connections are created
remember that postgres memory ids are UUIDs
```

Minimal successful day:

1. `load session context for ...`
2. `index this project`
3. ask 2-3 semantic questions
4. store one durable decision

## Troubleshooting

### Vibe does not seem to remember anything

- check `DATABASE_URL` in `.vibe/config.toml`
- run `vibe-rag status`
- run `search memory for ...` explicitly

### Search feels stale

```text
index this project
```

### Vibe keeps using grep instead of memory tools

- ask directly with `search memory for ...`
- ask directly with `search the code for ...`
- make sure the repo is trusted so project-local skills load

### Background bootstrap does not seem to happen

Check:

- the repo is trusted
- `[background_mcp_hook]` exists in `.vibe/config.toml`
- the MCP server `env` includes both `MISTRAL_API_KEY` and `DATABASE_URL`

On macOS, also verify the trusted path matches the real repo path from:

```bash
pwd -P
```

## Quick Reference

```text
load session context for continuing auth work
index this project
search memory for auth decisions
search the code for where config is written
search docs for release steps
remember that we deploy from main after green tests
forget memory <id>
```
