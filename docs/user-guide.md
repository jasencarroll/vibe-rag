# vibe-rag User Guide

Use `vibe-rag` as two systems:

| System | Purpose | Storage |
| --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` |
| User memory | durable cross-session memory | `~/.vibe/memory.db` |

If setup is not done yet, start with the [Setup Guide](setup-guide.md).

## Normal Flow

Start with:

```text
load session context for understanding this repo
index this project
```

Then use:

```text
search the code for where config is written
search docs for release steps
search memory for prior decisions
```

## What to Remember

Good durable memories:

- architecture decisions
- service boundaries
- naming rules
- chosen libraries
- migration notes
- deployment constraints
- repo-specific gotchas

Do not store:

- secrets
- temporary debugging noise
- facts obvious from one current file

## Useful Prompts

```text
load session context for continuing the auth refactor
load session context for release automation work
search the code for authentication handling
search docs for deployment instructions
remember that auth tokens are validated in the API gateway
search memory for auth tokens
```

## Structured Memory

Use:

- `remember_structured`
- `supersede_memory`

Good memory kinds:

- `decision`
- `constraint`
- `todo`
- `summary`
- `fact`

## Session Memory

If the Vibe fork hooks are enabled:

- completed turns are distilled into durable memory
- rolling session summaries are updated automatically
- new sessions can pull that context back through `memory_load_session_context`

## Retrieval Order

Memory retrieval is merged in this order:

1. current project memory
2. user memory

That keeps repo-local facts first while still allowing cross-repo carry-over.
