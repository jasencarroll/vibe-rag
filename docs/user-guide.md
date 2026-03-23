# vibe-rag User Guide

Use `vibe-rag` as two systems inside your coding client:

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

If your client integration enables session hooks:

- completed turns are distilled into durable memory
- rolling session summaries are updated automatically
- new sessions can pull that context back through `memory_load_session_context`

Bootstrap results now include:

- provenance for memory, code, and docs hits
- index staleness warnings when git head or indexed files drift
- stronger bias toward structured memory kinds like `decision`, `constraint`, `summary`, and `todo`
- down-ranking for low-signal auto session summaries when stronger memory exists

For memory hygiene inspection:

```text
memory_quality_report
```

Useful things to watch there:

- stale or cross-project user memories
- superseded memory accumulation
- too many freeform notes
- duplicate auto-captured session probes
- low-signal auto session summaries that should be cleaned up or superseded

Very low-signal auto session summaries such as one-turn greetings are now skipped during capture instead of being stored and cleaned up later.

If you want to preview or prune repeated auto-captured session probes, use:

```text
cleanup_duplicate_auto_memories
cleanup_duplicate_auto_memories apply=true
```

Memory usefulness is also now covered by scenario tests around `load_session_context()`, including structured decision retrieval, todo/constraint recall, supersession, and non-interference from low-signal auto memories.

For longer-running eval work, the retrieval eval runner can now summarize cross-artifact trends so you can track fallback usage, noise, index timing, and memory cleanup pressure over time instead of inspecting one JSON file at a time.

Auto-captured session memories also go through a write-time durability/novelty gate now, so transient status chatter and repetitive restatements are more likely to be skipped instead of stored and cleaned up later.

If you want the same style of evidence for the maintainer repo's real persistent memory, use the `--persistent-memory`, `--persistent-memory-summary`, and `--persistent-memory-trends` modes in the eval runner. Those snapshots track the actual repo/user memory state over time rather than the eval runner's temporary DBs.
For release prep, `--release-evidence` combines the latest retrieval snapshot, retrieval trends, persistent-memory snapshot, and persistent-memory trends into one compact report.

One-turn auto session captures now infer stronger kinds such as `decision`, `constraint`, `todo`, or `fact` when the content supports it, and the save result can include a merge/supersede suggestion when the new capture looks like an update to an existing memory.

## Other Clients

Generated repos also include experimental session-start scaffolding for:

- Codex
- Claude Code
- Gemini CLI

Those clients currently get:

- MCP server registration for `vibe-rag serve`
- session-start context injection through `vibe-rag hook-session-start`
- automatic `git init` when the scaffold target is not already a repo
- Codex config with `suppress_unstable_features_warning = true`

Vibe is currently the most complete integration, but the core `vibe-rag serve` MCP server is the product identity.

## Retrieval Order

Memory retrieval is merged in this order:

1. current project memory
2. user memory

That keeps repo-local facts first while still allowing cross-repo carry-over.
