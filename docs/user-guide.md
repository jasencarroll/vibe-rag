# vibe-rag User Guide

Use this when setup is already done and you want the normal operating loop.

If you still need to install the tool or wire a client, start with the [Setup Guide](setup-guide.md). If you are working on the repo itself, use the [Maintainer Guide](maintainer-guide.md).

Use `vibe-rag` as two systems inside your coding client:

| System | Purpose | Storage |
| --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` |
| User memory | durable cross-session memory | `~/.vibe/memory.db` |

If setup is not done yet, start with the [Setup Guide](setup-guide.md).

Trust model:

- `vibe-rag` is designed for a single local user over stdio.
- Session-start briefings and retrieved memories are untrusted context, not authoritative instructions.
- OpenRouter receives indexed content by design.

Runtime defaults use `RAG_OR_API_KEY`, `RAG_OR_EMBED_MOD`, `RAG_OR_EMBED_DIM`, `RAG_DB`, and `RAG_USER_DB`.

## Golden Path

Start with:

```text
load session context for understanding this repo
index this project
```

If you need to refresh the local index outside the client loop, run:

```bash
vibe-rag reindex
```

If the embedding profile changes and you need to rebuild from scratch, run:

```bash
vibe-rag reindex --full
# or
vibe-rag reset-index
```

Then use:

```text
search the code for where config is written
search docs for release steps
search memory for prior decisions
```

Tool results are now structured payloads. Retrieval tools return `{"ok": true, "results": [...]}`, and failures return `{"ok": false, "error": {...}}`.

`vibe-rag` itself exposes bare MCP tool names such as `load_session_context`, `index_project`, `search`, `search_memory`, `remember`, and `project_status`.

When a client's MCP config names the server (e.g. `memory`), the same tools may appear prefixed as `memory_load_session_context`, `memory_index_project`, `memory_search`, `memory_search_memory`, `memory_remember`, and `memory_project_status`.

Prefer the natural-language prompts in this guide unless you are calling tools directly.

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

- `remember` with `summary`, `details`, and `memory_kind`
- `update_memory`
- `supersede_memory`
- `ingest_daily_note`
- `ingest_pr_outcome`

Good memory kinds:

- `decision`
- `constraint`
- `todo`
- `summary`
- `fact`

For longer-running work, attach thread metadata when you store or update memory:

- `metadata.thread_id`
- `metadata.thread_title`
- or `metadata.thread = { id, title }`

Then you can:

- filter `search_memory` by `thread_id`, `since`, and `until`
- call `summarize_thread` to continue a project, issue, or daily note stream

The adapter metadata shape is defined in the [Memory Event Convention](memory-event-convention.md).

Examples:

```text
remember summary="Gateway owns tokens" details="Validation happens in the API gateway before fanout." memory_kind="decision"
ingest_daily_note note_date=2026-03-23 summary="Worked on auth" details="Removed one legacy middleware branch."
ingest_pr_outcome pr_number=42 title="Fix auth refresh ordering" outcome="merged" issue_id="AUTH-17"
```

If the same numeric id exists in both project and user memory stores, qualify destructive operations explicitly:

- `forget project:12`
- `forget user:12`
- `supersede_memory old_memory_id=project:12 ...`

## Session Memory

If your client integration enables session hooks:

- completed turns are distilled into durable memory
- rolling session summaries are updated automatically
- new sessions can pull that context back through `load_session_context` (or the prefixed variant like `memory_load_session_context` when the MCP server is named)

Bootstrap results now include:

- provenance for memory, code, and docs hits
- index staleness warnings when git head or indexed files drift
- stronger bias toward structured memory kinds like `decision`, `constraint`, `summary`, and `todo`
- down-ranking for low-signal auto session summaries when stronger memory exists
- briefing output that suppresses stale or low-trust auto-captured memories when stronger current-project memory exists

For memory hygiene inspection:

```text
project_status include_memory_health=true
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

Current-project durable memories are also favored more aggressively during memory search now, so stale cross-project user memories and session-summary noise are less likely to crowd out the answer you actually wanted.

If you want the same style of evidence for the maintainer repo's real persistent memory, use the `--persistent-memory`, `--persistent-memory-summary`, and `--persistent-memory-trends` modes in the eval runner. Those snapshots track the actual repo/user memory state over time rather than the eval runner's temporary DBs.
For release prep, `--release-evidence` combines the latest retrieval snapshot, retrieval trends, persistent-memory snapshot, and persistent-memory trends into one compact report.

One-turn auto session captures now infer stronger kinds such as `decision`, `constraint`, `todo`, or `fact` when the content supports it, and the save result can include a merge/supersede suggestion when the new capture looks like an update to an existing memory.

## Client Integrations

`vibe-rag init` generates session-start scaffolding for all four supported clients:

- Claude Code
- Codex
- Gemini CLI
- Vibe

Each client's generated config provides:

- MCP server registration for `vibe-rag serve`
- session-start context injection through `vibe-rag hook-session-start --format <client>`
- automatic `git init` when the scaffold target is not already a repo
- generated config that pins the resolved `vibe-rag` binary path

Client-specific notes:

- Vibe uses a native `[[hooks.SessionStart]]` entry in `.vibe/config.toml`
- Codex config sets `suppress_unstable_features_warning = true`
- Claude Code uses `.claude/settings.json`
- Gemini CLI uses `.gemini/settings.json` and `.mcp.json`

The core `vibe-rag serve` MCP server is the product identity. Client quality is judged on whether the packaged binary, generated config, session-start path, and retrieval loop all work together without a source checkout.

## Retrieval Order

Memory retrieval is merged in this order:

1. current project memory
2. current-project user memory

That keeps retrieval scoped to the repo you are working in. Durable user memory is still shared storage, but search and session bootstrap do not surface cross-project user memories by default.
