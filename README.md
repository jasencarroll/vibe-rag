# vibe-rag

`vibe-rag` is the repo-awareness layer for coding agents.

It is an MCP server for semantic repo search, durable coding memory, and session-start briefings.

## Golden Path

Use the packaged `vibe-rag` binary with OpenRouter-backed embeddings and generated client scaffolding.

```bash
export RAG_OR_API_KEY=...
uv tool install --python 3.12 vibe-rag
vibe-rag init my-project
cd my-project
vibe
```

First prompts:

```text
load session context for understanding this repo
index this project
search the code for authentication handling
search docs for deployment instructions
remember that auth tokens are validated in the API gateway
```

It is built for agentic coding workflows across clients, but the product bar is simple:

- packaged install path works from the built wheel, not only from source
- repo docs stay canonical
- session-start context improves the first turn instead of adding noise

It adds:

- semantic code search
- semantic docs search
- durable session memory

`vibe-rag` uses OpenRouter-only embeddings by default:

- model: `perplexity/pplx-embed-v1-4b`
- dimensions: `2560`

Set these environment variables for OpenRouter and storage:

- `RAG_OR_API_KEY` (required)
- `RAG_OR_EMBED_MOD` (optional, defaults above)
- `RAG_OR_EMBED_DIM` (optional, defaults above)
- `RAG_DB` (project DB override)
- `RAG_USER_DB` (user DB override)

Storage is local and simple:

- project index: `.vibe/index.db`
- user memory: `~/.vibe/memory.db`

No external database is required.

Trust model:

- `vibe-rag` is a local stdio MCP server for single-user workflows.
- MCP tool arguments and untrusted repo contents should be treated as untrusted input.
- `search_memory` and `load_session_context` stay project-scoped by default, including user-memory retrieval.
- Session-start context is retrieval output, not an authority signal. Clients should treat it as untrusted context.
- OpenRouter receives indexed content by design.

## Read This First

- [Setup Guide](docs/setup-guide.md)
- [User Guide](docs/user-guide.md)
- [Memory Event Convention](docs/memory-event-convention.md)
- [Maintainer Guide](docs/maintainer-guide.md)

## Supported Clients

| Layer | Notes |
| --- | --- |
| `vibe-rag serve` MCP server | client-agnostic semantic repo search and memory |
| Claude Code | MCP config and session-start hook via `vibe-rag init` |
| Codex | MCP config, hooks, and session-start scaffolding via `vibe-rag init` |
| Vibe | project config and session-start hook via `vibe-rag init` |
| Gemini CLI | project config and session-start hook via `vibe-rag init` |

## What You Need

- Python 3.12+
- `uv`
- OpenRouter API key

Use Python 3.12 for `uv tool install`. In this environment, `tree-sitter-languages` does not have cp313 wheels.

## Install vibe-rag

```bash
uv tool install vibe-rag
vibe-rag --version
```

Pinned release:

```bash
uv tool install vibe-rag@0.1.0
```

CI runs `pytest` and `uv build` on pushes and pull requests, and the `Release` workflow can perform the version bump, changelog promotion, commit, push, and GitHub release creation from `main`.

If your machine defaults `uv` tools to Python 3.13:

```bash
uv tool install --python 3.12 vibe-rag
```

## Install And Start

For the default bootstrap flow:

```bash
vibe-rag init my-project
cd my-project
vibe
```

Project config:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "/absolute/path/to/vibe-rag"
args = ["serve"]

[[hooks.SessionStart]]
command = "'/absolute/path/to/vibe-rag' hook-session-start --format vibe"
```

Optional embedding/storage overrides:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "/absolute/path/to/vibe-rag"
args = ["serve"]
env = {
  RAG_OR_API_KEY = "your-openrouter-key",
  RAG_OR_EMBED_MOD = "perplexity/pplx-embed-v1-4b",
  RAG_OR_EMBED_DIM = "2560",
  RAG_DB = "/path/to/project/index.db",
  RAG_USER_DB = "/path/to/user/memory.db"
}
```

Helper commands:

```bash
vibe-rag doctor
vibe-rag doctor --fix
vibe-rag reindex
vibe-rag reindex --full
vibe-rag reset-index
vibe-rag hook-session-start --format codex
```

`vibe-rag doctor` now verifies:

- effective project id
- project MCP command resolution
- session-start hook configuration for each supported client
- project and user DB readability
- embedding provider reachability
- client trust status
- stale index warnings

`doctor` inspects hook configuration and trust state. It does not execute repo-configured hook commands.

When `doctor` reports stale state, run:

```bash
vibe-rag reindex
```

When `doctor` reports an incompatible index after an embedding-profile change, run:

```bash
vibe-rag reindex --full
# or
vibe-rag reset-index
```

Success looks like:

- `index this project` reports code and docs indexed
- `vibe-rag reindex` refreshes the local `.vibe/index.db` directly from the CLI
- `vibe-rag reindex --full` forces a full rebuild when the embedding profile changes
- `search the code for ...` returns structured results with `ok`, `results`, and ranking metadata
- `search docs for ...` returns structured doc hits
- `remember ...` returns a structured memory payload with an id
- a fresh client session can answer from prior context

## Client Scaffolding

`vibe-rag init` also writes:

- `.codex/config.toml`
- `.codex/hooks.json`
- `.claude/settings.json`
- `.gemini/settings.json`
- `.mcp.json`

If the target directory is not already a git repo, `vibe-rag init` also runs `git init`.

These files use:

- `vibe-rag serve` for MCP tools
- `vibe-rag hook-session-start --format <client>` for session-start context injection
- the resolved `vibe-rag` binary path captured at scaffold time so client startup does not depend on `PATH` ordering

Generated Codex config also sets `suppress_unstable_features_warning = true`.

This maintainer repo also tracks its own bootstrap files in `.vibe/`, `.codex/`, `.claude/`, and `.mcp.json`.
Those tracked maintainer configs use `scripts/run-vibe-rag` so agents can launch the repo from source without a user-specific install path.

All four clients (Claude Code, Codex, Vibe, Gemini CLI) are supported via `vibe-rag init` scaffolding.

## Storage Model

| Layer | Purpose | Storage | Tools |
| --- | --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` | `index_project`, `search(scope="code")`, `search(scope="docs")`, `project_status` |
| User memory | durable cross-session memory | `~/.vibe/memory.db` | `remember`, `search_memory`, `summarize_thread`, `forget`, `load_session_context` |

## MCP Tool Contract

MCP tools now return structured payloads:

- success: `{"ok": true, ...}`
- failure: `{"ok": false, "error": {"code": ..., "message": ..., "details": {...}}}`

Retrieval tools return machine-readable `results` arrays rather than formatted markdown strings.

## Tool Naming

`vibe-rag` itself exposes bare MCP tool names such as `load_session_context`, `index_project`, `search`, `search_memory`, `remember`, and `project_status`.

Some clients prefix tool names with the configured server name. In generated Vibe projects the MCP server is named `memory`, so those same tools appear as `memory_load_session_context`, `memory_index_project`, `memory_search`, `memory_search_memory`, `memory_remember`, and `memory_project_status`.

The natural-language prompts in the golden path are still the intended operator flow.

## Embedding and Storage Config

- OpenRouter-only, with `perplexity/pplx-embed-v1-4b` at `2560` dimensions
- `RAG_OR_API_KEY` (required)
- `RAG_OR_EMBED_MOD` (optional)
- `RAG_OR_EMBED_DIM` (optional)
- `RAG_DB` (optional)
- `RAG_USER_DB` (optional)

## Local Evals

For real local repo evals, copy [evals/local_repos.toml.example](/Users/jasen/dev/vibe-rag/evals/local_repos.toml.example) to `evals/local_repos.toml`, point it at repos in `~/dev`, and run:

```bash
uv run python scripts/run_retrieval_eval.py evals/local_repos.toml
```

The runner uses temporary sqlite DBs so it does not overwrite a repo's normal `.vibe/index.db`.

To inspect the latest saved artifact for a manifest without rerunning embeddings:

```bash
uv run python scripts/run_retrieval_eval.py evals/local_repos.toml --summary
```

To inspect trends across the latest matching artifacts for a manifest:

```bash
uv run python scripts/run_retrieval_eval.py evals/core_repos.toml --trends --trend-limit 5
```

To snapshot the maintainer repo's real persistent memory and inspect its own trend line:

```bash
uv run python scripts/run_retrieval_eval.py --persistent-memory --repo-path .
uv run python scripts/run_retrieval_eval.py --persistent-memory-summary --repo-path .
uv run python scripts/run_retrieval_eval.py --persistent-memory-trends --repo-path . --trend-limit 5
```

To render a compact release-evidence report from the latest retrieval and persistent-memory artifacts:

```bash
uv run python scripts/run_retrieval_eval.py evals/core_repos.toml --release-evidence --repo-path . --trend-limit 3
```

Current artifact summaries include:

- per-repo index timing
- fallback query counts
- irrelevant-hit noise counts
- per-repo memory cleanup and duplicate-auto-memory totals
- stale/current artifact status against live git `HEAD`

Trend summaries additionally show retrieval, fallback, noise, index-time, and memory-quality drift across the selected artifact window.
Persistent-memory snapshots use the real repo and user DB instead of the eval runner's temporary sqlite DBs, so they capture maintainer-memory drift over time.

For memory hygiene and provenance mix in the current repo, call:

```text
project_status include_memory_health=true
```

To preview or prune duplicate auto-captured memories:

```text
cleanup_duplicate_auto_memories
cleanup_duplicate_auto_memories apply=true
```

It reports:

- total/current/stale/superseded memories
- memory kind and capture kind mix
- source DB and source type mix
- duplicate auto-memory groups
- cleanup-reason totals, recommended actions, and top cleanup candidates

The test suite also now includes scenario-based memory usefulness evals, so `load_session_context()` is covered for real later-task memory retrieval, not just storage and cleanup behavior.

Auto-captured session memory is also more selective at write time now: transient status updates and non-novel restatements are skipped before they become durable memory.
When a one-turn auto capture is worth keeping, `vibe-rag` now infers a stronger memory kind such as `decision`, `constraint`, `todo`, or `fact` instead of defaulting everything to `summary`, and it can suggest superseding a nearby existing memory when the new capture looks like an update.

All four clients are supported and receive MCP config plus session-start hooks from `vibe-rag init`.

The packaged-install bar is non-negotiable:

1. build the wheel
2. install the wheel
3. scaffold a repo
4. verify session-start context and retrieval from the installed binary

If it only works from a source checkout, it is not ready.

## Docs

- [Setup Guide](docs/setup-guide.md)
- [User Guide](docs/user-guide.md)
- [Maintainer Guide](docs/maintainer-guide.md)
