# vibe-rag

`vibe-rag` is an MCP server for semantic repo search and durable coding memory.

It is built for agentic coding workflows across clients.

The most complete integration today is Vibe.

Generated repos also include scaffolding for:

- Codex
- Claude Code
- Gemini CLI

It adds:

- semantic code search
- semantic docs search
- durable session memory

When Ollama is unavailable, `vibe-rag` now auto-falls back to a configured hosted embedding provider instead of failing just because no explicit provider env was set.

Storage is local and simple:

- project index: `.vibe/index.db`
- user memory: `~/.vibe/memory.db`

No external database is required.

## Support Levels

| Layer | Status | Notes |
| --- | --- | --- |
| `vibe-rag serve` MCP server | core identity | client-agnostic semantic repo search and memory |
| Vibe integration | most complete | first-class path with session bootstrap and session memory hooks |
| Codex integration | experimental | project scaffolding plus session-start hook |
| Claude Code integration | experimental | project scaffolding plus session-start hook |
| Gemini CLI integration | experimental | project scaffolding plus session-start hook |

## What You Need

- Python 3.12+
- `uv`
- Ollama
- Vibe

Use Python 3.12 for `uv tool install`. In this environment, `tree-sitter-languages` does not have cp313 wheels.

## Vibe Integration

If you want the most complete integration path today, use the fork that contains the background MCP session bootstrap hook:

- `https://github.com/jasencarroll/mistral-vibe`

Install it:

```bash
uv tool uninstall mistral-vibe || true
uv tool install git+https://github.com/jasencarroll/mistral-vibe.git
vibe --version
```

## Install vibe-rag

```bash
uv tool install vibe-rag
vibe-rag --version
```

Pinned release:

```bash
uv tool install vibe-rag@0.0.25
```

CI runs `pytest` and `uv build` on pushes and pull requests, and the `Release` workflow can perform the version bump, changelog promotion, commit, push, and GitHub release creation from `main`.

If your machine defaults `uv` tools to Python 3.13:

```bash
uv tool install --python 3.12 vibe-rag
```

## Quick Start

Choose the client path you want to start with.

For the most complete Vibe path:

```bash
vibe-rag setup-ollama
vibe-rag init my-project
cd my-project
vibe
```

Project config:

```toml
active_model = "devstral-2"
skill_paths = [".vibe/skills"]

[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "/absolute/path/to/vibe-rag"
args = ["serve"]

[[hooks.SessionStart]]
command = "'/absolute/path/to/vibe-rag' hook-session-start --format vibe"
```

Optional Ollama overrides:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "/absolute/path/to/vibe-rag"
args = ["serve"]
env = {
  VIBE_RAG_EMBEDDING_PROVIDER = "ollama",
  VIBE_RAG_EMBEDDING_MODEL = "qwen3-embedding:0.6b",
  VIBE_RAG_EMBEDDING_DIMENSIONS = "1024"
}
```

Optional:

- `VIBE_RAG_OLLAMA_HOST`
- `MISTRAL_API_KEY` if you switch to the Mistral provider
- `OPENAI_API_KEY` if you switch to the OpenAI provider
- `VOYAGE_API_KEY` if you switch to the Voyage provider

If `VIBE_RAG_OLLAMA_HOST` is not set, `vibe-rag` checks:

- `OLLAMA_HOST`
- `http://localhost:11434`
- `http://127.0.0.1:11434`

Helper commands:

```bash
vibe-rag doctor
vibe-rag doctor --fix
vibe-rag setup-ollama
vibe-rag hook-session-start --format codex
```

`vibe-rag doctor` now verifies:

- effective project id
- project MCP command resolution
- Vibe SessionStart hook execution
- Codex SessionStart hook execution
- project and user DB readability
- embedding provider reachability
- Vibe and Codex trust status
- stale index warnings

When `doctor` reports stale state, run:

```bash
vibe-rag reindex
```

First prompts:

```text
load session context for understanding this repo
index this project
search the code for authentication handling
search docs for deployment instructions
remember that auth tokens are validated in the API gateway
```

Success looks like:

- `index this project` reports code and docs indexed
- `vibe-rag reindex` refreshes the local `.vibe/index.db` directly from the CLI
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

Current support level is shown in the support table above.

## Storage Model

| Layer | Purpose | Storage | Tools |
| --- | --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` | `index_project`, `search_code`, `search_docs` |
| User memory | durable cross-session memory | `~/.vibe/memory.db` | `remember`, `search_memory`, `forget`, `load_session_context` |

## MCP Tool Contract

MCP tools now return structured payloads:

- success: `{"ok": true, ...}`
- failure: `{"ok": false, "error": {"code": ..., "message": ..., "details": {...}}}`

Retrieval tools return machine-readable `results` arrays rather than formatted markdown strings.

## Embedding Providers

Current providers:

- `ollama` (default)
- `mistral`
- `openai`
- `voyage`

Provider env vars:

- `VIBE_RAG_EMBEDDING_PROVIDER`
- `VIBE_RAG_EMBEDDING_MODEL`
- `VIBE_RAG_EMBEDDING_DIMENSIONS`
- `VIBE_RAG_OLLAMA_HOST` for Ollama
- `OPENAI_API_KEY` for OpenAI
- `VOYAGE_API_KEY` for Voyage
- `VIBE_RAG_CODE_EMBEDDING_MODEL` for Voyage code embeddings

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

For memory hygiene and provenance mix in the current repo, call the MCP tool:

```text
memory_quality_report
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

## Docs

- [Setup Guide](docs/setup-guide.md)
- [User Guide](docs/user-guide.md)
- [Maintainer Guide](docs/maintainer-guide.md)
