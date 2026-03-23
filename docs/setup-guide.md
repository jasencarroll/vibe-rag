# vibe-rag Setup Guide

Use this when you are wiring a fresh machine, validating a packaged install, or checking whether a client bootstrap path is actually healthy.

If you already have the tool installed and want the day-to-day operating flow, jump to the [User Guide](user-guide.md). If you are changing packaging, scaffold, or release behavior in this repo, use the [Maintainer Guide](maintainer-guide.md).

If you just want the working path first, do this:

```bash
uv tool install --python 3.12 vibe-rag
vibe-rag init demo
cd demo
# start your generated client (vibe, codex, claude, or gemini)
```

Then start the session with:

```text
load session context for understanding this repo
index this project
```

Target state:

- packaged `vibe-rag` from an installed wheel
- project-local config for your chosen client
- local durable memory in `~/.vibe/memory.db`
- session bootstrap where supported
- OpenRouter embeddings with `perplexity/pplx-embed-v1-4b` at `2560` dims
- client scaffolding for Vibe, Codex, Claude Code, or Gemini CLI

`vibe-rag` itself is the MCP server and memory/search layer.
Client integrations sit on top of that core.

## 1. Install the Tools

Install `vibe-rag`:

```bash
uv tool install vibe-rag
vibe-rag --version
```

If `uv` defaults to Python 3.13:

```bash
uv tool install --python 3.12 vibe-rag
```

Set the OpenRouter key once in your environment:

```bash
export RAG_OR_API_KEY="..."
```

## 2. Scaffold a Repo

```bash
vibe-rag init demo
cd demo
```

This writes:

- `AGENTS.md`
- `.vibe/config.toml`
- `.vibe/skills/semantic-repo-search/SKILL.md`
- `.codex/config.toml`
- `.codex/hooks.json`
- `.claude/settings.json`
- `.gemini/settings.json`
- `.mcp.json`

If the target directory is not already a git repo, `vibe-rag init` also runs `git init`.

This is the path that matters for release quality. Generated repos pin the resolved installed `vibe-rag` binary, so you should treat the installed-wheel scaffold as the default reality, not the source checkout.

## 3. Configure the MCP Server

Each client has its own config format. Here is the Vibe example (`.vibe/config.toml`):

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

Claude Code (`.claude/settings.json`), Codex (`.codex/config.toml` + `.codex/hooks.json`), and Gemini CLI (`.gemini/settings.json` + `.mcp.json`) have equivalent generated configs. `vibe-rag init` writes all four.

Notes:

- Durable user memory is stored automatically in `~/.vibe/memory.db`.
- Project code and docs index stay in `.vibe/index.db`.
- Embeddings use OpenRouter, defaulting to `perplexity/pplx-embed-v1-4b` and `2560` dimensions.
- Retrieval stays project-scoped by default, including user-memory results used by session bootstrap.
- `vibe-rag` exposes bare MCP tool names like `load_session_context`, `index_project`, `search`, `remember`, and `project_status`.
- When a client's MCP config names the server (e.g. `memory`), those same tools may appear prefixed as `memory_load_session_context`, `memory_index_project`, `memory_search`, `memory_remember`, and `memory_project_status`.

Optional embed/storage env block:

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
  RAG_DB = "/path/to/.vibe/index.db",
  RAG_USER_DB = "/path/to/.vibe/memory.db"
}
```

Helper commands:

```bash
vibe-rag doctor
vibe-rag doctor --fix
vibe-rag hook-session-start --format codex
```

`vibe-rag doctor` verifies the startup path, not just the provider:

- effective project id
- MCP command resolution from project config
- Codex `SessionStart` hook configuration
- project and user sqlite readability
- embedding provider reachability
- client trust state (Vibe, Codex, and others where applicable)
- stale index warnings

`doctor` does not execute repo-configured hook commands. It only inspects the configured command and reports trust state.

If you want to refresh the local index outside the client loop, run:

```bash
vibe-rag reindex
```

If the embedding profile changes or `doctor` reports an incompatible index, run:

```bash
vibe-rag reindex --full
# or
vibe-rag reset-index
```

If `RAG_OR_API_KEY` is missing, `vibe-rag` reports a provider configuration error.

## 3A. Per-Client Scaffolding

`vibe-rag init` writes config for all four clients:

- `.vibe/config.toml` and `.vibe/skills/semantic-repo-search/SKILL.md` (Vibe)
- `.codex/config.toml` and `.codex/hooks.json` (Codex)
- `.claude/settings.json` (Claude Code)
- `.gemini/settings.json` and `.mcp.json` (Gemini CLI)

Each client config does the same things:

- registers `vibe-rag serve` as an MCP server
- runs `vibe-rag hook-session-start --format <client>` at session start
- pins the resolved `vibe-rag` binary path captured at scaffold time so client startup does not depend on `PATH` ordering

Generated Codex config also sets `suppress_unstable_features_warning = true`.

For this maintainer repo specifically, the tracked `.vibe/`, `.codex/`, `.claude/`, and `.mcp.json` files use `scripts/run-vibe-rag` instead. That repo-local runner is only for the source checkout; generated repos pin the resolved installed `vibe-rag` path directly.

## 4. Trust the Repo

If your client prompts for trust (Vibe, Codex, or others), trust the real resolved path.

This matters for:

- project MCP config
- project skills
- local indexing

## 5. Smoke Test

Run:

```text
load session context for understanding this repo
index this project
search the code for config loading
search docs for setup steps
remember that config is loaded from the project root
search memory for config is loaded from the project root
```

Expected:

- indexing succeeds
- code search returns a useful file
- docs search returns a useful chunk
- memory search finds what you just stored

If you are calling MCP tools programmatically, expect structured responses with `ok`, `results`, and structured `error` payloads instead of freeform strings.

## Validation Bar

This is the acceptance bar for the product:

1. install the packaged binary
2. scaffold a repo from that binary
3. verify session-start context and retrieval from the generated client config

If you only proved the source checkout path, you did not prove the product.

## 6. Check Local State

```bash
vibe-rag status
```

Expected shape:

- code chunk count
- doc chunk count
- project memory count
- user memory count

## 7. Troubleshooting

If code or docs search is empty:

- trust the repo
- run `index this project`
- or run `vibe-rag reindex`
- if the index is incompatible, run `vibe-rag reindex --full` or `vibe-rag reset-index`
- run `vibe-rag doctor`
- verify `RAG_OR_API_KEY` is set and `RAG_OR_EMBED_DIM` is `2560` when needed

If `vibe-rag doctor` reports stale state:

- run `vibe-rag reindex`
- make sure you are in the intended repo root
- check whether git `HEAD` changed since the last index

If `vibe-rag doctor` reports an incompatible index:

- run `vibe-rag reindex --full` or `vibe-rag reset-index`
- then rerun `vibe-rag doctor`

If memory search is empty:

- store one memory explicitly with `remember ...`
- run `vibe-rag status`
- check that `~/.vibe/memory.db` exists

If project ids feel wrong:

- restart your client session
- make sure you are in the intended repo root

If packaged behavior looks stale:

```bash
uv tool install --upgrade --python 3.12 vibe-rag
```

## 8. Local Retrieval Evals

To evaluate real repos in `~/dev` without mutating their normal index state:

1. Copy `evals/local_repos.toml.example` to `evals/local_repos.toml`
2. Point each `path` at a real local repo such as `~/dev/vibe-rag`, `~/dev/other-project`, or `~/dev/another-project`
3. Run:

```bash
uv run python scripts/run_retrieval_eval.py evals/local_repos.toml
```

The script creates temporary project/user sqlite DBs for each repo under test.

To inspect the latest saved artifact summary later without rerunning the eval:

```bash
uv run python scripts/run_retrieval_eval.py evals/local_repos.toml --summary
```

To inspect the recent trend line for a manifest:

```bash
uv run python scripts/run_retrieval_eval.py evals/core_repos.toml --trends --trend-limit 5
```

To snapshot and trend the maintainer repo's real persistent memory:

```bash
uv run python scripts/run_retrieval_eval.py --persistent-memory --repo-path .
uv run python scripts/run_retrieval_eval.py --persistent-memory-summary --repo-path .
uv run python scripts/run_retrieval_eval.py --persistent-memory-trends --repo-path . --trend-limit 5
```

To render a cut-ready release-evidence report from the latest retrieval and maintainer-memory artifacts:

```bash
uv run python scripts/run_retrieval_eval.py evals/core_repos.toml --release-evidence --repo-path . --trend-limit 3
```

The saved summary now shows:

- repo and task pass counts
- per-repo index timing
- fallback-query usage
- irrelevant-hit noise totals
- memory cleanup and duplicate-auto-memory totals
- whether the artifact is stale relative to the repo's current git `HEAD`

The trend view rolls those metrics across multiple saved artifacts so you can see whether retrieval quality and memory hygiene are improving or drifting over time.
The persistent-memory snapshot path uses the real repo and `~/.vibe/memory.db`, so it gives you a separate maintainer-memory history instead of eval-local temp DB numbers.
One-turn auto session captures now also infer stronger memory kinds and can surface merge/supersede suggestions when a new capture looks like an update to an existing durable memory.

To inspect memory hygiene in the current repo after a session-heavy run, call:

```text
project_status include_memory_health=true
```

To preview or apply duplicate-auto-memory cleanup:

```text
cleanup_duplicate_auto_memories
cleanup_duplicate_auto_memories apply=true
```

Low-signal auto session summaries are now skipped earlier during capture, so the report should stay focused on more durable memories.
