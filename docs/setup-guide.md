# vibe-rag Setup Guide

Target state:

- packaged `vibe-rag`
- project-local config for your chosen client
- local durable memory in `~/.vibe/memory.db`
- session bootstrap where supported
- Ollama embeddings with `qwen3-embedding:0.6b`
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

Start Ollama and pull the default embedding model:

```bash
vibe-rag setup-ollama
```

If you want the most complete Vibe integration, install the Vibe fork:

```bash
uv tool uninstall mistral-vibe || true
uv tool install git+https://github.com/jasencarroll/mistral-vibe.git
vibe --version
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

## 3. Configure the MCP Server

Use a project config like this:

```toml
active_model = "devstral-2"
skill_paths = [".vibe/skills"]

[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
[background_mcp_hook]
enabled = true
tool_name = "memory_load_session_context"
task_arg = "task"

[session_memory_hook]
enabled = true
tool_name = "memory_save_session_memory"
summary_tool_name = "memory_save_session_summary"
```

Notes:

- Durable user memory is stored automatically in `~/.vibe/memory.db`.
- Project code and docs index stay in `.vibe/index.db`.
- Ollama is the default embedding provider.

If Ollama is running on a non-default host, use:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
env = {
  VIBE_RAG_OLLAMA_HOST = "http://192.168.1.5:11434",
  VIBE_RAG_EMBEDDING_DIMENSIONS = "1024"
}
```

Optional:

- `VIBE_RAG_OLLAMA_HOST`

If `VIBE_RAG_OLLAMA_HOST` is not set, `vibe-rag` checks `OLLAMA_HOST`, then `localhost`, then `127.0.0.1`.

Helper commands:

```bash
vibe-rag doctor
vibe-rag doctor --fix
vibe-rag setup-ollama
vibe-rag hook-session-start --format codex
```

`vibe-rag doctor` verifies the startup path, not just the provider:

- effective project id
- MCP command resolution from project config
- Codex `SessionStart` hook execution
- project and user sqlite readability
- embedding provider reachability
- Vibe and Codex trust state
- stale index warnings

If you want to refresh the local index outside the client loop, run:

```bash
vibe-rag reindex
```

If Ollama is unavailable but a hosted provider key is already configured, `vibe-rag` now falls back to that hosted provider automatically.

## 3A. Optional Codex And Claude Code Scaffolding

`vibe-rag init` also writes:

- `.codex/config.toml`
- `.codex/hooks.json`
- `.claude/settings.json`
- `.gemini/settings.json`
- `.mcp.json`

Those files do two things:

- register `vibe-rag serve` as an MCP server for the client
- run `vibe-rag hook-session-start --format <client>` at session start
- rely on `vibe-rag` being available on `PATH` rather than an absolute binary path

Generated Codex config also sets `suppress_unstable_features_warning = true`.

For this maintainer repo specifically, the tracked `.vibe/`, `.codex/`, `.claude/`, and `.mcp.json` files use `scripts/run-vibe-rag` instead. That repo-local runner is only for the source checkout; generated repos still use `vibe-rag` directly.

Current support level:

- `vibe-rag serve`: core identity
- Vibe: most complete integration
- Codex: experimental
- Claude Code: experimental
- Gemini CLI: experimental

## 4. Trust the Repo

If Vibe prompts for trust, trust the real resolved path.

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
- run `vibe-rag doctor`
- make sure Ollama is running and `qwen3-embedding:0.6b` is pulled

If `vibe-rag doctor` reports stale state:

- run `vibe-rag reindex`
- make sure you are in the intended repo root
- check whether git `HEAD` changed since the last index

If memory search is empty:

- store one memory explicitly with `remember ...`
- run `vibe-rag status`
- check that `~/.vibe/memory.db` exists

If project ids feel wrong:

- restart `vibe`
- make sure you are in the intended repo root

If packaged behavior looks stale:

```bash
uv tool install --upgrade --python 3.12 vibe-rag
```

## 8. Local Retrieval Evals

To evaluate real repos in `~/dev` without mutating their normal index state:

1. Copy `evals/local_repos.toml.example` to `evals/local_repos.toml`
2. Point each `path` at a real local repo such as `~/dev/vibe-rag`, `~/dev/mistral-vibe`, or `~/dev/codex`
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
memory_quality_report
```

To preview or apply duplicate-auto-memory cleanup:

```text
cleanup_duplicate_auto_memories
cleanup_duplicate_auto_memories apply=true
```

Low-signal auto session summaries are now skipped earlier during capture, so the report should stay focused on more durable memories.
