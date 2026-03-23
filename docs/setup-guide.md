# vibe-rag Setup Guide

Target state:

- packaged `vibe`
- packaged `vibe-rag`
- project-local config
- local durable memory in `~/.vibe/memory.db`
- background session bootstrap
- Ollama embeddings with `qwen3-embedding:0.6b`
- optional Codex and Claude Code scaffolding

## 1. Install the Tools

Install the Vibe fork:

```bash
uv tool uninstall mistral-vibe || true
uv tool install git+https://github.com/jasencarroll/mistral-vibe.git
vibe --version
```

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

Generated Codex config also sets `suppress_unstable_features_warning = true`.

Current support level:

- Vibe: first-class
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
- run `vibe-rag doctor`
- make sure Ollama is running and `qwen3-embedding:0.6b` is pulled

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
