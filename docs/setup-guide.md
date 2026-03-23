# vibe-rag Setup Guide

Target state:

- packaged `vibe`
- packaged `vibe-rag`
- project-local config
- local durable memory in `~/.vibe/memory.db`
- background session bootstrap

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

## 2. Scaffold a Repo

```bash
vibe-rag init demo
cd demo
```

This writes:

- `AGENTS.md`
- `.vibe/config.toml`
- `.vibe/skills/semantic-repo-search/SKILL.md`

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
env = { MISTRAL_API_KEY = "your_mistral_api_key" }

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

- `MISTRAL_API_KEY` is required for embeddings.
- Durable user memory is stored automatically in `~/.vibe/memory.db`.
- Project code and docs index stay in `.vibe/index.db`.

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
- make sure `MISTRAL_API_KEY` is present in the MCP server `env`

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
