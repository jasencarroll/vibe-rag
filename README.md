# vibe-rag

`vibe-rag` is a memory and semantic search MCP server for Vibe.

It adds:

- semantic code search
- semantic docs search
- durable session memory

Storage is local and simple:

- project index: `.vibe/index.db`
- user memory: `~/.vibe/memory.db`

No external database is required.

## What You Need

- Python 3.12+
- `uv`
- a Mistral API key
- Vibe

Use Python 3.12 for `uv tool install`. In this environment, `tree-sitter-languages` does not have cp313 wheels.

## Required Vibe Fork

Use the fork that contains the background MCP session bootstrap hook:

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
uv tool install vibe-rag@0.0.14
```

If your machine defaults `uv` tools to Python 3.13:

```bash
uv tool install --python 3.12 vibe-rag
```

## Quick Start

```bash
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
command = "vibe-rag"
args = ["serve"]
env = { MISTRAL_API_KEY = "your_mistral_api_key" }

[background_mcp_hook]
enabled = true
tool_name = "memory_load_session_context"
task_arg = "task"
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
- `search the code for ...` returns a relevant file or snippet
- `search docs for ...` returns a relevant text chunk
- `remember ...` returns a memory id
- a fresh Vibe session can answer from prior context

## Storage Model

| Layer | Purpose | Storage | Tools |
| --- | --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` | `index_project`, `search_code`, `search_docs` |
| User memory | durable cross-session memory | `~/.vibe/memory.db` | `remember`, `search_memory`, `forget`, `load_session_context` |

## Docs

- [Setup Guide](docs/setup-guide.md)
- [User Guide](docs/user-guide.md)
- [Maintainer Guide](AGENTS.md)
