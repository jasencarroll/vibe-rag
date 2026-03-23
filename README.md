# vibe-rag

`vibe-rag` is a memory and semantic search MCP server for Vibe.

Vibe is the first-class client.

Generated repos also include experimental scaffolding for:

- Codex
- Claude Code
- Gemini CLI

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
- Ollama
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
uv tool install vibe-rag@0.0.17
```

If your machine defaults `uv` tools to Python 3.13:

```bash
uv tool install --python 3.12 vibe-rag
```

## Quick Start

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
command = "vibe-rag"
args = ["serve"]

[background_mcp_hook]
enabled = true
tool_name = "memory_load_session_context"
task_arg = "task"
```

Optional Ollama overrides:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
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

## Experimental Codex And Claude Code Scaffolding

`vibe-rag init` also writes:

- `.codex/config.toml`
- `.codex/hooks.json`
- `.claude/settings.json`
- `.gemini/settings.json`
- `.mcp.json`

These files use:

- `vibe-rag serve` for MCP tools
- `vibe-rag hook-session-start --format <client>` for session-start context injection

Current support level:

- Vibe: first-class
- Codex: experimental
- Claude Code: experimental
- Gemini CLI: experimental

## Storage Model

| Layer | Purpose | Storage | Tools |
| --- | --- | --- | --- |
| Project index | semantic code and docs retrieval in the current repo | `.vibe/index.db` | `index_project`, `search_code`, `search_docs` |
| User memory | durable cross-session memory | `~/.vibe/memory.db` | `remember`, `search_memory`, `forget`, `load_session_context` |

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

## Docs

- [Setup Guide](docs/setup-guide.md)
- [User Guide](docs/user-guide.md)
- [Maintainer Guide](AGENTS.md)
