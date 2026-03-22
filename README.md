# vibe-rag

Local semantic code search and persistent memory for [Mistral Vibe](https://docs.mistral.ai/mistral-vibe/). An MCP server that gives Vibe the ability to understand your codebase by meaning, remember things across sessions, and search your docs — all stored in a single local sqlite file. No external database. No cloud. Fully local.

## Install

```bash
uv tool install vibe-rag
```

Requires Python 3.12+ and a [Mistral API key](https://console.mistral.ai/api-keys).

## Quick start

```bash
# Create a new project
vibe-rag init my-project
cd my-project

# Launch Vibe
vibe --agent builder
```

Inside Vibe:

```
index this project
search the code for authentication handling
search docs for deployment instructions
remember we decided to use JWT for auth
what did we decide about auth?
```

## What it does

Six MCP tools, available to Vibe as soon as the server connects:

| Tool | What it does | Embedding model |
|------|-------------|-----------------|
| `index_project` | Indexes code and docs for semantic search | codestral-embed (code), mistral-embed (docs) |
| `search_code` | Finds code by meaning, not just keywords | codestral-embed |
| `search_docs` | Finds documentation by meaning | mistral-embed |
| `remember` | Stores a memory with a vector embedding | mistral-embed |
| `search_memory` | Recalls memories by semantic similarity | mistral-embed |
| `forget` | Deletes a memory by ID | — |

Everything lives in `.vibe/index.db` — a single [sqlite-vec](https://github.com/asg017/sqlite-vec) file, gitignored, local to the project. No Postgres. No cloud vector DB. No account needed beyond a Mistral API key.

## API key

One key: `MISTRAL_API_KEY` from [console.mistral.ai](https://console.mistral.ai/api-keys). It's used for both embedding models:

- **`codestral-embed`** — optimized for code, used by `index_project` and `search_code`
- **`mistral-embed`** — optimized for natural language, used by `search_docs`, `remember`, and `search_memory`

The key is passed via the MCP server's `env` config. `vibe-rag init` sets this up automatically if `MISTRAL_API_KEY` is in your environment.

## How indexing works

`index_project` does two passes:

**Code** (`.py`, `.js`, `.ts`, `.rs`, `.go`, `.java`, `.c`, `.cpp`, etc.):
- Chunked using tree-sitter (AST-aware — splits on functions, classes, methods)
- Falls back to a 60-line sliding window for unsupported languages
- Embedded with `codestral-embed`

**Docs** (`.md`, `.txt`, `.rst`):
- Markdown: split on `##` headers, sub-split large sections on paragraphs
- Plain text / RST: 2000-char sliding window
- Embedded with `mistral-embed`

Both are stored in the same sqlite-vec database and searched with the appropriate model.

## Setup for an existing project

If you already have a Vibe project (or any existing repository), initialize vibe-rag scaffolding:

```bash
cd your-existing-project
vibe-rag init
```

This creates:
- `.vibe/config.toml` — MCP server configuration
- `.vibe/index.db` — local vector database
- `.gitignore` entry for `.vibe/index.db`
- `AGENTS.md` — coding rules template

If you prefer to configure manually instead, add this to your `.vibe/config.toml`:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
env = { "MISTRAL_API_KEY" = "your-key-here" }
```

## CLI

```bash
vibe-rag init [name]    # Create a new project with MCP server configured
vibe-rag status         # Show code chunks, doc chunks, and memory count
vibe-rag serve          # Start the MCP server (called by Vibe, not you)
```

## Agent profiles

`vibe-rag init` installs agent profiles to `~/.vibe/agents/` on first run:

| Agent | Description |
|-------|-------------|
| `builder` | Auto-approves all tools. For trusted projects. |
| `reviewer` | Read-only. For code review. |
| `planner` | Read-only. For exploration and planning. |
| `devops` | Auto-approves with Docker/psql/gh access. |
| `researcher` | Read-only subagent for background research. |

Use with `vibe --agent builder`.

## Development

```bash
git clone https://github.com/jasencarroll/vibe-rag.git
cd vibe-rag
uv sync --python 3.12 --all-extras
uv run --python 3.12 pytest tests/ -v
```

## License

MIT
