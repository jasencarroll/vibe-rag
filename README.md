# vibe-rag

Semantic code search and persistent memory for [Mistral Vibe](https://docs.mistral.ai/mistral-vibe/). An MCP server that gives Vibe the ability to index your codebase, remember things, and recall them later — all stored locally in a single sqlite file.

## What it does

Five MCP tools, available to Vibe as soon as the server connects:

| Tool | What it does |
|------|-------------|
| `index_project` | Indexes source files for semantic code search using Codestral Embed |
| `search_code` | Finds code by meaning, not just keywords |
| `remember` | Stores a memory with a vector embedding |
| `search_memory` | Recalls memories by semantic similarity |
| `forget` | Deletes a memory by ID |

Everything lives in `.vibe/index.db` — a single sqlite-vec file, gitignored, local to the project.

## Install

```bash
uv tool install vibe-rag
```

Requires Python 3.12+ and a [Mistral API key](https://console.mistral.ai/api-keys) with embeddings access (not a Vibe-scoped key).

## Quick start

```bash
# Create a new project with vibe-rag configured
cd ~/dev
vibe-rag init my-project
cd my-project

# Launch Vibe
vibe --agent builder
```

Inside Vibe:

```
index this project
search the code for authentication handling
remember we decided to use JWT for auth
what did we decide about auth?
```

## Setup for an existing project

If you already have a Vibe project, add this to your `.vibe/config.toml`:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
env = { "MISTRAL_API_KEY" = "your-mistral-console-key" }
```

Add `.vibe/index.db` to your `.gitignore`.

## API key

vibe-rag needs a Mistral **console** API key (not a Vibe-scoped key) because it calls the embeddings API. The key is passed via the `MISTRAL_API_KEY` environment variable in the MCP server config.

Two embedding models are used:
- **`codestral-embed`** for code indexing and search
- **`mistral-embed`** for memory storage and recall

## CLI commands

```bash
vibe-rag init [name]    # Create a new project with MCP server configured
vibe-rag status         # Show code chunks and memory count for current project
vibe-rag serve          # Start the MCP server (called by Vibe, not you)
```

## How it works

- **Code search**: `index_project` walks your source files, chunks them using tree-sitter (AST-aware) with a sliding window fallback, embeds each chunk with Codestral Embed, and stores them in sqlite-vec. `search_code` embeds your query and finds the closest chunks by cosine similarity.

- **Memory**: `remember` embeds whatever you tell it and stores it. `search_memory` finds the closest memories to your query. `forget` deletes by ID.

- **Storage**: Everything is in `.vibe/index.db` — a sqlite database with the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension for vector search. No external database needed.

## Agent profiles

`vibe-rag init` installs these agent profiles to `~/.vibe/agents/` on first run:

| Agent | Description |
|-------|-------------|
| `builder` | Auto-approves all tools. For trusted projects. |
| `reviewer` | Read-only. For code review. |
| `planner` | Read-only. For exploration and planning. |
| `devops` | Auto-approves with Docker/psql/gh access. |
| `researcher` | Read-only subagent for background research. |

Use with `vibe --agent builder`.

## Project structure

```
src/vibe_rag/
├── cli.py              # init, status, serve
├── server.py           # FastMCP server with 5 tools
├── config.py           # Project ID resolution
├── db/sqlite.py        # sqlite-vec for code chunks + memories
└── indexing/
    ├── embedder.py     # Mistral + Codestral embedding client
    ├── code_chunker.py # tree-sitter AST + sliding window
    └── doc_chunker.py  # Markdown/text chunking
```

## Development

```bash
git clone https://github.com/jasencarroll/vibe-rag.git
cd vibe-rag
uv sync --python 3.12 --all-extras
uv run --python 3.12 pytest tests/ -v
```

## License

MIT
