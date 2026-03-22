# vibe-rag

Local semantic code search and persistent memory for [Mistral Vibe](https://docs.mistral.ai/mistral-vibe/). An MCP server that gives Vibe the ability to understand your codebase by meaning, remember things across sessions, and search your docs — all stored in a single local sqlite file. No external database. No cloud. Fully local.

## v0.0.10 Highlights

- **🧠 Session Bootstrap:** `load_session_context` pulls prior memories, code, and docs in one call
- **🗂 Structured Memory:** pgvector memories now support kinds, summaries, metadata, and supersession
- **🔌 Vibe Integration:** generated scaffold now teaches Vibe to use memory tools first
- **📘 User Guide:** day-to-day setup and prompt patterns are documented
- **🐛 Reliability:** pgvector/sqlite schemas upgrade cleanly and indexing handles stale hashes

## Install

```bash
# Latest stable release (v0.0.10)
uv tool install vibe-rag

# Or install specific version
uv tool install vibe-rag@0.0.10
```

Requires Python 3.12+ and a [Mistral API key](https://console.mistral.ai/api-keys).

## Quick start

```bash
# Create a new project
vibe-rag init my-project
cd my-project

# Launch Vibe
vibe
```

Inside Vibe:

```
index this project
load session context for continuing the auth refactor
search the code for authentication handling
search docs for deployment instructions
remember we decided to use JWT for auth
```

The generated scaffold includes:

- an `AGENTS.md` that explains the memory-first workflow
- a `.vibe/skills/semantic-repo-search` skill that steers Vibe toward `memory_index_project`,
  `memory_load_session_context`, `memory_search_code`, `memory_search_docs`, and `memory_search_memory` before exact-match tools

## Features

### Local semantic code search
- Index your entire codebase by meaning, not just text
- Search across all files using natural language
- Find relevant code chunks with context

### Persistent memory
- Remember decisions, insights, and context across sessions
- Search your memories semantically
- Hybrid storage: sqlite for local, pgvector for cross-repo (optional)
- Structured session bootstrap: `memory_load_session_context` can retrieve prior memories, code, and docs in one call

### Language support
- Python, JavaScript, TypeScript, Rust, Go, Java, C++, C, Ruby, PHP
- Swift, Kotlin, Scala, Bash, SQL, TOML, YAML, JSON
- Smart chunking with tree-sitter for supported languages

### Security
- `vibe-rag init` does not write credentials into generated config
- Credentials can be inherited from the launch shell or set explicitly in Vibe MCP config
- Local-only by default (pgvector optional)

## Configuration

`vibe-rag init` generates a credential-free `.vibe/config.toml`:

```toml
# .vibe/config.toml
active_model = "devstral-2"
skill_paths = [".vibe/skills"]

[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
```

For Vibe itself, there are two supported ways to provide credentials.

Option 1: export them before launching `vibe`:

```bash
export MISTRAL_API_KEY=your_api_key_here
export DATABASE_URL=postgresql://user:pass@localhost:5432/vibe_rag
vibe
```

Option 2: add them to the MCP server `env` block in your project-local `.vibe/config.toml` or global `~/.vibe/config.toml`. This is the most reliable option if Vibe is launched from different shells or GUI sessions:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
env = {
  MISTRAL_API_KEY = "your_api_key_here",
  DATABASE_URL = "postgresql://user:pass@localhost:5432/vibe_rag"
}
```

Notes:

- `MISTRAL_API_KEY` is required for indexing, code search, docs search, and memory embeddings.
- `DATABASE_URL` is optional and only enables cross-repo memory via pgvector.
- Use `.vibe/config.toml` for project-specific credentials and `~/.vibe/config.toml` for a global setup.
- For normal use, prefer the installed package entrypoint above.
- If you want Vibe to test the working tree instead of an installed release, point `command` at your repo venv, for example:

```toml
[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "/absolute/path/to/repo/.venv/bin/python"
args = ["-m", "vibe_rag.cli", "serve"]
env = {
  PYTHONPATH = "/absolute/path/to/repo/src",
  MISTRAL_API_KEY = "your_api_key_here",
  DATABASE_URL = "postgresql://user:pass@localhost:5432/vibe_rag"
}
```

Vibe only loads project skills and project MCP config from trusted folders. If the repo is not trusted yet, trust it first, then restart Vibe and run `index this project`.

## Architecture

- **Local-first**: SQLite vector database for code search
- **Hybrid option**: PostgreSQL pgvector for cross-repo memories
- **Modular**: Separate components for db, indexing, tools
- **MCP protocol**: Standard Mistral Vibe agent communication

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Build package
uv build
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## User Guide

See [docs/user-guide.md](docs/user-guide.md) for the day-to-day workflow:

- how to talk to Vibe across sessions
- when to use `remember` vs `index this project`
- where to put `MISTRAL_API_KEY` and `DATABASE_URL`
- how pgvector cross-session memory works

## License

MIT © 2026 Jasen Carroll

## Support

- Issues: [GitHub Issues](https://github.com/jasencarroll/vibe-rag/issues)
- Discussions: [GitHub Discussions](https://github.com/jasencarroll/vibe-rag/discussions)
- Source: [GitHub Repository](https://github.com/jasencarroll/vibe-rag)
