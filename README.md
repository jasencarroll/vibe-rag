# vibe-rag

Local semantic code search and persistent memory for [Mistral Vibe](https://docs.mistral.ai/mistral-vibe/). An MCP server that gives Vibe the ability to understand your codebase by meaning, remember things across sessions, and search your docs — all stored in a single local sqlite file. No external database. No cloud. Fully local.

## v0.0.7 Highlights

- **🔒 Security:** API keys never stored in config files
- **🚀 Performance:** 5-10x faster file collection on large projects  
- **🐛 Reliability:** Fixed race conditions and error handling
- **📚 Language Support:** Complete mappings for all file types

## Install

```bash
# Latest stable release (v0.0.7)
uv tool install vibe-rag

# Or install specific version
uv tool install vibe-rag@0.0.7
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
search the code for authentication handling
search docs for deployment instructions
remember we decided to use JWT for auth
```

## Features

### Local semantic code search
- Index your entire codebase by meaning, not just text
- Search across all files using natural language
- Find relevant code chunks with context

### Persistent memory
- Remember decisions, insights, and context across sessions
- Search your memories semantically
- Hybrid storage: sqlite for local, pgvector for cross-repo (optional)

### Language support
- Python, JavaScript, TypeScript, Rust, Go, Java, C++, C, Ruby, PHP
- Swift, Kotlin, Scala, Bash, SQL, TOML, YAML, JSON
- Smart chunking with tree-sitter for supported languages

### Security
- API keys only passed via environment variables
- No credentials stored in configuration files
- Local-only by default (pgvector optional)

## Configuration

Set your Mistral API key:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

For cross-repo memory (optional):

```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/vibe_rag
```

`vibe-rag init` keeps `.vibe/config.toml` free of credentials. Launch Vibe from a shell or environment that already exports the variables you want the MCP server to inherit.

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

## License

MIT © 2026 Jasen Carroll

## Support

- Issues: [GitHub Issues](https://github.com/jasencarroll/vibe-rag/issues)
- Discussions: [GitHub Discussions](https://github.com/jasencarroll/vibe-rag/discussions)
- Source: [GitHub Repository](https://github.com/jasencarroll/vibe-rag)
