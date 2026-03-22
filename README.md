# vibe-rag

`vibe-rag` is a memory and semantic search MCP server for Vibe.

It adds:

- semantic code search
- semantic docs search
- durable project memory across sessions

Code and docs search live in local sqlite in the repo. Durable memory can live in PostgreSQL with `pgvector`.

## What You Need

- Python 3.12+
- `uv`
- a Mistral API key
- Vibe
- local PostgreSQL with `pgvector` if you want durable cross-session memory

Notes:

- Use Python 3.12 for `uv tool install`. In this environment, `tree-sitter-languages` does not have cp313 wheels.
- If you only install `vibe-rag` and skip PostgreSQL, semantic code/docs search still works, but durable cross-session memory will be weaker and local-only.
- Vibe project-local config only works in trusted folders.

## Required Vibe Fork

Use the fork that contains the background MCP session bootstrap hook:

- Repo: `https://github.com/jasencarroll/mistral-vibe`

Install it with `uv`:

```bash
uv tool uninstall mistral-vibe || true
uv tool install git+https://github.com/jasencarroll/mistral-vibe.git
vibe --version
```

Expected: `vibe --version` prints `2.5.0` or later.

## Install vibe-rag

```bash
uv tool install vibe-rag
vibe-rag --version
```

Pinned release:

```bash
uv tool install vibe-rag@0.0.11
```

If your machine defaults `uv` tools to Python 3.13, install explicitly with Python 3.12:

```bash
uv tool install --python 3.12 vibe-rag
```

## Quick Start

Use this order:

1. Install the Vibe fork.
2. Install `vibe-rag`.
3. Make sure PostgreSQL plus `pgvector` work.
4. Scaffold a project.
5. Add `MISTRAL_API_KEY`, `DATABASE_URL`, and `background_mcp_hook` to `.vibe/config.toml`.
6. Launch Vibe and run the smoke-test prompts.

Scaffold:

```bash
vibe-rag init my-project
cd my-project
```

Config:

```toml
active_model = "devstral-2"
skill_paths = [".vibe/skills"]

[[mcp_servers]]
name = "memory"
transport = "stdio"
command = "vibe-rag"
args = ["serve"]
env = {
  MISTRAL_API_KEY = "your_mistral_api_key",
  DATABASE_URL = "postgresql://localhost:5432/vibe_memory"
}

[background_mcp_hook]
enabled = true
tool_name = "memory_load_session_context"
task_arg = "task"
```

3. Launch Vibe:

```bash
vibe
```

4. In your first session, use:

```text
load session context for understanding this repo
index this project
search the code for authentication handling
search docs for deployment instructions
remember that auth tokens are validated in the API gateway
```

If PostgreSQL is not already working, use:

- [Setup Guide](docs/setup-guide.md)

Success:

- `index this project` reports code and docs indexed
- `search the code for ...` returns a relevant file/snippet
- `search docs for ...` returns a markdown/text chunk
- `remember ...` returns a memory id
- a fresh Vibe session can answer from prior context if the background hook is configured

## How It Works

`vibe-rag` has two storage layers:

| Layer | Purpose | Storage | Tools |
| --- | --- | --- | --- |
| Project index | semantic code/docs search in the current repo | `.vibe/index.db` | `index_project`, `search_code`, `search_docs` |
| Durable memory | decisions, constraints, conventions, session carry-over | PostgreSQL via `pgvector` when `DATABASE_URL` is set | `remember`, `search_memory`, `forget`, `load_session_context` |

If `DATABASE_URL` is not set, memory falls back to local sqlite.

## Setup Guide

Full setup:

- [Setup Guide](docs/setup-guide.md)

Includes:

- installing the required Vibe fork
- creating the PostgreSQL database
- enabling `pgvector`
- exact `psql` commands
- configuring `.vibe/config.toml`
- first-run verification
- common failure modes

Shortest path:

- install the Vibe fork
- install `vibe-rag`
- create the PostgreSQL database and enable `vector`
- scaffold a demo repo
- run the smoke test prompts

If local PostgreSQL is not set up yet, start with the “Known-Good Local PostgreSQL Setups” section:

- Postgres.app
- Homebrew PostgreSQL

## Daily Workflow

Daily workflow:

- [User Guide](docs/user-guide.md)

Includes:

- when to index
- when to remember
- how to resume work later
- how to phrase prompts so Vibe actually uses memory

## Generated Scaffold

`vibe-rag init` creates:

- `AGENTS.md` with a memory-first workflow
- `.vibe/config.toml` scaffold
- `.vibe/skills/semantic-repo-search/SKILL.md`

The generated skill prefers:

- `memory_load_session_context`
- `memory_index_project`
- `memory_search_code`
- `memory_search_docs`
- `memory_search_memory`

before `grep`.

Expected:

- the developer adds `MISTRAL_API_KEY` and `DATABASE_URL` to `.vibe/config.toml`
- the repo is trusted in Vibe
- the first useful prompts are `load session context for ...` and `index this project`

## Core Commands

### CLI

```bash
vibe-rag init my-project
vibe-rag status
vibe-rag serve
```

### Inside Vibe

```text
index this project
load session context for continuing the auth refactor
search the code for where config is written
search docs for release steps
remember that invoice ids must stay human-readable
search memory for invoice decisions
forget memory <id>
```

## Configuration Notes

- `MISTRAL_API_KEY` is required for embeddings.
- `DATABASE_URL` is optional but strongly recommended.
- Vibe only loads project-local config and skills from trusted folders.
- If the repo is not trusted yet, trust it and restart Vibe.
- `background_mcp_hook` belongs in Vibe config, not in `vibe-rag`.

Common mistakes:

- setting `DATABASE_URL` in your shell but not in the MCP server `env`
- trusting `/tmp/...` while Vibe is actually resolving the repo as `/private/tmp/...`
- expecting background bootstrap without `[background_mcp_hook]`
- expecting cross-session memory after only running `index this project`

## Development

```bash
uv sync
uv run pytest
uv build
```

## Support

- Issues: `https://github.com/jasencarroll/vibe-rag/issues`
- Source: `https://github.com/jasencarroll/vibe-rag`
