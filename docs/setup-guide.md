# vibe-rag Setup Guide

This guide is for a developer who wants the whole experience working on the first pass:

- packaged `vibe`
- packaged `vibe-rag`
- PostgreSQL with `pgvector`
- project-local Vibe config
- background session bootstrap

## Known-Good Outcome

At the end of this guide, all of these should be true:

- `vibe --version` works from the forked install
- `vibe-rag --version` works from the packaged install
- `psql "$DATABASE_URL" -c '\dx'` shows `vector`
- `vibe-rag init demo-memory-project` succeeds
- `index this project` indexes code and docs
- `remember ...` returns a memory id
- a fresh Vibe session can answer from prior memory/doc context

## Fast Path

If you already have all of these working:

- local PostgreSQL is running
- `psql` works
- `CREATE EXTENSION vector;` works in your target database

skip directly to:

- [4. Choose a DATABASE_URL](#4-choose-a-database_url)

If any of those are not true, keep going from section 1.

## 1. Install the Tools

Install the Vibe fork that includes the background MCP hook:

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

If your `uv` tool environment defaults to Python 3.13, use:

```bash
uv tool install --python 3.12 vibe-rag
```

## 2. Prepare PostgreSQL

You need a local PostgreSQL server with the `vector` extension available.

This guide assumes:

- PostgreSQL is already running locally
- you can connect with `psql`
- `pgvector` is installed into that Postgres instance

If you prefer a simple local setup, use Postgres.app or any local PostgreSQL distribution that includes `pgvector` or lets you install it.

Quick sanity checks before continuing:

```bash
psql --version
pg_isready
```

If `psql` is missing, stop here and install PostgreSQL client tools first.

## 2A. Known-Good Local PostgreSQL Setups

Pick one path and finish it before continuing.

### Option A: Postgres.app on macOS

This is the easiest path if you want the least friction.

1. Install and launch Postgres.app.
2. Make sure its binaries are on your `PATH`:

```bash
echo 'export PATH="/Applications/Postgres.app/Contents/Versions/latest/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

3. Verify:

```bash
psql --version
pg_isready
```

4. If Postgres.app already includes `pgvector` in your install, great. If not, stop here and install a PostgreSQL distribution that does.

### Option B: Homebrew PostgreSQL

Use this if you prefer managing PostgreSQL with Homebrew.

Install PostgreSQL:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Make sure the client binaries are on your `PATH`:

```bash
echo 'export PATH="$(brew --prefix postgresql@16)/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify:

```bash
psql --version
pg_isready
```

Important:

- PostgreSQL alone is not enough.
- You still need the `vector` extension available inside that server.

If `CREATE EXTENSION vector` fails later, your PostgreSQL install does not have `pgvector` available yet.

### Quick Decision Rule

- If you want the simplest GUI-first local DB setup, use Postgres.app.
- If you already use Homebrew for local infra, use Homebrew PostgreSQL.
- If neither setup can run `CREATE EXTENSION vector`, do not continue until that is fixed.

## 3. Create the Database

Open `psql` against your local server:

```bash
psql postgres
```

Create a role if you do not already have one:

```sql
CREATE ROLE jasen LOGIN SUPERUSER;
```

If you already have a local role you use for development, reuse that instead of creating `jasen`.

Create the database:

```sql
CREATE DATABASE vibe_memory OWNER jasen;
```

Connect to it:

```sql
\c vibe_memory
```

Enable `pgvector`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If this fails with “extension `vector` is not available”, the problem is your PostgreSQL installation, not `vibe-rag`.

Check that it worked:

```sql
\dx
```

You should see `vector` in the installed extensions list.

## 4. Choose a DATABASE_URL

Examples:

```bash
export DATABASE_URL=postgresql://jasen@localhost:5432/vibe_memory
```

If you use password auth:

```bash
export DATABASE_URL=postgresql://username:password@localhost:5432/vibe_memory
```

Verify it:

```bash
psql "$DATABASE_URL" -c '\dx'
```

If this command cannot connect, do not move on to the Vibe setup yet. Fix database access first.

## 5. Set Your Mistral Key

```bash
export MISTRAL_API_KEY=your_real_key_here
```

Sanity check:

```bash
printenv MISTRAL_API_KEY
printenv DATABASE_URL
```

If either variable is empty, Vibe will fail later in a less obvious way.

## 6. Scaffold a Project

```bash
mkdir -p ~/tmp
cd ~/tmp
vibe-rag init demo-memory-project
cd demo-memory-project
```

That gives you:

- `AGENTS.md`
- `.vibe/config.toml`
- `.vibe/skills/semantic-repo-search/SKILL.md`

If `vibe-rag init` fails here, check:

```bash
vibe-rag --version
```

You want `0.0.11` or later.

## 7. Configure `.vibe/config.toml`

Start from the generated file and make it explicit:

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
  DATABASE_URL = "postgresql://jasen@localhost:5432/vibe_memory"
}

[background_mcp_hook]
enabled = true
tool_name = "memory_load_session_context"
task_arg = "task"
```

Important:

- put both `MISTRAL_API_KEY` and `DATABASE_URL` inside the MCP server `env`
- do not rely on shell inheritance if you launch Vibe from mixed terminals or apps
- `background_mcp_hook` belongs at top level in this config

This is the most common configuration failure in real use:

- shell env is set
- Vibe launches from somewhere else
- MCP server starts without `DATABASE_URL`
- `remember` appears to work, but durable memory is not shared

Putting both vars in the MCP `env` block avoids that.

## 8. Trust the Repo in Vibe

Vibe ignores project-local config and skills for untrusted folders.

When Vibe asks whether to trust the folder, say yes.

If you are testing under `/tmp` on macOS, the real path may resolve under `/private/tmp/...`. Trust the resolved path, not just the symlinked one.

You can check the real path with:

```bash
pwd -P
```

## 9. First Project Smoke Test

Create a little code and docs surface:

```bash
mkdir -p src docs
cat > src/greetings.py <<'EOF'
def greeting_for(name: str) -> str:
    return format_greeting(name)

def format_greeting(name: str) -> str:
    return f"Hello, {name}!"
EOF

cat > docs/deployment.md <<'EOF'
# Deployment

Deploy with `uv run app` after tests pass.
EOF
```

Run Vibe:

```bash
vibe
```

Use these prompts:

```text
load session context for understanding this demo project
index this project
search the code for the helper that formats the greeting
search docs for deployment steps
remember that greeting_for delegates to format_greeting
search memory for format_greeting
```

You should see:

- code search find `format_greeting`
- docs search find `docs/deployment.md`
- `remember` return a UUID-like memory id if pgvector is active

If `remember` does not return a UUID-like id, inspect the MCP config again. In practice that usually means `DATABASE_URL` did not reach the `vibe-rag` process.

## 10. Verify the Background Hook

Close Vibe and start a fresh session in the same repo.

Ask:

```text
What durable note exists for this project, and what deployment step is documented?
```

If the hook is working, Vibe can answer from injected context before you manually invoke memory tools.

If it cannot, check this order:

1. the repo is trusted
2. the repo has been indexed
3. the MCP `env` has both `MISTRAL_API_KEY` and `DATABASE_URL`
4. `[background_mcp_hook]` exists in `.vibe/config.toml`

## 11. Check Database State from psql

Connect:

```bash
psql "$DATABASE_URL"
```

Useful commands:

```sql
\dt
\dx
SELECT COUNT(*) FROM memories;
SELECT id, project_id, memory_kind, summary, created_at
FROM memories
ORDER BY created_at DESC
LIMIT 20;
```

You can also inspect whether memories are being superseded:

```sql
SELECT id, summary, supersedes, superseded_by
FROM memories
ORDER BY created_at DESC
LIMIT 20;
```

If you want to inspect the raw content:

```sql
SELECT id, content
FROM memories
ORDER BY created_at DESC
LIMIT 5;
```

## 12. Check Local Index State

From the repo root:

```bash
vibe-rag status
```

That shows:

- local sqlite index path
- code chunk count
- doc chunk count
- pgvector memory count if `DATABASE_URL` is set

Healthy output should show non-zero code/doc counts after indexing.

## Troubleshooting

### `vibe-rag init` fails

You want `v0.0.11` or later:

```bash
vibe-rag --version
```

### Vibe ignores project-local config

- trust the repo
- restart Vibe
- on macOS under `/tmp`, trust the resolved `/private/tmp/...` path

### `remember` works but memory is not shared across sessions

Check that `DATABASE_URL` is inside the MCP server `env` block in `.vibe/config.toml`.

Also check that the new session is running in the same trusted repo root.

### `search_code` or `search_docs` returns nothing useful

Re-index:

```text
index this project
```

### `load session context` seems empty

Check both:

- the project has been indexed
- the memory MCP server has both `MISTRAL_API_KEY` and `DATABASE_URL`

Also remember that `load_session_context` is retrieval, not magic. If you have not stored durable notes yet, code/docs may still be useful while memory is sparse.

### PostgreSQL extension error

Run:

```bash
psql "$DATABASE_URL" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

If that fails, your Postgres install does not yet have `pgvector` available.

### PyPI release exists but `uv tool install vibe-rag@...` fails

This can happen for a few minutes after a successful GitHub publish workflow.

Wait and retry.

## Recommended Real-World Rollout

1. Test in a disposable repo.
2. Test in a repo you know well.
3. Add durable notes for architecture conventions.
4. Start opening new tasks with `load session context for ...`.
