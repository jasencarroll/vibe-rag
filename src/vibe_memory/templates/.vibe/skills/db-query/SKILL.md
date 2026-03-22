---
name: db-query
description: Query and inspect PostgreSQL databases using psql
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - glob
  - run_command
  - ask_user_question
---

# DB Query

Query and inspect PostgreSQL databases using `psql`.

## Workflow

1. Ask for connection details if not provided (or check for `DATABASE_URL` env var)
2. Test connection: `psql <conn> -c "SELECT 1"`
3. Execute the requested operation
4. Format and present results

## Common Operations

### Inspect Schema
- `psql <conn> -c "\dt"` — list tables
- `psql <conn> -c "\d <table>"` — describe table
- `psql <conn> -c "\di"` — list indexes
- `psql <conn> -c "\df"` — list functions

### Query
- `psql <conn> -c "SELECT ..."` — run a query
- `psql <conn> -c "\x" -c "SELECT ..."` — expanded output for wide rows
- Add `LIMIT 20` to exploratory queries

### Migrations
- Check migration files in the project for context
- `psql <conn> -c "SELECT * FROM schema_migrations ORDER BY version DESC LIMIT 10"`

### Data Export
- `psql <conn> -c "COPY (SELECT ...) TO STDOUT WITH CSV HEADER"`

## Guidelines

- NEVER run `DROP`, `DELETE`, `TRUNCATE`, or `ALTER` without explicit user confirmation
- Always `LIMIT` exploratory queries to avoid dumping huge tables
- Use `\x` (expanded display) for tables with many columns
- Prefer read-only operations unless the user explicitly asks to modify data
- Check for `DATABASE_URL` in `.env` or environment before asking for connection details
- Never display passwords in output — use env vars for connection strings
