# AGENTS.md

## Stack

This is a Python-first monolithic project. The backend is the source of truth.

- **Runtime:** Python 3.12+
- **Package manager:** uv (never use pip, pip install, or python -m directly)
- **Framework:** Starlette (or FastAPI if needed — both are ASGI)
- **Frontend:** Bun + vanilla TypeScript + shadcn/ui, built into `templates/` as Jinja2 templates
- **Database:** PostgreSQL with asyncpg (or SQLite for local/embedded use)
- **Testing:** pytest with pytest-asyncio
- **Linting/Formatting:** ruff
- **Type checking:** pyright in strict mode

## Commands

```bash
# Dependencies
uv sync                          # Install all deps from pyproject.toml
uv add <package>                 # Add a dependency
uv remove <package>              # Remove a dependency

# Running
uv run python -m <module>        # Run a module
uv run uvicorn app:app --reload  # Dev server
uv run pytest                    # Run tests
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run pyright                   # Type check

# Frontend (if applicable)
bun install                      # Install JS deps
bun run build                    # Build into templates/
bun run dev                      # Dev with hot reload
```

Never run bare `python`, `pip`, or `pip install`. Always use `uv run` or `uv add`.

## Code style

### Python

- Use match-case over if/elif chains for pattern matching
- Use the walrus operator (:=) when it simplifies assignment + test
- Never nest more than 2 levels deep — use early returns and guard clauses
- Modern type hints only: `list[str]`, `dict[str, int]`, `str | None` — never `Optional`, `Union`, `List`, `Dict` from typing
- All functions must have type annotations on parameters and return values
- Use `pathlib.Path` for all file operations, never `os.path`
- Use f-strings for string formatting, never `.format()` or `%`
- Use comprehensions over `map()`/`filter()` when readable
- Use context managers (`with`) for resource management
- Pydantic v2 for all data validation — use `model_validate`, `field_validator`, not manual parsing
- Use `StrEnum` for string enums, `auto()` for values, UPPERCASE for members

### TypeScript (frontend)

- Vanilla TypeScript — no React, no Vue, no framework
- shadcn/ui components via Bun
- Build output goes to the Python project's `templates/` directory
- Jinja2 templates for server-rendered HTML
- Minimal JS — progressive enhancement over SPA patterns

### General

- Line length: 100 characters max
- One blank line between functions, two between top-level classes
- No inline type: ignore or noqa comments — fix the actual type/lint issue instead
- No dead code, no commented-out code — delete it, git has history

## Testing

- Write tests for all new functionality
- Prefer integration tests that hit real code paths over mocked unit tests
- Don't mock the database — use a test database or SQLite in-memory
- Test names must describe the scenario: `test_login_fails_with_expired_token`
- One assertion per test when practical
- Run the full suite before committing: `uv run pytest`
- Tests live in `tests/` mirroring the source structure

## Error handling

- Handle errors explicitly — never silently swallow exceptions
- Use structured error types, not bare strings
- Log errors with context (what was happening, what input caused it)
- At system boundaries (user input, API responses, file I/O): validate and handle
- Internal code: trust types and let exceptions propagate
- Never use bare `except:` or `except Exception:` without re-raising or logging

## Security

- Never hardcode secrets, API keys, or credentials in source code
- Use environment variables or `.env` files (gitignored)
- Validate and sanitize all external input
- Use parameterized queries — never string-interpolate SQL
- Set CORS, CSP, and security headers on all responses

## Git conventions

- Commit messages explain *why*, not *what* — the diff shows the what
- Imperative mood: "add auth middleware" not "added auth middleware"
- Keep commits atomic — one logical change per commit
- Never commit: `.env`, `node_modules/`, `__pycache__/`, `.vibe/index.db`, build artifacts
- Branch naming: `feature/short-description`, `fix/short-description`

## Project structure

```
├── src/                    # Python source
│   └── <package>/
│       ├── __init__.py
│       ├── app.py          # ASGI app / Starlette routes
│       ├── models.py       # Pydantic models
│       └── db.py           # Database layer
├── templates/              # Jinja2 templates (frontend builds here)
├── static/                 # Static assets
├── tests/                  # Test suite
├── pyproject.toml          # Python project config (single source of truth)
├── .env                    # Local env vars (gitignored)
└── AGENTS.md               # This file
```

## Dependencies

- Keep dependencies minimal — prefer the standard library
- Pin versions in pyproject.toml via uv.lock
- Audit new dependencies for maintenance status and security
- Prefer Astral/Rust-backed tooling (uv, ruff) over legacy Python tools
- For async: use anyio or raw asyncio, not threading

## What NOT to do

- Don't add features that weren't asked for
- Don't refactor code you're not working on
- Don't add comments to code you didn't write
- Don't add type annotations to code you didn't change
- Don't create abstractions for one-time operations
- Don't design for hypothetical future requirements
- Three similar lines of code are better than a premature helper function
