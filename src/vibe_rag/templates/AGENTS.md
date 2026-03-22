# AGENTS.md

## What this is

<!-- Replace this section with your project's purpose -->
A Python-first monolithic web application. The backend is the source of truth. Frontend is server-rendered HTML with progressive enhancement — not a SPA.

## Stack

- **Runtime:** Python 3.12+
- **Package manager:** uv (Astral) — never pip, never bare python
- **Framework:** Starlette (or FastAPI) on ASGI via uvicorn
- **Frontend:** Bun + vanilla TypeScript + shadcn/ui → builds into `templates/` as Jinja2
- **Database:** PostgreSQL with asyncpg (SQLite for local/embedded)
- **ORM:** None — raw SQL with asyncpg, Pydantic models for validation
- **Testing:** pytest + pytest-asyncio
- **Linting:** ruff (replaces flake8, isort, black, pyflakes)
- **Types:** pyright strict mode
- **CI:** GitHub Actions

This is a monolithic architecture. The Python server owns routing, auth, data, and serves the HTML. Bun compiles TypeScript and shadcn components into the `templates/` folder where Jinja2 picks them up. There is no API/frontend split. There is no SPA. The browser gets HTML.

## Environment setup

```bash
# First time
git clone <repo>
cd <project>
uv sync                              # Install Python deps
cp .env.example .env                  # Set up local env vars
bun install                           # Install JS deps (if frontend exists)
bun run build                         # Build frontend into templates/

# Verify everything works
uv run pytest                         # Tests pass
uv run ruff check .                   # No lint errors
uv run pyright                        # No type errors
```

## Commands

```bash
# Python (always through uv, never bare python/pip)
uv sync                               # Install deps from pyproject.toml + uv.lock
uv add <package>                      # Add dependency
uv remove <package>                   # Remove dependency
uv run uvicorn app.main:app --reload  # Dev server
uv run pytest                         # Run all tests
uv run pytest tests/test_auth.py -v   # Run specific test file
uv run pytest -k "test_login" -v      # Run tests matching pattern
uv run ruff check .                   # Lint
uv run ruff format .                  # Format
uv run pyright                        # Type check

# Frontend (if applicable)
bun install                            # Install JS deps
bun run build                          # Build into templates/
bun run dev                            # Watch mode with hot reload

# Database
uv run python -m app.db.migrate        # Run migrations
uv run python -m app.db.seed           # Seed dev data
```

## Architecture

```
HTTP request
  → uvicorn (ASGI server)
    → Starlette middleware (auth, CORS, logging)
      → Route handler
        → Pydantic model (validate input)
          → Database query (asyncpg, raw SQL)
          → Business logic
        → Jinja2 template (render HTML) or JSON response
  → HTTP response
```

### Key patterns

- **Route handlers** are thin — validate input, call business logic, return response
- **Business logic** lives in service modules, not in route handlers
- **Database access** is always async, always parameterized, never in route handlers directly
- **Pydantic models** validate all input at the boundary — inside the app, trust the types
- **Templates** are server-rendered — the browser gets HTML, not JSON + a JS framework

## Project structure

```
├── src/
│   └── <package>/
│       ├── __init__.py
│       ├── main.py              # ASGI app, middleware stack, route mounting
│       ├── routes/              # Route handlers, grouped by domain
│       │   ├── auth.py
│       │   ├── users.py
│       │   └── health.py
│       ├── services/            # Business logic, one module per domain
│       │   ├── auth.py
│       │   └── users.py
│       ├── models/              # Pydantic models for validation + serialization
│       │   ├── auth.py
│       │   └── users.py
│       ├── db/                  # Database layer
│       │   ├── pool.py          # Connection pool setup
│       │   ├── queries/         # Raw SQL files or query functions
│       │   └── migrate.py       # Migration runner
│       └── config.py            # Settings from env vars via Pydantic
├── templates/                   # Jinja2 templates (frontend builds here)
│   ├── base.html                # Base layout
│   ├── components/              # Reusable template fragments
│   └── pages/                   # Full page templates
├── static/                      # CSS, JS bundles, images
├── tests/                       # Mirrors src/ structure
│   ├── conftest.py              # Fixtures: test client, test DB, factories
│   ├── test_auth.py
│   └── test_users.py
├── frontend/                    # TypeScript source (if applicable)
│   ├── package.json
│   └── src/
├── pyproject.toml               # Single source of truth for Python config
├── .env                         # Local env vars (gitignored)
├── .env.example                 # Template for .env (committed)
└── AGENTS.md                    # This file
```

### File naming

- Python modules: `snake_case.py`
- One domain per file in `routes/`, `services/`, `models/`
- Tests mirror source: `src/app/services/auth.py` → `tests/test_auth.py`
- SQL migrations: `001_create_users.sql`, `002_add_sessions.sql`
- Templates: `snake_case.html`, partials prefixed with `_` (e.g., `_navbar.html`)

## Code style

### Python

- Python 3.12+ features only — match-case, walrus operator, modern type hints
- Type hints on everything: `def get_user(user_id: int) -> User | None:`
- Modern types only: `list[str]`, `dict[str, int]`, `str | None`
- Never import from `typing`: no `Optional`, `Union`, `List`, `Dict`, `Tuple`
- `pathlib.Path` for all file ops, never `os.path`
- f-strings only, never `.format()` or `%`
- Comprehensions over `map()`/`filter()` when readable
- Context managers (`with`) for all resource management
- Early returns and guard clauses — never nest more than 2 levels
- Pydantic v2 for all validation: `model_validate`, `field_validator`
- `StrEnum` with `auto()` and UPPERCASE members for all enums
- No `# type: ignore`, no `# noqa` — fix the underlying issue
- No dead code, no commented-out code — git has history

### SQL

- All queries use parameterized placeholders (`$1`, `$2`) — never string interpolation
- Use CTEs over subqueries for readability
- Explicit column lists in SELECT — never `SELECT *` in production code
- Migrations are forward-only SQL files, numbered sequentially

### TypeScript (frontend)

- Vanilla TypeScript — no React, no Vue, no framework
- shadcn/ui for components, Bun for bundling
- Build output → `templates/` and `static/`
- Progressive enhancement: pages work without JS, JS adds interactivity
- Minimal client-side state — server is the source of truth

### Formatting

- Line length: 100 characters
- One blank line between functions, two between top-level classes
- Imports: stdlib → third-party → local (ruff handles this)
- Trailing commas on multi-line collections

## Testing

- Every new feature or bugfix gets a test
- Integration tests over unit tests — hit the real code path
- Don't mock the database — use a test database or SQLite in-memory
- Don't mock HTTP clients — use `httpx_mock` or a test server
- Test names describe the scenario: `test_login_fails_with_expired_token`
- One logical assertion per test
- Tests must pass before committing: `uv run pytest`
- Test files mirror source structure in `tests/`
- Use `conftest.py` for shared fixtures: test client, test DB, factories
- Factories over fixtures for test data — explicit beats implicit

### Test structure

```python
def test_create_user_returns_201(client):
    # Arrange
    payload = {"email": "test@example.com", "name": "Test User"}

    # Act
    response = client.post("/users", json=payload)

    # Assert
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

## Error handling

- Handle errors explicitly — never silently swallow exceptions
- Structured error responses: `{"error": "not_found", "message": "User 123 not found"}`
- Log errors with context: what operation, what input, what went wrong
- At boundaries (user input, API calls, file I/O): validate and catch
- Internal code: trust types, let exceptions propagate up
- Never bare `except:` or `except Exception:` without re-raising or logging
- HTTP errors: use Starlette's `HTTPException` with appropriate status codes
- Database errors: catch `asyncpg` specific exceptions, not generic ones

## Security

- No secrets in source code — environment variables or `.env` (gitignored)
- `.env.example` committed with placeholder values for documentation
- Parameterized SQL only — never string-interpolate user input into queries
- Validate all input at the boundary with Pydantic before it touches business logic
- Hash passwords with bcrypt or argon2 — never store plaintext
- Set security headers: CORS, CSP, X-Frame-Options, Strict-Transport-Security
- Rate limit auth endpoints
- Session tokens: httponly, secure, samesite=strict cookies
- Never log sensitive data (passwords, tokens, PII)

## Database conventions

- Raw SQL with asyncpg — no ORM
- Connection pool created at app startup, closed at shutdown
- All queries are async: `await conn.fetch(...)`, `await conn.fetchrow(...)`
- Parameterized queries with `$1`, `$2` positional params
- Migrations are numbered SQL files run in order: `001_`, `002_`, etc.
- Each migration is idempotent where possible (`CREATE TABLE IF NOT EXISTS`)
- Foreign keys, not null constraints, and indexes defined in migrations
- Use `RETURNING` clause to get inserted/updated rows without a second query

## API conventions (if applicable)

- RESTful routes: `GET /users`, `POST /users`, `GET /users/{id}`, `PUT /users/{id}`, `DELETE /users/{id}`
- Consistent response shape: `{"data": ...}` for success, `{"error": ..., "message": ...}` for errors
- 201 for created, 204 for deleted, 400 for bad input, 401 for unauthorized, 404 for not found
- Pagination: `?page=1&per_page=20`, return `{"data": [...], "total": N, "page": 1}`
- Dates in ISO 8601 UTC: `2026-03-22T17:00:00Z`
- IDs: UUIDs for external-facing, integers for internal

## Git conventions

- Commit messages explain *why*, not *what*
- Imperative mood: "add auth middleware" not "added auth middleware"
- Atomic commits — one logical change per commit
- Never commit: `.env`, `node_modules/`, `__pycache__/`, `.vibe/index.db`, `dist/`, build artifacts
- Branch naming: `feature/short-description`, `fix/short-description`
- PR titles: imperative, under 72 chars
- Squash merge to main

## Dependencies

- Minimal — prefer stdlib over third-party
- Astral tooling first: uv (packaging), ruff (linting/formatting)
- Rust/Zig-backed Python tools when available (orjson, uvloop, httptools)
- Pin all versions via uv.lock
- Audit new deps: maintenance status, security history, transitive dependency count
- Async-first: anyio or raw asyncio, never threading for I/O
- No legacy tools: no setuptools, no tox, no flake8, no black, no isort, no mypy

## Deployment

- Docker: multi-stage build, `uv` in builder stage, minimal runtime image
- Single `Dockerfile`, single process per container
- Health check endpoint at `GET /health`
- Config via environment variables only — no config files in the image
- Logs to stdout in structured JSON
- Graceful shutdown: handle SIGTERM, drain connections

## Known gotchas

<!-- Add project-specific gotchas here as you discover them -->
- asyncpg returns `Record` objects, not dicts — use `dict(row)` to convert
- Starlette's `Request.json()` is a coroutine — must `await` it
- Pydantic v2 `model_validate` replaces v1 `parse_obj` — don't use the old API
- `uv run` creates the venv automatically — you never need `uv venv` manually
- ruff's import sorting replaces isort — don't install isort separately
- Bun's build output path must match Jinja2's `templates/` directory exactly

## What NOT to do

- Don't add features that weren't asked for
- Don't refactor code you're not working on
- Don't add comments to code you didn't write or change
- Don't add type annotations to code you didn't change
- Don't create abstractions for one-time operations
- Don't design for hypothetical future requirements
- Don't wrap simple operations in helper functions
- Don't add error handling for cases that can't happen
- Don't add backwards-compatibility shims — just change the code
- Don't suggest installing mypy, black, isort, flake8, or pip — we use ruff, pyright, and uv
- Three similar lines of code are better than a premature abstraction
