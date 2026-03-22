# AGENTS.md

## What this is

<!-- REPLACE THIS: Describe your project in 2-3 sentences. What does it do? Who is it for? -->
A Python-first monolithic web application. The backend is the source of truth. Frontend is server-rendered HTML with progressive enhancement — not a SPA.

## Stack

- **Runtime:** Python 3.12+
- **Package manager:** uv (Astral) — never pip, never bare python
- **Framework:** Starlette (or FastAPI) on ASGI via uvicorn
- **Frontend:** Bun + vanilla TypeScript + shadcn/ui → builds into `templates/` as Jinja2
- **Database:** PostgreSQL with asyncpg driver
- **ORM:** SQLModel (SQLAlchemy + Pydantic in one) with async sessions
- **Migrations:** Alembic (async template)
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
uv run alembic upgrade head           # Run database migrations
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

# Database (Alembic)
uv run alembic upgrade head             # Run all pending migrations
uv run alembic revision --autogenerate -m "description"  # Generate migration from model changes
uv run alembic downgrade -1             # Roll back one migration
uv run alembic history                  # Show migration history
```

## Architecture

```
HTTP request
  → uvicorn (ASGI server)
    → Starlette middleware (auth, CORS, logging)
      → Route handler
        → SQLModel (validate input)
          → Service layer (business logic)
            → AsyncSession → SQLModel query → asyncpg → PostgreSQL
        → Jinja2 template (render HTML) or JSON response
  → HTTP response
```

### Key patterns

- **Route handlers** are thin — validate input, call service, return response
- **Business logic** lives in service modules, not in route handlers
- **Database access** via SQLModel async sessions, never in route handlers directly
- **SQLModel models** validate input at the boundary and define the schema — one model system, not two
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
│       ├── models/              # SQLModel models (validation + database tables)
│       │   ├── auth.py
│       │   └── users.py
│       ├── db/                  # Database layer
│       │   ├── engine.py        # Async engine + session factory
│       │   └── models.py        # SQLModel table definitions
│       └── config.py            # Settings from env vars via pydantic-settings
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
├── alembic/                     # Database migrations (auto-generated)
│   ├── env.py                   # Alembic config (imports SQLModel metadata)
│   └── versions/                # Migration files
├── alembic.ini                  # Alembic settings
├── pyproject.toml               # Single source of truth for Python config
├── .env                         # Local env vars (gitignored)
├── .env.example                 # Template for .env (committed)
└── AGENTS.md                    # This file
```

### File naming

- Python modules: `snake_case.py`
- One domain per file in `routes/`, `services/`, `models/`
- Tests mirror source: `src/app/services/auth.py` → `tests/test_auth.py`
- Alembic migrations: `alembic/versions/` (auto-generated, don't hand-edit)
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
- `StrEnum` with `auto()` and UPPERCASE members for all enums
- Validation happens via SQLModel (which is Pydantic v2) — don't import Pydantic separately
- No `# type: ignore`, no `# noqa` — fix the underlying issue
- No dead code, no commented-out code — git has history

### SQLModel / Database

- Define models with `SQLModel` — they're both Pydantic models and SQLAlchemy tables
- Separate read models from write models: `UserCreate(SQLModel)` vs `User(SQLModel, table=True)`
- Always use async sessions: `async with AsyncSession(engine) as session:`
- Never use raw SQL string interpolation — use SQLModel/SQLAlchemy query builders
- Connection string: `postgresql+asyncpg://user:pass@host/db`
- Alembic for all migrations: `alembic revision --autogenerate -m "description"`
- Run migrations: `alembic upgrade head`
- Never manually edit the migration version table

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
- Use SQLModel/SQLAlchemy query builders — never string-interpolate user input into queries
- Validate all input at the boundary with SQLModel before it touches business logic
- Hash passwords with bcrypt or argon2 — never store plaintext
- Set security headers: CORS, CSP, X-Frame-Options, Strict-Transport-Security
- Rate limit auth endpoints
- Session tokens: httponly, secure, samesite=strict cookies
- Never log sensitive data (passwords, tokens, PII)

## Database conventions

- SQLModel for models, SQLAlchemy async engine under the hood, asyncpg as the driver
- Create engine at app startup with `create_async_engine("postgresql+asyncpg://...")`
- Pool config: `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`
- Always use async sessions with context managers: `async with AsyncSession(engine) as session:`
- Models define the schema — Alembic auto-generates migrations from model changes
- `alembic revision --autogenerate -m "add users table"` → `alembic upgrade head`
- Foreign keys, not null constraints, indexes defined on the model, not in raw SQL
- Relationships via `Relationship()` on SQLModel classes
- Read models (no `table=True`) for API input/output, table models (`table=True`) for persistence
- Never call `session.commit()` in a service function — let the route handler own the transaction

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
- SQLModel `table=True` models are mutable ORM objects, not frozen Pydantic models — don't return them from APIs directly, convert to read models
- Alembic `--autogenerate` doesn't detect all changes (e.g., column renames) — review generated migrations before running
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
