# AGENTS.md

## Purpose

This is the maintainer guide for the `vibe-rag` repo itself.

Use it when making changes in this repository, especially for packaging, releases, and publishing.

## Repo Rules

- Read the current `README.md`, `docs/setup-guide.md`, and `docs/user-guide.md` before changing onboarding or setup behavior.
- Keep the packaged install path working. `vibe-rag init` must work from an installed wheel, not only from source.
- If changing generated project behavior, update both:
  - `src/vibe_rag/templates/AGENTS.md`
  - the user-facing docs in `README.md` and `docs/`
- Do not assume PyPI propagation is immediate after a successful publish workflow.

## Documentation Rules

- `README.md` is the first-run and install path.
- `docs/setup-guide.md` is the full environment/bootstrap guide.
- `docs/user-guide.md` is the day-to-day operator guide.
- Avoid leaving internal planning artifacts in `docs/`.

## Release Workflow

Preferred path: trigger the `Release` GitHub Actions workflow from `main`. It now prepares the version bump, promotes `CHANGELOG.md`, runs tests, builds the wheel, commits `Release vX.Y.Z`, pushes `main`, and creates the GitHub release that triggers PyPI publishing.

If you need the manual fallback, follow this exact order.

### 1. Prepare versioned files

Use:

```bash
python scripts/prepare_release.py --version X.Y.Z --notes-out /tmp/release-notes.md
```

This updates:

- `pyproject.toml`
- `src/vibe_rag/__init__.py`
- `tests/test_cli.py`
- `README.md`
- `uv.lock`
- `CHANGELOG.md`

### 2. If packaging or scaffold behavior changed

Verify the built wheel, not just source execution.

Required checks:

```bash
uv run pytest tests/test_cli.py tests/test_tools.py
uv build
```

If `init` or templates changed, inspect wheel contents:

```bash
python3 - <<'PY'
import zipfile, pathlib
wheel = sorted(pathlib.Path('dist').glob('vibe_rag-*.whl'))[-1]
print(wheel)
zf = zipfile.ZipFile(wheel)
for name in sorted(n for n in zf.namelist() if 'template' in n or 'templates' in n):
    print(name)
PY
```

Then verify installed-wheel behavior in an isolated tool dir:

```bash
tmp_root=$(mktemp -d /tmp/vibe-rag-wheeltest.XXXXXX)
tmp_tools="$tmp_root/tools"
tmp_bin="$tmp_root/bin"
tmp_proj="$tmp_root/proj"
mkdir -p "$tmp_tools" "$tmp_bin" "$tmp_proj"

UV_TOOL_DIR="$tmp_tools" UV_TOOL_BIN_DIR="$tmp_bin" \
  uv tool install --python 3.12 --force dist/vibe_rag-<VERSION>-py3-none-any.whl

cd "$tmp_proj"
PATH="$tmp_bin:$PATH" vibe-rag init demo
find "$tmp_proj/demo" -maxdepth 4 -print | sort
```

`vibe-rag init` must produce:

- `AGENTS.md`
- `.vibe/config.toml`
- `.vibe/skills/semantic-repo-search/SKILL.md`

### 3. Commit the release

Use a simple release commit:

```bash
git add <changed files>
git commit -m "Release vX.Y.Z"
```

### 4. Push main first

```bash
git push origin main
```

### 5. Create the GitHub release

This repo publishes to PyPI from the GitHub `release.published` event.

Do not rely on pushing the tag alone. Create the release:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
gh release create vX.Y.Z --repo jasencarroll/vibe-rag --title "vX.Y.Z" --notes-file /tmp/release-notes.md
```

Important:

- `gh release create` will create the remote tag if it does not already exist.
- If `git push origin vX.Y.Z` later says the ref already exists, that is expected.

### 6. Watch the publish workflow

Check the release-triggered PyPI workflow:

```bash
gh run list --repo jasencarroll/vibe-rag --workflow publish.yml --limit 5
gh run watch <RUN_ID> --repo jasencarroll/vibe-rag --exit-status
```

The expected workflow file is:

- `.github/workflows/publish.yml`

### 7. Confirm release state

Verify:

```bash
gh release view vX.Y.Z --repo jasencarroll/vibe-rag --json tagName,targetCommitish,url,isDraft,isPrerelease,publishedAt
git ls-remote --tags origin | rg 'refs/tags/vX\.Y\.Z$'
```

### 8. Update local tool installation

After PyPI publishes, update the local tool:

```bash
uv tool install --upgrade --python 3.12 vibe-rag@X.Y.Z
vibe-rag --version
```

If PyPI resolution fails immediately after a successful publish workflow, wait and retry. This has happened on recent releases because of propagation lag.

## Local Tooling Notes

- `vibe-rag` currently needs Python 3.12 for reliable `uv tool install` because `tree-sitter-languages` does not ship cp313 wheels in this environment.
- When installing or upgrading the tool locally, prefer:

```bash
uv tool install --upgrade --python 3.12 vibe-rag@X.Y.Z
```

## Vibe Integration Notes

- The required Vibe fork for the background MCP hook is:
  - `https://github.com/jasencarroll/mistral-vibe`
- If docs reference the recommended Vibe install, keep that link current.
- When changing session bootstrap behavior, verify:
  - packaged `vibe`
  - packaged `vibe-rag`
  - trusted project-local `.vibe/config.toml`
  - `MISTRAL_API_KEY` in the MCP server `env`
  - local user memory at `~/.vibe/memory.db`

## E2E Verification Pattern

For a real packaged E2E smoke test:

1. `vibe-rag init demo`
2. add small `src/` and `docs/` files
3. add `MISTRAL_API_KEY` to `.vibe/config.toml`
4. add:

```toml
[background_mcp_hook]
enabled = true
tool_name = "memory_load_session_context"
task_arg = "task"
```

5. trust the repo in Vibe
6. run prompts for:
   - `index this project`
   - `search the code for ...`
   - `search docs for ...`
   - `remember ...`
   - `search memory for ...`
   - `forget memory <id>`

On macOS, remember that `/tmp/...` may resolve to `/private/tmp/...` for trust checks.
