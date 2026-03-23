# vibe-rag Maintainer Guide

This is the public maintainer guide for the `vibe-rag` repo.

If you are doing packaging, release, scaffold, or docs work in this repository, start here.

The internal agent-oriented maintainer contract still lives in `AGENTS.md`.

If you are looking for the user-facing product docs, start with the [README](../README.md), the [Setup Guide](setup-guide.md), and the [User Guide](user-guide.md).

Current public posture should stay consistent across those docs:

- Claude Code is strong
- Codex is strong
- Vibe is bootstrapped
- Gemini CLI is untested

## Source Of Truth

- `README.md` is the first-run and install path.
- `docs/setup-guide.md` is the full environment and bootstrap guide.
- `docs/user-guide.md` is the operator guide.
- `AGENTS.md` is the internal maintainer and agent contract for this repo.

If you change onboarding or scaffold behavior, update the user-facing docs together with the code.

## Security Posture

Keep the public posture explicit and consistent:

- `vibe-rag` is a local stdio MCP server for single-user workflows, not a network service.
- Untrusted MCP clients are out of scope for strong authz; write tools are inherently unsafe if the client is outside the user trust domain.
- Repo-configured hook commands must not be executed by diagnostics like `vibe-rag doctor`.
- Project-scoped retrieval is the default for memory search, session bootstrap, cleanup, and status surfaces.
- Session-start context is untrusted retrieval output and should never be described as authoritative policy.
- OpenRouter is an explicit off-host data flow, not a transparent local fallback.

## Packaging Rules

- Keep the packaged install path working.
- `vibe-rag init` must work from an installed wheel, not only from source.
- Treat the installed-wheel path as the release bar: build the wheel, install it, scaffold a repo, and verify session-start plus retrieval from that installed binary.
- If generated project behavior changes, update both:
  - `src/vibe_rag/templates/AGENTS.md`
  - the user-facing docs in `README.md` and `docs/`
- Do not assume PyPI propagation is immediate after publish.

## Preferred Release Path

Use the `Release` GitHub Actions workflow from `main`.

That workflow prepares the version bump, promotes `CHANGELOG.md`, runs tests, builds the wheel, commits `Release vX.Y.Z`, pushes `main`, creates the GitHub release, and dispatches the PyPI publish workflow against the release tag.

## Manual Release Fallback

### 1. Prepare versioned files

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

### 2. Verify packaging

At minimum:

```bash
uv run pytest tests/test_cli.py tests/test_tools.py
uv build
```

If packaging or scaffold behavior changed, verify the built wheel and the installed-wheel path:

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

Then smoke-test the installed wheel in an isolated tool dir:

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

Expected generated files:

- `AGENTS.md`
- `.vibe/config.toml`
- `.vibe/skills/semantic-repo-search/SKILL.md`

### 3. Commit the release

```bash
git add <changed files>
git commit -m "Release vX.Y.Z"
```

### 4. Push `main`

```bash
git push origin main
```

### 5. Create the GitHub release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
gh release create vX.Y.Z --repo jasencarroll/vibe-rag --title "vX.Y.Z" --notes-file /tmp/release-notes.md
```

### 6. Watch publish

```bash
gh run list --repo jasencarroll/vibe-rag --workflow publish.yml --limit 5
gh run watch <RUN_ID> --repo jasencarroll/vibe-rag --exit-status
```

Manual backstop:

```bash
gh workflow run publish.yml --repo jasencarroll/vibe-rag -f ref=vX.Y.Z
```

### 7. Confirm release state

```bash
gh release view vX.Y.Z --repo jasencarroll/vibe-rag --json tagName,targetCommitish,url,isDraft,isPrerelease,publishedAt
git ls-remote --tags origin | rg 'refs/tags/vX\.Y\.Z$'
```

### 8. Update the local tool

```bash
uv tool install --upgrade --python 3.12 vibe-rag@X.Y.Z
vibe-rag --version
```

If PyPI resolution fails immediately after a successful publish workflow, wait and retry.

## Vibe And E2E Checks

When session bootstrap or scaffold behavior changes, verify:

- packaged `vibe`
- packaged `vibe-rag`
- trusted project-local `.vibe/config.toml`
- OpenRouter env behavior (`RAG_OR_*`) and DB env behavior
- local user memory at `~/.vibe/memory.db`

For a real packaged E2E smoke:

1. `vibe-rag init demo`
2. add small `src/` and `docs/` files
3. run `vibe-rag doctor`
4. run `vibe-rag reindex`
5. confirm the client sees session-start context and retrieval

On macOS, remember that `/tmp/...` may resolve to `/private/tmp/...` for trust checks.
