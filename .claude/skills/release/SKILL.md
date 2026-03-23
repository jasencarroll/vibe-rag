---
name: release
description: Run the vibe-rag maintainer release workflow with the required file updates, checks, and publish verification.
license: MIT
user-invocable: true
---

# Release

Use this skill when preparing or reviewing a `vibe-rag` release.

## Workflow

1. Read `AGENTS.md` and follow the release workflow exactly.
2. Run `python scripts/prepare_release.py --version X.Y.Z --notes-out /tmp/release-notes.md`.
3. Verify the touched release files and changelog promotion.
4. Run the required tests and build checks.
5. If packaging or scaffolding changed, verify the built wheel contents and `vibe-rag init` from the installed wheel.
6. Commit with `Release vX.Y.Z`, push `main`, create the GitHub release, and verify the publish workflow.
