---
name: refactor
description: Refactor code for clarity and maintainability while preserving behavior
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - write_file
  - edit_file
  - grep
  - glob
  - list_directory
  - run_command
  - ask_user_question
---

# Refactor

Refactor the specified code while preserving all existing behavior.

## Workflow

1. Read the target code and all its callers/dependents
2. Identify existing tests — if none exist, ask the user before proceeding
3. Propose changes and ask for confirmation before applying
4. Apply refactoring incrementally
5. Run tests after each change to confirm nothing broke
6. Summarize what changed and why

## Principles

- Never change behavior — refactoring is structure-only
- Prefer small, reversible steps over big-bang rewrites
- Extract only when duplication is real (3+ occurrences)
- Simplify control flow (flatten nesting, early returns)
- Improve naming for clarity
- Remove dead code confidently
