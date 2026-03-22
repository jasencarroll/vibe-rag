---
name: code-review
description: Perform thorough code reviews with actionable feedback
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - glob
  - list_directory
  - ask_user_question
---

# Code Review

Review the specified files or recent changes for:

1. **Bugs & logic errors** — off-by-one, null derefs, race conditions, unhandled edge cases
2. **Security issues** — injection, hardcoded secrets, improper auth checks
3. **Code quality** — naming, complexity, duplication, dead code
4. **Style consistency** — adherence to project conventions in AGENTS.md
5. **Test coverage gaps** — untested branches, missing edge case tests

## Workflow

1. If no files specified, run `grep` or `glob` to find recently modified files
2. Read each file thoroughly
3. For each issue found, report:
   - **File and line number**
   - **Severity** (critical / warning / nit)
   - **What's wrong** and **how to fix it**
4. End with a summary: total issues by severity, overall assessment

Keep feedback specific and actionable. Don't nitpick formatting if a linter handles it.
