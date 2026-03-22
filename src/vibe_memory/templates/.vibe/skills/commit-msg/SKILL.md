---
name: commit-msg
description: Generate a commit message from staged changes
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - run_command
  - ask_user_question
---

# Commit Message

Generate a well-crafted commit message for the current staged changes.

## Workflow

1. Run `git diff --cached` to see staged changes
2. Run `git log --oneline -10` to match the repo's commit style
3. Analyze the changes: what was added, modified, removed, and *why*
4. Draft a commit message:
   - **Subject line**: imperative mood, under 72 chars, explains the *why*
   - **Body** (if needed): bullet points explaining non-obvious decisions
5. Present the message and ask the user to confirm or edit
6. On confirmation, run `git commit -m "<message>"`

## Rules

- Focus on *why*, not *what* — the diff shows the what
- One logical change per commit
- No generic messages like "fix bug" or "update code"
