---
name: gh-pr
description: Create and manage GitHub PRs using gh CLI
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - glob
  - run_command
  - ask_user_question
---

# GitHub PR

Create, manage, and review pull requests using the `gh` CLI.

## Workflow

### Create a PR
1. Run `git status` and `git diff` to understand current changes
2. Run `git log --oneline -10` to match commit style
3. Create a branch if not already on one: `git checkout -b <branch>`
4. Stage and commit changes
5. Push: `git push -u origin <branch>`
6. Create PR: `gh pr create --title "..." --body "..."`
7. Report the PR URL

### Review a PR
1. `gh pr view <number>` for overview
2. `gh pr diff <number>` for changes
3. `gh pr checks <number>` for CI status
4. Provide review feedback

### Check CI
1. `gh pr checks <number>` to see status
2. `gh run view <run-id>` for details on failures
3. `gh run view <run-id> --log-failed` for failure logs

## Guidelines

- PR titles under 72 characters, imperative mood
- PR body includes: Summary (bullet points), Test Plan
- Never force push unless explicitly asked
- Check CI status before marking ready for review
- Use `gh pr merge --squash` by default when merging
