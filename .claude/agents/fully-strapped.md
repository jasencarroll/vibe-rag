# Fully Strapped: Loaded Skills, MCPs, and Plugins

## Loaded Skills

| Skill | Description |
|-------|-------------|
| `using-superpowers` | Establishes how to find and use skills |
| `brainstorming` | Creative exploration before implementation |
| `writing-plans` | Multi-step task planning |
| `executing-plans` | Execute implementation plans with review checkpoints |
| `test-driven-development` | TDD workflow |
| `systematic-debugging` | Bug/failure investigation |
| `verification-before-completion` | Verify work before claiming done |
| `requesting-code-review` | Request review of completed work |
| `receiving-code-review` | Handle code review feedback |
| `dispatching-parallel-agents` | Run independent tasks in parallel |
| `subagent-driven-development` | Execute plans with independent subtasks |
| `using-git-worktrees` | Isolated feature work |
| `finishing-a-development-branch` | Merge/PR/cleanup decisions |
| `writing-skills` | Create/edit skills |
| `execute-plan` | *(deprecated -> executing-plans)* |
| `write-plan` | *(deprecated -> writing-plans)* |
| `brainstorm` | *(deprecated -> brainstorming)* |
| `update-config` | Configure settings.json, hooks, permissions |
| `keybindings-help` | Customize keyboard shortcuts |
| `simplify` | Review changed code for quality |
| `loop` | Run commands on recurring intervals |
| `schedule` | Cron-scheduled remote agents |
| `claude-api` | Build apps with Claude API/SDK |
| `release` | vibe-rag release workflow |
| `eval` | vibe-rag retrieval evaluation |
| `code-review` | Review a PR |
| `commit-push-pr` | Commit, push, open PR |
| `clean_gone` | Clean up deleted remote branches |
| `commit` | Create a git commit |
| `revise-claude-md` | Update CLAUDE.md with learnings |
| `claude-md-improver` | Audit/improve CLAUDE.md files |
| `feature-dev` | Guided feature development |
| `ralph-loop:help/cancel-ralph/ralph-loop` | Ralph Loop plugin |
| `review-pr` | Comprehensive PR review (pr-review-toolkit) |
| `new-sdk-app` | Create new Agent SDK app |
| `hookify` | Create hooks from conversation analysis |
| `frontend-design` | Production-grade frontend interfaces |
| `figma:implement-design` | Figma -> code |
| `figma:code-connect-components` | Figma Code Connect |
| `figma:create-design-system-rules` | Design system rules |
| `claude-automation-recommender` | Recommend Claude Code automations |
| `skill-creator` | Create/modify/eval skills |

## MCP Servers

| Server | Tools |
|--------|-------|
| **vibe-rag** | `search`, `remember`, `forget`, `update_memory`, `search_memory`, `save_session_memory`, `save_session_summary`, `load_session_context`, `project_status`, `index_project`, `ingest_daily_note`, `ingest_pr_outcome`, `cleanup_duplicate_auto_memories`, `supersede_memory`, `summarize_thread` |
| **ruflo** | ~150+ tools (agents, browser, claims, config, coordination, DAA, embeddings, github, hive-mind, hooks, memory, neural, performance, progress, sessions, swarm, system, tasks, terminal, transfer, wasm, workflows) |
| **context7** | `resolve-library-id`, `query-docs` |
| **playwright** | Browser automation (navigate, click, fill, screenshot, snapshot, etc.) |
| **claude.ai Apollo.io** | Contacts, accounts, enrichment, campaigns |
| **claude.ai Figma** | Design context, screenshots, Code Connect, diagrams |
| **claude.ai Gmail** | Read/search messages, drafts |
| **claude.ai Google Calendar** | Events, free time, calendars |
| **claude.ai Indeed** | Job search, company data |
| **claude.ai Dice** | Job search |
| **claude.ai Linear** | Issues, projects, documents, teams |
| **claude.ai Notion** | Pages, databases, search, comments |
| **claude.ai Stripe** | Payments, subscriptions, products |

## Subagent Types

| Agent | Purpose |
|-------|---------|
| `general-purpose` | Complex multi-step tasks |
| `Explore` | Fast codebase exploration |
| `Plan` | Architecture/implementation planning |
| `claude-code-guide` | Claude Code/API questions |
| `superpowers:code-reviewer` | Review against plan & standards |
| `code-simplifier` | Simplify/refine code |
| `feature-dev:code-reviewer/explorer/architect` | Feature development agents |
| `pr-review-toolkit:code-reviewer/silent-failure-hunter/code-simplifier/comment-analyzer/pr-test-analyzer/type-design-analyzer` | PR review agents |
| `agent-sdk-dev:agent-sdk-verifier-ts/py` | Agent SDK verification |
| `hookify:conversation-analyzer` | Analyze conversations for hook creation |
