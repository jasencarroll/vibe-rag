# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `load_session_context` now synthesizes a situational-awareness briefing with pulse, hazards, live decisions, and task context so session-start hooks can brief the client before the first prompt
- `vibe-rag doctor` now validates the native Vibe `[[hooks.SessionStart]]` path in generated projects

### Changed
- Generated Vibe scaffolds now use native `[[hooks.SessionStart]]` hook configuration instead of the older Vibe-only bootstrap path
- Generated client configs now pin the resolved `vibe-rag` binary path so installed-wheel scaffolds work reliably outside the source checkout
- Codex hook output now emits `suppressOutput: true` on successful session-start briefings so compatible clients can keep the context hidden from the user transcript

### Fixed
- `load_session_context` now degrades in the intended order by collecting pulse before database-backed retrieval work
- Workspace git status parsing no longer misclassifies a leading-space `git status --short` line as staged work
- Stale index file-count drift no longer counts `.mypy_cache` or `.ruff_cache`, and skips the expensive file walk when git HEAD already proves staleness
- Pre-v0.1.0 session narrative fallback now renders readable prose instead of dumping the raw prior summary
- `live_decisions` no longer emits a fake similarity score for recency-ranked memory items

## [0.0.23] - 2026-03-23

### Changed
- Git now ignores local `.codex/` dogfood config alongside local `.vibe/` client config

## [0.0.22] - 2026-03-23

### Fixed
- `release.yml` now explicitly dispatches `publish.yml` against the release tag instead of relying on GitHub's `release.published` event fan-out
- `publish.yml` now supports a manual `workflow_dispatch` backstop for republishing a specific tag when needed

## [0.0.21] - 2026-03-23

### Added
- `vibe-rag reindex` as a direct CLI path for refreshing the local `.vibe/index.db` without going through a client MCP prompt

### Changed
- CLI help text now describes `vibe-rag` as a client-agnostic MCP repo-memory/search tool instead of narrowing it to Mistral Vibe
- Embedding provider resolution now falls back to configured hosted providers when Ollama is unavailable and no explicit provider env was set
- Freeform `remember(...)` captures now infer stronger durable kinds like `constraint` when the content supports it
- Memory search now filters stale cross-project user memories and current-project auto-session noise more aggressively when durable local memory exists

### Fixed
- Normal CLI commands like `doctor` and `reindex` now suppress noisy hosted-provider `httpx` request logs in regular output
- Generated Vibe scaffolds now launch the MCP server through the user's shell startup files so persisted local env setup reaches `vibe-rag serve` without storing secrets in project config

## [0.0.20] - 2026-03-23

### Added
- Five-repo real-eval baseline covering `vibe-rag`, `mistral-vibe`, `fda-platform`, `agent-os`, and `codex`
- Eval artifact summaries now report per-repo timing, fallback-query usage, and noise counts
- `memory_quality_report` for inspecting memory mix, stale/superseded state, capture provenance, and cleanup pressure
- `cleanup_duplicate_auto_memories` for previewing or pruning repeated auto-captured session memories
- Scenario-based memory usefulness evals for decisions, constraints, todos, supersession, cross-project staleness, and low-signal non-interference
- GitHub Actions `CI` and `Release` workflows plus a release-prep script that automate version bumps, changelog promotion, build/test checks, commit/push, and GitHub release creation
- Cross-artifact trend reporting for retrieval and memory-quality drift over time
- Persistent-memory snapshot and trend reporting for the maintainer repo's real user-memory drift over time
- Compact release-evidence rendering that combines the latest retrieval and persistent-memory artifact trails

### Changed
- Eval summaries now backfill computed timing and fallback metrics even for older saved artifacts
- Eval artifacts now include per-repo memory-quality snapshots, including cleanup pressure and duplicate auto-memory groups
- Memory ranking and cleanup now treat low-signal auto session summaries as weaker than substantive structured memory
- Low-signal auto session summaries are now skipped earlier during session capture instead of only being down-ranked later
- Auto-captured session memory now applies a write-time durability and novelty gate so transient status chatter and non-novel restatements are rejected before they enter the memory corpus
- One-turn auto session captures now infer stronger memory kinds and surface merge/supersede suggestions when a nearby existing memory looks like the update target

### Fixed
- Voyage doc embeddings now split and retry on real provider token-cap errors seen in large repos like `codex`

## [0.0.19] - 2026-03-22

### Added
- Real retrieval eval harness for local repos, including query variants, progress output, and timestamped JSON result artifacts
- Cross-repo eval coverage for fixture repos plus validated real-repo baselines on `vibe-rag`, `mistral-vibe`, and `fda-platform`
- Indexing and embedding progress events across Ollama, Mistral, OpenAI, and Voyage providers

### Changed
- Retrieval reranking now uses bounded similarity scoring with stronger intent-aware path weighting for procedural, setup, API, pipeline, and MCP queries
- Doc retrieval now favors canonical operator docs such as `README.md`, `docs/API.md`, `docs/PIPELINE.md`, and `docs/MCP-TOOLS.md` over planning noise when the query intent matches
- Session bootstrap and release/process queries were tuned against real repo artifacts instead of micro-repo assumptions

### Fixed
- Voyage embeddings now batch by both item count and token budget instead of failing on large repo runs
- Ollama embeddings now batch large indexing requests instead of sending a single oversized request
- Real repo eval output no longer creates circular JSON references when recording query attempts

## [0.0.18] - 2026-03-22

### Fixed
- `vibe-rag init` now rewrites placeholders only inside generated scaffold files instead of walking the entire target tree
- Generated repos now initialize `.git` automatically when scaffolding into a directory that is not already a git repo
- Generated Codex config now suppresses the unstable-features warning and the scaffold flow was revalidated against fresh Codex startup

## [0.0.17] - 2026-03-22

### Added
- Packaged Codex, Claude Code, and Gemini CLI scaffolding in `vibe-rag init`
- `hook-session-start` CLI bridge for client session-start context injection

### Changed
- Generated project docs now describe Vibe as first-class with experimental Codex, Claude Code, and Gemini CLI support

## [0.0.12] - 2026-03-22

### Added
- `save_session_memory` to distill completed chat turns into durable memory
- `save_session_summary` to maintain a rolling session summary with supersession
- Cross-repo pgvector retrieval fallback when current-project matches are sparse

### Changed
- pgvector metadata reads are normalized so rollup updates work reliably
- `project_status` separates local sqlite memory counts from pgvector counts

## [0.0.11] - 2026-03-22

### Fixed
- Packaged `vibe-rag init` now includes `.vibe/config.toml` and skill templates in built wheels/sdists
- Installed-package scaffold generation was revalidated end to end after the `0.0.10` packaging regression

## [0.0.10] - 2026-03-22

### Added
- `load_session_context` MCP tool to bootstrap a task with related memories, code hits, and docs hits
- Structured memory support for kinds, summaries, metadata, source session/message tracking, and supersession
- Day-to-day user guide covering setup, workflow, and prompt patterns
- Generated scaffold guidance for `memory_load_session_context` and memory-first repo search

### Changed
- PostgreSQL and sqlite memory layers now preserve richer metadata and hide superseded memories by default
- Index refresh now clears stale file hashes when the local index contents are missing
- README and scaffold docs now describe the session bootstrap workflow explicitly

### Fixed
- Legacy pgvector tables are migrated forward by adding missing columns and indexes
- Session-context retrieval reuses the same validated search helpers as standalone memory/code/docs tools

## [0.0.9] - 2026-03-22

### Fixed
- `python -m vibe_rag.cli serve` now starts the MCP server correctly for source-run Vibe sessions
- Relative `index_project(paths=["."])` indexing no longer breaks on repo-root runs
- pgvector operations now run on a dedicated async loop instead of failing under Vibe's running event loop
- `forget()` accepts pgvector UUID memory IDs correctly during MCP round-trips

### Changed
- pgvector migrations now detect and replace incompatible legacy `memories` schemas
- PostgreSQL memories use UUID IDs and `text[]` tags to match the live database shape
- README configuration docs now tell users exactly where to put `MISTRAL_API_KEY` and `DATABASE_URL`

### Added
- Regression coverage for source-run CLI entrypoints and single-loop async execution
- Real Vibe E2E verification for index, code/docs search, remember, search_memory, and forget

## [0.0.7] - 2026-03-22

### Fixed
- Race condition in `forget()` function using proper async/await patterns
- Inconsistent API responses in `search_memory()` between pgvector and sqlite backends
- Security vulnerability: API keys no longer written to config files

### Changed
- File collection optimized using specific glob patterns (5-10x faster on large directories)
- Complete language mappings for all supported file extensions
- Magic numbers replaced with named constants in code chunking

### Added
- Comprehensive documentation for constants and functions
- `_ensure_project_id()` safety function for robust pgvector operations
- Enhanced error handling and logging throughout

### Security
- API keys now only passed via environment variables
- No sensitive credentials stored in configuration files

## [0.0.5] - 2026-03-22

### Added
- Hybrid storage: pgvector for cross-repo memories, sqlite-vec for local code search
- End-to-end testing coverage
- Comprehensive test suite (23 tests)

## [0.0.4b] - 2026-03-22

### Added
- pgvector support for cross-repo persistent memories
- Async PostgreSQL integration using asyncpg
- Memory tools: remember, search_memory, forget with pgvector backend

## [0.0.4a] - 2026-03-22

### Fixed
- Embedding dimension consistency using codestral-embed (1536 dimensions)
- Large chunk truncation with MAX_CHARS limit
- Batch size optimization for embedding API calls

## [0.0.3] - 2026-03-22

### Added
- Modular architecture with separate db, indexing, and tools modules
- 8 comprehensive tool tests
- Polished AGENTS.md template with best practices
- Tree-sitter integration for syntax-aware code chunking

### Changed
- Improved error handling and user feedback
- Better documentation and code organization

## [0.0.2] - 2026-03-21

### Added
- Initial MCP server implementation
- SQLite vector database integration
- Basic code search functionality
- Project initialization with vibe-rag init

## [0.0.1] - 2026-03-20

### Added
- Initial project structure
- Basic CLI interface
- Core embedding functionality
