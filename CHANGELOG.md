# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
