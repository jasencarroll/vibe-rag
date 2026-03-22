# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
