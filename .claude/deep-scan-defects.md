# OPERATION DEEP SCAN — Defect Log

*Started: 2026-03-23*

---

### [P1] CLAUDE.md import side-effect line number was stale (FIXED)
- **File:** CLAUDE.md:40
- **Found:** Phase 1, Sprint 1
- **Description:** Documented `server.py line 112` but actual import is on line 140
- **Resolution:** Updated to line 140

### [P1] CLAUDE.md DB env var names were stale (FIXED)
- **File:** CLAUDE.md:54-55
- **Found:** Phase 1, Sprint 1
- **Description:** Documented `VIBE_RAG_DB`/`VIBE_RAG_USER_DB` but code uses `RAG_DB`/`RAG_USER_DB`
- **Resolution:** Updated to match actual env var names

### [P2] server.py _cleanup() does not null _project_db/_user_db after closing
- **File:** server.py:_cleanup()
- **Found:** Phase 1, Sprint 1
- **Description:** _embedder is set to None after close, but _project_db and _user_db are not. Harmless (runs at exit) but inconsistent.
- **Impact:** Low — only matters if someone calls _get_db() after _cleanup()

