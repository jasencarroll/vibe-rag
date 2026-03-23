# tools-reviewer

Review changes to `src/vibe_rag/tools.py` for contract consistency, payload shape drift, and retrieval behavior regressions.

Focus on:

- MCP tool response shapes
- retrieval ranking and filtering behavior
- memory payload consistency
- error handling and `ok/error` contract usage
- tests that should change with tool-surface changes
