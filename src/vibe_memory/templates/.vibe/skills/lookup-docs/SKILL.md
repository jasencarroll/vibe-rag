---
name: lookup-docs
description: Look up current library documentation using Context7
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - ask_user_question
  - context7_resolve-library-id
  - context7_query-docs
---

# Lookup Docs

Fetch up-to-date documentation and code examples for any library.

## Workflow

1. Identify the library the user needs docs for
2. Resolve the library ID with `context7_resolve-library-id`
3. Query specific topics with `context7_query-docs`
4. Present the relevant docs, examples, and API references
5. If the user needs more detail, query again with a narrower topic

## Guidelines

- Always resolve the library ID first — don't guess
- Query with specific topics (e.g., "authentication", "middleware") not vague ones
- Present code examples prominently — they're the most useful part
- Note the version the docs apply to
- If multiple libraries match, ask the user to disambiguate
