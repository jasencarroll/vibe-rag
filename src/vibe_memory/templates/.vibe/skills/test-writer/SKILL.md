---
name: test-writer
description: Generate tests for specified code with edge case coverage
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - write_file
  - edit_file
  - grep
  - glob
  - list_directory
  - run_command
  - ask_user_question
---

# Test Writer

Generate comprehensive tests for the specified code.

## Workflow

1. Read the target code and understand its behavior
2. Identify the project's testing framework by scanning existing tests
3. Follow existing test patterns (file naming, structure, assertions)
4. Write tests covering:
   - **Happy path** — normal expected behavior
   - **Edge cases** — empty inputs, boundaries, large values
   - **Error cases** — invalid input, failures, exceptions
   - **Integration** — real dependencies where practical (prefer over mocks)
5. Run the tests to confirm they pass
6. Report coverage summary

## Guidelines

- Match the project's existing test style exactly
- Use descriptive test names that explain the scenario
- One assertion per test where practical
- Don't mock what you can test directly
