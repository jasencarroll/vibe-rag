# general.rule
# General coding rules for this project.
# Customize the `files` glob and guidelines to match your stack.

description: "Project-wide coding standards and conventions."
files: "**/*"

guidelines:
  - title: "Code Style"
    description: >
      Follow consistent formatting and naming conventions.
      Use your language's idiomatic style (e.g., camelCase for JS/TS,
      snake_case for Python, PascalCase for Go exported names).
      Keep line length under 100 characters where practical.

  - title: "Minimize Nesting"
    description: >
      Prefer early returns and guard clauses over deeply nested
      if/else chains. Flat code is easier to read and maintain.

  - title: "Strong Typing"
    description: >
      Use explicit type annotations wherever supported by the language.
      Avoid `any` in TypeScript, bare `dict` in Python, or `interface{}`
      in Go. Prefer precise types that catch errors at compile time.

  - title: "Error Handling"
    description: >
      Handle errors explicitly. Never silently swallow exceptions.
      Log or propagate errors with meaningful context. Prefer
      structured error types over raw strings.

  - title: "Testing"
    description: >
      Write tests for new functionality. Prefer integration tests
      that exercise real behavior over mocks. Keep tests focused
      and named descriptively.

  - title: "Security"
    description: >
      Never hardcode secrets, API keys, or credentials. Use
      environment variables or secret managers. Validate and sanitize
      all external input. Avoid command injection, XSS, and SQL
      injection vulnerabilities.

  - title: "Dependencies"
    description: >
      Keep dependencies minimal. Prefer standard library solutions
      when they exist. Pin dependency versions. Review new dependencies
      for maintenance status and security.

  - title: "Git Conventions"
    description: >
      Write concise commit messages that explain *why*, not *what*.
      Keep commits atomic — one logical change per commit. Never
      commit generated files, secrets, or large binaries.

  - title: "Documentation"
    description: >
      Document public APIs and non-obvious logic. Keep comments
      accurate — remove stale comments rather than leaving them.
      Prefer self-documenting code over excessive comments.

  - title: "Simplicity"
    description: >
      Favor the simplest solution that works. Avoid premature
      abstraction, over-engineering, and designing for hypothetical
      future requirements. Three similar lines are better than a
      premature helper function.
