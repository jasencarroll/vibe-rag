---
name: eval
description: Run retrieval evaluation flows for vibe-rag and summarize the latest artifact or trend output.
license: MIT
user-invocable: true
---

# Eval

Use this skill when validating retrieval quality or comparing search changes.

## Workflow

1. Pick the appropriate manifest under `evals/`.
2. Run `uv run python scripts/run_retrieval_eval.py <manifest>`.
3. Use `--summary`, `--trends`, or persistent-memory modes when you need evidence without rerunning embeddings.
4. Report the key quality signals: hit quality, fallback usage, latency, and noise.
