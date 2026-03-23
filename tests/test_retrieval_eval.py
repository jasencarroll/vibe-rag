from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import vibe_rag.server as srv
from vibe_rag.tools import index_project, load_session_context, remember_structured


def _load_eval_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KeywordEmbedder:
    """Deterministic bag-of-keywords embedder for retrieval eval fixtures."""

    KEYWORDS = (
        "billing",
        "invoice",
        "customer",
        "release",
        "publish",
        "tag",
        "deploy",
        "gateway",
        "auth",
        "token",
        "summary",
        "constraint",
    )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        counts = [float(lowered.count(keyword)) for keyword in self.KEYWORDS]
        total = sum(counts) or 1.0
        normalized = [count / total for count in counts]
        padded = normalized + [0.0] * (2560 - len(normalized))
        return padded

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_code_query_sync(self, texts: list[str]) -> list[list[float]]:
        return self.embed_code_sync(texts)

    def close(self) -> None:
        return None


RETRIEVAL_EVAL_FIXTURES = [
    {
        "name": "billing_repo",
        "files": {
            "src/billing.py": "def create_invoice(customer_id):\n    return f'invoice:{customer_id}'\n",
            "src/auth.py": "def validate_token(token):\n    return token.startswith('tok_')\n",
            "docs/billing.md": "Billing flow creates an invoice for each customer order.\n",
            "docs/auth.md": "Authentication validates bearer token headers.\n",
        },
        "memories": [
            {
                "summary": "billing constraint",
                "details": "Invoices must include the customer id before checkout completes.",
                "memory_kind": "constraint",
            }
        ],
        "task": "continue the billing invoice flow for a customer",
        "expected_code": {"src/billing.py"},
        "expected_docs": {"docs/billing.md"},
        "unexpected_code": {"src/auth.py"},
        "unexpected_docs": {"docs/auth.md"},
    },
    {
        "name": "release_repo",
        "files": {
            "src/release.py": "def publish_release(tag):\n    return f'publish:{tag}'\n",
            "src/billing.py": "def create_invoice(customer_id):\n    return customer_id\n",
            "docs/release.md": "Release flow creates the git tag and publishes the package.\n",
            "docs/notes.md": "General project notes about billing and auth.\n",
        },
        "memories": [
            {
                "summary": "release decision",
                "details": "Always create the annotated tag before publishing the package.",
                "memory_kind": "decision",
            }
        ],
        "task": "prepare the release publish tag flow",
        "expected_code": {"src/release.py"},
        "expected_docs": {"docs/release.md"},
        "unexpected_code": {"src/billing.py"},
        "unexpected_docs": {"docs/notes.md"},
    },
]


def _write_fixture_repo(root: Path, fixture: dict) -> None:
    for relative_path, content in fixture["files"].items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _count_irrelevant(results: list[dict], unexpected_paths: set[str], key: str = "file_path") -> int:
    return sum(1 for item in results if item.get(key) in unexpected_paths)


def test_retrieval_eval_fixtures(tmp_db, tmp_path: Path):
    old_cwd = os.getcwd()
    old_embedder = srv._embedder
    old_project_id = srv._project_id
    srv._embedder = KeywordEmbedder()

    try:
        for fixture in RETRIEVAL_EVAL_FIXTURES:
            tmp_db.clear_code()
            tmp_db.clear_docs()
            conn = tmp_db._get_conn()
            conn.executescript(
                "DELETE FROM memories_vec; DELETE FROM memories; DELETE FROM file_hashes;"
            )
            conn.commit()

            repo_root = tmp_path / fixture["name"]
            repo_root.mkdir()
            _write_fixture_repo(repo_root, fixture)

            os.chdir(repo_root)
            srv._project_id = None

            index_result = index_project()
            assert index_result["ok"] is True
            assert "Indexed" in index_result["summary"]

            for memory in fixture["memories"]:
                remember_result = remember_structured(
                    summary=memory["summary"],
                    details=memory["details"],
                    memory_kind=memory["memory_kind"],
                )
                assert remember_result["ok"] is True

            payload = load_session_context(
                fixture["task"],
                memory_limit=3,
                code_limit=3,
                docs_limit=3,
            )

            assert payload["ok"] is True
            assert payload["project_id"] == srv._ensure_project_id()
            assert any(item["file_path"] in fixture["expected_code"] for item in payload["code"])
            assert any(item["file_path"] in fixture["expected_docs"] for item in payload["docs"])
            assert _count_irrelevant(payload["code"], fixture["unexpected_code"]) <= 1
            assert _count_irrelevant(payload["docs"], fixture["unexpected_docs"]) <= 1
            assert payload["memories"]
            assert payload["memories"][0]["memory_kind"] in {"decision", "constraint", "summary", "todo"}
    finally:
        os.chdir(old_cwd)
        srv._embedder = old_embedder
        srv._project_id = old_project_id


def test_eval_task_tries_query_variants_until_one_passes():
    eval_runner = _load_eval_runner()
    task = {
        "query": "too broad",
        "query_variants": ["release workflow automation", "release procedure for maintainers"],
        "expected_code": ["src/release.py"],
        "expected_docs": ["AGENTS.md"],
    }

    payloads = {
        "too broad": {
            "ok": True,
            "project_id": "demo-project",
            "code": [{"file_path": "src/other.py"}],
            "docs": [{"file_path": "README.md"}],
            "stale": {"is_stale": False},
        },
        "release workflow automation": {
            "ok": True,
            "project_id": "demo-project",
            "code": [{"file_path": "src/release.py"}],
            "docs": [{"file_path": "README.md"}],
            "stale": {"is_stale": False},
        },
        "release procedure for maintainers": {
            "ok": True,
            "project_id": "demo-project",
            "code": [{"file_path": "src/release.py"}],
            "docs": [{"file_path": "AGENTS.md"}],
            "stale": {"is_stale": False},
        },
    }

    original = eval_runner.load_session_context
    eval_runner.load_session_context = lambda query, **_: payloads[query]
    try:
        result = eval_runner._eval_task(task)
    finally:
        eval_runner.load_session_context = original

    assert result["ok"] is True
    assert result["query"] == "release procedure for maintainers"
    assert result["attempt_count"] == 3
    assert len(result["attempts"]) == 3


def test_write_artifact_persists_eval_report(tmp_path: Path, monkeypatch):
    eval_runner = _load_eval_runner()
    manifest_path = tmp_path / "demo.toml"
    manifest_path.write_text("[[repo]]\nname='demo'\npath='~/dev/demo'\n")
    monkeypatch.setenv("RAG_OR_API_KEY", "test-key")
    monkeypatch.setenv("RAG_OR_EMBED_MOD", "perplexity/pplx-embed-v1-4b")

    output_path = eval_runner._write_artifact(
        {"ok": True, "repos": [{"name": "demo", "ok": True}]},
        manifest_path,
        tmp_path / "artifacts",
    )

    payload = __import__("json").loads(output_path.read_text())
    assert output_path.parent == tmp_path / "artifacts"
    assert payload["ok"] is True
    assert payload["manifest"] == str(manifest_path.resolve())
    assert payload["embedding_provider"] == "openrouter"
    assert payload["summary"]["repo_total"] == 1
    assert payload["summary"]["passed_repo_total"] == 1
    assert payload["summary"]["passed_task_total"] == 0
    assert payload["previous_artifact"] is None


def test_write_artifact_compares_with_previous_run(tmp_path: Path):
    eval_runner = _load_eval_runner()
    manifest_path = tmp_path / "demo.toml"
    manifest_path.write_text("[[repo]]\nname='demo'\npath='~/dev/demo'\n")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    previous_payload = {
        "ok": False,
        "repos": [
            {
                "name": "demo",
                "path": "/tmp/demo",
                "ok": False,
                "tasks": [
                    {"query": "task a", "ok": False},
                    {"query": "task b", "ok": True},
                ],
            }
        ],
    }
    previous_path = artifact_dir / "demo.20260323T010000Z.json"
    previous_path.write_text(__import__("json").dumps(previous_payload) + "\n")

    output_path = eval_runner._write_artifact(
        {
            "ok": True,
            "repos": [
                {
                    "name": "demo",
                    "path": "/tmp/demo",
                    "ok": True,
                    "tasks": [
                        {"query": "task a", "ok": True},
                        {"query": "task b", "ok": True},
                        {"query": "task c", "ok": False},
                    ],
                }
            ],
        },
        manifest_path,
        artifact_dir,
    )

    payload = __import__("json").loads(output_path.read_text())
    assert payload["previous_artifact"] == str(previous_path.resolve())
    assert payload["summary"]["task_total"] == 3
    assert payload["comparison"]["counts"]["newly_passing"] == 1
    assert payload["comparison"]["counts"]["unchanged_passing"] == 1
    assert payload["comparison"]["counts"]["new_tasks"] == 1
    assert payload["comparison"]["newly_passing"] == [{"repo": "demo", "query": "task a"}]
    assert payload["comparison"]["new_tasks"] == [{"repo": "demo", "query": "task c"}]


def test_artifact_comparison_uses_primary_task_query_for_variant_passes():
    eval_runner = _load_eval_runner()

    previous = {
        "ok": False,
        "repos": [
            {
                "name": "demo",
                "tasks": [
                    {"task_query": "release task", "query": "release task", "ok": False},
                ],
            }
        ],
    }
    current = {
        "ok": True,
        "repos": [
            {
                "name": "demo",
                "tasks": [
                    {
                        "task_query": "release task",
                        "query": "release procedure for maintainers",
                        "ok": True,
                    },
                ],
            }
        ],
    }

    comparison = eval_runner._compare_results(previous, current)
    assert comparison["counts"]["newly_passing"] == 1
    assert comparison["newly_passing"] == [{"repo": "demo", "query": "release task"}]
    assert comparison["counts"]["new_tasks"] == 0
    assert comparison["counts"]["removed_tasks"] == 0


def test_format_artifact_summary_includes_repo_and_change_counts(tmp_path: Path):
    eval_runner = _load_eval_runner()
    artifact_path = tmp_path / "demo.20260323T000000Z.json"
    artifact = {
        "generated_at": "2026-03-23T03:00:00Z",
        "manifest": str((tmp_path / "demo.toml").resolve()),
        "embedding_provider": "openrouter",
        "summary": {
            "repo_total": 1,
            "passed_repo_total": 1,
            "task_total": 2,
            "passed_task_total": 1,
            "fallback_task_total": 1,
            "irrelevant_code_total": 2,
            "irrelevant_docs_total": 1,
            "memory_cleanup_candidate_total": 3,
            "duplicate_auto_memory_group_total": 1,
            "total_index_seconds": 12.5,
            "repos": [
                {
                    "name": "demo",
                    "task_total": 2,
                    "passed_task_total": 1,
                    "index_seconds": 12.5,
                    "fallback_task_total": 1,
                    "memory_cleanup_candidate_total": 3,
                    "duplicate_auto_memory_group_total": 1,
                    "failed_queries": ["mcp route task"],
                    "fallback_queries": ["release task"],
                }
            ],
        },
        "comparison": {
            "counts": {
                "newly_passing": 1,
                "newly_failing": 0,
                "new_tasks": 0,
                "removed_tasks": 0,
            }
        },
    }

    text = eval_runner._format_artifact_summary(artifact, artifact_path)
    assert "Artifact:" in text
    assert "Repos: 1/1 passed | Tasks: 1/2 passed" in text
    assert "Timing: 12.5s total index | Fallbacks: 1 | Noise: code 2 docs 1" in text
    assert "Memory: cleanup 3 | duplicate auto groups 1" in text
    assert "Changes: +1 passing -0 failing | new 0 | removed 0" in text
    assert "- demo: 1/2 tasks passed | 12.5s | fallbacks 1 | memory cleanup 3 | dup auto 1" in text
    assert "failed: mcp route task" in text
    assert "fallback: release task" in text


def test_main_summary_reads_latest_artifact_for_manifest(tmp_path: Path, monkeypatch, capsys):
    eval_runner = _load_eval_runner()
    manifest_path = tmp_path / "demo.toml"
    manifest_path.write_text("[[repo]]\nname='demo'\npath='~/dev/demo'\n")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "demo.20260323T000000Z.json"
    artifact_path.write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-03-23T03:00:00Z",
                "manifest": str(manifest_path.resolve()),
                "embedding_provider": "openrouter",
                "summary": {
                    "repo_total": 1,
                    "passed_repo_total": 1,
                    "task_total": 1,
                    "passed_task_total": 1,
                    "fallback_task_total": 0,
                    "irrelevant_code_total": 0,
                    "irrelevant_docs_total": 0,
                    "total_index_seconds": 1.2,
                    "repos": [
                        {
                            "name": "demo",
                            "task_total": 1,
                            "passed_task_total": 1,
                            "failed_queries": [],
                            "fallback_queries": [],
                            "index_seconds": 1.2,
                            "fallback_task_total": 0,
                        }
                    ],
                },
            }
        )
        + "\n"
    )

    monkeypatch.setattr(
        "sys.argv",
        ["run_retrieval_eval.py", str(manifest_path), "--artifact-dir", str(artifact_dir), "--summary"],
    )

    exit_code = eval_runner.main()
    output = capsys.readouterr().out
    assert exit_code == 0
    assert str(artifact_path) in output
    assert "demo: 1/1 tasks passed" in output


def test_format_trend_summary_backfills_old_and_new_artifacts(tmp_path: Path):
    eval_runner = _load_eval_runner()
    old_path = tmp_path / "demo.20260323T000000Z.json"
    new_path = tmp_path / "demo.20260323T010000Z.json"
    old_artifact = {
        "generated_at": "2026-03-23T00:00:00Z",
        "embedding_provider": "openrouter",
        "repos": [
            {
                "name": "demo",
                "ok": False,
                "index_result": "Indexed 3 code files in 12.5s",
                "tasks": [
                    {"query": "task a", "ok": False, "attempt_count": 2, "irrelevant_code": ["noise.py"], "irrelevant_docs": []},
                    {"query": "task b", "ok": True, "attempt_count": 1, "irrelevant_code": [], "irrelevant_docs": ["noise.md"]},
                ],
            }
        ],
    }
    new_artifact = {
        "generated_at": "2026-03-23T01:00:00Z",
        "embedding_provider": "openrouter",
        "summary": {
            "repo_total": 1,
            "passed_repo_total": 1,
            "task_total": 2,
            "passed_task_total": 2,
            "fallback_task_total": 0,
            "irrelevant_code_total": 0,
            "irrelevant_docs_total": 0,
            "memory_cleanup_candidate_total": 3,
            "duplicate_auto_memory_group_total": 1,
            "total_index_seconds": 8.1,
            "repos": [
                {
                    "name": "demo",
                    "task_total": 2,
                    "passed_task_total": 2,
                    "index_seconds": 8.1,
                    "fallback_task_total": 0,
                    "memory_cleanup_candidate_total": 3,
                    "duplicate_auto_memory_group_total": 1,
                }
            ],
        },
        "comparison": {
            "counts": {
                "newly_passing": 1,
                "newly_failing": 0,
                "new_tasks": 0,
                "removed_tasks": 0,
            }
        },
        "repos": [{"name": "demo", "ok": True, "tasks": []}],
    }

    text = eval_runner._format_trend_summary([(old_path, old_artifact), (new_path, new_artifact)])
    assert "Trend Window: 2 artifacts" in text
    assert "Retrieval: repos 0/1 -> 1/1 | tasks 1/2 -> 2/2" in text
    assert "Operational: index 12.5s -> 8.1s | fallbacks 1 -> 0 | noise code 1 -> 0 docs 1 -> 0" in text
    assert "Memory: cleanup 0 -> 3 | duplicate auto groups 0 -> 1" in text
    assert "Latest delta: +1 passing -0 failing | new 0 | removed 0" in text


def test_main_trends_reads_recent_matching_artifacts(tmp_path: Path, monkeypatch, capsys):
    eval_runner = _load_eval_runner()
    manifest_path = tmp_path / "demo.toml"
    manifest_path.write_text("[[repo]]\nname='demo'\npath='~/dev/demo'\n")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for index in range(3):
        artifact_path = artifact_dir / f"demo.20260323T00000{index}Z.json"
        artifact_path.write_text(
            __import__("json").dumps(
                {
                    "generated_at": f"2026-03-23T00:00:0{index}Z",
                    "embedding_provider": "openrouter",
                    "summary": {
                        "repo_total": 1,
                        "passed_repo_total": 1,
                        "task_total": 1,
                        "passed_task_total": 1,
                        "fallback_task_total": index,
                        "irrelevant_code_total": 0,
                        "irrelevant_docs_total": 0,
                        "memory_cleanup_candidate_total": index,
                        "duplicate_auto_memory_group_total": 0,
                        "total_index_seconds": float(index + 1),
                        "repos": [{"name": "demo", "task_total": 1, "passed_task_total": 1, "index_seconds": float(index + 1)}],
                    },
                    "repos": [{"name": "demo", "ok": True, "tasks": []}],
                }
            )
            + "\n"
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_retrieval_eval.py",
            str(manifest_path),
            "--artifact-dir",
            str(artifact_dir),
            "--trends",
            "--trend-limit",
            "2",
        ],
    )

    exit_code = eval_runner.main()
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trend Window: 2 artifacts" in output
    assert "2026-03-23T00:00:01Z" in output
    assert "2026-03-23T00:00:02Z" in output


def test_format_release_evidence_combines_eval_and_persistent_memory(tmp_path: Path):
    eval_runner = _load_eval_runner()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    manifest_path = tmp_path / "demo.toml"
    manifest_path.write_text("[[repo]]\nname='demo'\npath='~/dev/demo'\n")
    repo_path = tmp_path / "demo-repo"
    repo_path.mkdir()

    (artifact_dir / "demo.20260323T000000Z.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-03-23T00:00:00Z",
                "manifest": str(manifest_path.resolve()),
                "embedding_provider": "openrouter",
                "summary": {
                    "repo_total": 1,
                    "passed_repo_total": 1,
                    "task_total": 1,
                    "passed_task_total": 1,
                    "fallback_task_total": 0,
                    "irrelevant_code_total": 0,
                    "irrelevant_docs_total": 0,
                    "memory_cleanup_candidate_total": 0,
                    "duplicate_auto_memory_group_total": 0,
                    "total_index_seconds": 3.2,
                    "repos": [{"name": "demo", "task_total": 1, "passed_task_total": 1, "index_seconds": 3.2}],
                },
                "repos": [{"name": "demo", "ok": True, "tasks": []}],
            }
        )
        + "\n"
    )
    (artifact_dir / "persistent-memory.demo-repo.20260323T000000Z.json").write_text(
        __import__("json").dumps(
            {
                "generated_at": "2026-03-23T00:00:00Z",
                "repo_name": "demo-repo",
                "repo_path": str(repo_path),
                "project_id": "demo-project",
                "summary": {
                    "total_memories": 5,
                    "current_project_memories": 5,
                    "stale_memories": 0,
                    "superseded_memories": 1,
                    "cleanup_candidate_total": 2,
                    "duplicate_auto_memory_groups": 0,
                },
            }
        )
        + "\n"
    )

    text = eval_runner._format_release_evidence(
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        repo_path=repo_path,
        trend_limit=5,
    )
    assert "## Release Evidence" in text
    assert "### Retrieval Snapshot" in text
    assert "### Persistent Memory Snapshot" in text
    assert "### Retrieval Trends" in text
    assert "### Persistent Memory Trends" in text


def test_persistent_memory_artifact_and_summary(tmp_db, mock_embedder, tmp_path: Path):
    eval_runner = _load_eval_runner()
    repo_path = tmp_path / "demo-repo"
    repo_path.mkdir()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    import vibe_rag.server as srv

    old_project_id = srv._project_id
    old_cwd = os.getcwd()
    srv._project_id = "demo-project"
    try:
        os.chdir(repo_path)
        artifact = eval_runner._persistent_memory_artifact(repo_path, artifact_dir)
    finally:
        srv._project_id = old_project_id
        os.chdir(old_cwd)

    assert artifact["repo_name"] == "demo-repo"
    assert artifact["summary"]["total_memories"] >= 0

    artifact_path = artifact_dir / "persistent-memory.demo-repo.20260323T000000Z.json"
    artifact_path.write_text(__import__("json").dumps(artifact) + "\n")
    text = eval_runner._format_persistent_memory_summary(artifact, artifact_path)
    assert "Repo: demo-repo" in text
    assert "Memory: total" in text
    assert "Cleanup:" in text


def test_format_persistent_memory_trend_summary(tmp_path: Path):
    eval_runner = _load_eval_runner()
    first_path = tmp_path / "persistent-memory.demo.20260323T000000Z.json"
    second_path = tmp_path / "persistent-memory.demo.20260323T010000Z.json"
    first = {
        "generated_at": "2026-03-23T00:00:00Z",
        "summary": {
            "total_memories": 5,
            "stale_memories": 1,
            "superseded_memories": 0,
            "cleanup_candidate_total": 2,
            "duplicate_auto_memory_groups": 1,
        },
    }
    second = {
        "generated_at": "2026-03-23T01:00:00Z",
        "summary": {
            "total_memories": 7,
            "stale_memories": 0,
            "superseded_memories": 1,
            "cleanup_candidate_total": 3,
            "duplicate_auto_memory_groups": 0,
        },
        "comparison": {
            "deltas": {
                "total_memories": 2,
                "stale_memories": -1,
                "cleanup_candidate_total": 1,
                "duplicate_auto_memory_groups": -1,
            }
        },
    }
    text = eval_runner._format_persistent_memory_trend_summary([(first_path, first), (second_path, second)])
    assert "Persistent Memory Trend Window: 2 artifacts" in text
    assert "Memory: total 5 -> 7 | stale 1 -> 0 | superseded 0 -> 1" in text
    assert "Cleanup: 2 -> 3 | duplicate auto groups 1 -> 0" in text
    assert "Latest delta: total +2 | stale -1 | cleanup +1 | dup auto -1" in text


def test_format_artifact_summary_reports_stale_git_heads(tmp_path: Path, monkeypatch):
    eval_runner = _load_eval_runner()
    repo_path = tmp_path / "demo-repo"
    repo_path.mkdir()
    artifact = {
        "generated_at": "2026-03-23T03:00:00Z",
        "manifest": str((tmp_path / "demo.toml").resolve()),
        "embedding_provider": "openrouter",
        "summary": {
            "repo_total": 1,
            "passed_repo_total": 1,
            "task_total": 1,
            "passed_task_total": 1,
            "fallback_task_total": 0,
            "irrelevant_code_total": 0,
            "irrelevant_docs_total": 0,
            "total_index_seconds": 1.0,
            "repos": [
                {
                    "name": "demo",
                    "task_total": 1,
                    "passed_task_total": 1,
                    "failed_queries": [],
                    "fallback_queries": [],
                    "index_seconds": 1.0,
                    "fallback_task_total": 0,
                }
            ],
        },
        "repos": [
            {
                "name": "demo",
                "path": str(repo_path),
                "tasks": [
                    {
                        "stale": {
                            "metadata": {
                                "git_head": "old-head",
                            }
                        }
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(eval_runner, "_current_git_head", lambda path: "new-head")

    text = eval_runner._format_artifact_summary(artifact, tmp_path / "demo.json")
    assert "Artifact freshness: stale repos 1 | missing repos 0" in text
    assert "stale: demo" in text


def test_format_artifact_summary_reports_current_when_git_heads_match(tmp_path: Path, monkeypatch):
    eval_runner = _load_eval_runner()
    repo_path = tmp_path / "demo-repo"
    repo_path.mkdir()
    artifact = {
        "generated_at": "2026-03-23T03:00:00Z",
        "manifest": str((tmp_path / "demo.toml").resolve()),
        "embedding_provider": "openrouter",
        "summary": {
            "repo_total": 1,
            "passed_repo_total": 1,
            "task_total": 1,
            "passed_task_total": 1,
            "fallback_task_total": 0,
            "irrelevant_code_total": 0,
            "irrelevant_docs_total": 0,
            "total_index_seconds": 1.0,
            "repos": [
                {
                    "name": "demo",
                    "task_total": 1,
                    "passed_task_total": 1,
                    "failed_queries": [],
                    "fallback_queries": [],
                    "index_seconds": 1.0,
                    "fallback_task_total": 0,
                }
            ],
        },
        "repos": [
            {
                "name": "demo",
                "path": str(repo_path),
                "tasks": [
                    {
                        "stale": {
                            "metadata": {
                                "git_head": "same-head",
                            }
                        }
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(eval_runner, "_current_git_head", lambda path: "same-head")

    text = eval_runner._format_artifact_summary(artifact, tmp_path / "demo.json")
    assert "Artifact freshness: current" in text


def test_result_summary_tracks_timing_fallbacks_and_noise():
    eval_runner = _load_eval_runner()
    result = {
        "ok": False,
        "repos": [
            {
                "name": "demo",
                "ok": False,
                "index_result": "Indexed 3 code files in 12.5s",
                "memory_quality": {
                    "summary": {
                        "cleanup_candidate_total": 4,
                        "duplicate_auto_memory_groups": 2,
                    }
                },
                "tasks": [
                    {
                        "query": "task a",
                        "ok": True,
                        "attempt_count": 2,
                        "irrelevant_code": ["noise.py"],
                        "irrelevant_docs": [],
                    },
                    {
                        "query": "task b",
                        "ok": False,
                        "attempt_count": 1,
                        "irrelevant_code": [],
                        "irrelevant_docs": ["noise.md"],
                    },
                ],
            }
        ],
    }

    summary = eval_runner._result_summary(result)
    repo = summary["repos"][0]
    assert summary["fallback_task_total"] == 1
    assert summary["irrelevant_code_total"] == 1
    assert summary["irrelevant_docs_total"] == 1
    assert summary["memory_cleanup_candidate_total"] == 4
    assert summary["duplicate_auto_memory_group_total"] == 2
    assert summary["total_index_seconds"] == 12.5
    assert repo["fallback_task_total"] == 1
    assert repo["fallback_queries"] == ["task a"]
    assert repo["index_seconds"] == 12.5
    assert repo["memory_cleanup_candidate_total"] == 4
    assert repo["duplicate_auto_memory_group_total"] == 2
