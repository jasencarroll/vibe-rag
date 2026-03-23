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
        padded = normalized + [0.0] * (1024 - len(normalized))
        return padded

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


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
            assert "Indexed" in index_result

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
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("VIBE_RAG_EMBEDDING_MODEL", "voyage-4")

    output_path = eval_runner._write_artifact(
        {"ok": True, "repos": [{"name": "demo", "ok": True}]},
        manifest_path,
        tmp_path / "artifacts",
    )

    payload = __import__("json").loads(output_path.read_text())
    assert output_path.parent == tmp_path / "artifacts"
    assert payload["ok"] is True
    assert payload["manifest"] == str(manifest_path.resolve())
    assert payload["embedding_provider"] == "voyage"
