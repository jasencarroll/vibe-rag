from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
import tomllib
from contextlib import contextmanager
from pathlib import Path

import vibe_rag.server as srv
from vibe_rag.tools import _index_project_impl, load_session_context


def _read_manifest(path: Path) -> dict:
    return tomllib.loads(path.read_text())


@contextmanager
def _repo_eval_context(repo_path: Path):
    old_cwd = Path.cwd()
    old_env = {
        "VIBE_RAG_DB": os.environ.get("VIBE_RAG_DB"),
        "VIBE_RAG_USER_DB": os.environ.get("VIBE_RAG_USER_DB"),
    }
    old_project_db = srv._project_db
    old_user_db = srv._user_db
    old_project_id = srv._project_id

    with tempfile.TemporaryDirectory(prefix="vibe-rag-eval.") as tmp_dir:
        tmp_root = Path(tmp_dir)
        os.environ["VIBE_RAG_DB"] = str(tmp_root / "project.db")
        os.environ["VIBE_RAG_USER_DB"] = str(tmp_root / "user.db")
        srv._project_db = None
        srv._user_db = None
        srv._project_id = None
        os.chdir(repo_path)
        try:
            yield
        finally:
            os.chdir(old_cwd)
            if srv._project_db is not None:
                srv._project_db.close()
            if srv._user_db is not None:
                srv._user_db.close()
            srv._project_db = old_project_db
            srv._user_db = old_user_db
            srv._project_id = old_project_id
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _eval_task(task: dict) -> dict:
    queries = [str(task["query"])]
    for candidate in task.get("query_variants", []):
        candidate_query = str(candidate).strip()
        if candidate_query and candidate_query not in queries:
            queries.append(candidate_query)

    expected_code = set(task.get("expected_code", []))
    expected_docs = set(task.get("expected_docs", []))
    unexpected_code = set(task.get("unexpected_code", []))
    unexpected_docs = set(task.get("unexpected_docs", []))
    max_irrelevant_code = int(task.get("max_irrelevant_code", 1))
    max_irrelevant_docs = int(task.get("max_irrelevant_docs", 1))

    attempts: list[dict] = []
    for attempt_index, query in enumerate(queries, start=1):
        if len(queries) > 1:
            print(
                f"[task] query attempt {attempt_index}/{len(queries)}: {query}",
                file=sys.stderr,
            )

        payload = load_session_context(
            query,
            memory_limit=int(task.get("memory_limit", 3)),
            code_limit=int(task.get("code_limit", 5)),
            docs_limit=int(task.get("docs_limit", 3)),
        )

        code_hits = {item["file_path"] for item in payload.get("code", [])}
        doc_hits = {item["file_path"] for item in payload.get("docs", [])}
        expected_code_found = sorted(expected_code & code_hits)
        expected_docs_found = sorted(expected_docs & doc_hits)
        irrelevant_code = sorted(code_hits & unexpected_code)
        irrelevant_docs = sorted(doc_hits & unexpected_docs)

        ok = (
            payload.get("ok") is True
            and (not expected_code or bool(expected_code_found))
            and (not expected_docs or bool(expected_docs_found))
            and len(irrelevant_code) <= max_irrelevant_code
            and len(irrelevant_docs) <= max_irrelevant_docs
        )

        attempt = {
            "query": query,
            "ok": ok,
            "project_id": payload.get("project_id"),
            "expected_code_found": expected_code_found,
            "expected_docs_found": expected_docs_found,
            "irrelevant_code": irrelevant_code,
            "irrelevant_docs": irrelevant_docs,
            "code_hits": sorted(code_hits),
            "doc_hits": sorted(doc_hits),
            "stale": payload.get("stale"),
        }
        attempts.append(attempt)
        if ok:
            return _finalize_attempt(attempt, attempts)

    return _finalize_attempt(attempts[-1], attempts)


def _finalize_attempt(selected_attempt: dict, attempts: list[dict]) -> dict:
    result = dict(selected_attempt)
    result["attempt_count"] = len(attempts)
    if len(attempts) > 1:
        result["attempts"] = [dict(attempt) for attempt in attempts]
    return result


def _print_progress(event: dict[str, object]) -> None:
    phase = str(event.get("phase") or "")
    if phase == "file_discovery_complete":
        print(
            f"[index] discovered {event['code_file_total']} code files and {event['doc_file_total']} docs",
            file=sys.stderr,
        )
        return
    if phase == "code_chunking_start":
        print(f"[index] chunking code ({event['file_total']} files)", file=sys.stderr)
        return
    if phase == "code_chunking_complete":
        print(
            f"[index] code chunking complete: {event['chunk_total']} chunks, {event['unchanged_total']} unchanged",
            file=sys.stderr,
        )
        return
    if phase == "doc_chunking_start":
        print(f"[index] chunking docs ({event['file_total']} files)", file=sys.stderr)
        return
    if phase == "doc_chunking_complete":
        print(
            f"[index] doc chunking complete: {event['chunk_total']} chunks, {event['unchanged_total']} unchanged",
            file=sys.stderr,
        )
        return
    if phase in {"code_embedding_start", "doc_embedding_start"}:
        label = "code" if phase.startswith("code") else "docs"
        print(f"[index] embedding {label} ({event['chunk_total']} chunks)", file=sys.stderr)
        return
    if phase in {"embedding_batch_start", "embedding_batch_complete"}:
        marker = "batch" if phase.endswith("start") else "done"
        print(
            f"[embed] {event['provider']} {event['input_kind']} {marker} "
            f"{event['batch_current']}/{event['batch_total']} "
            f"items {event['items_completed']}/{event['items_total']}",
            file=sys.stderr,
        )
        return
    if phase == "index_complete":
        print(
            f"[index] complete in {event['elapsed_seconds']}s "
            f"({event['code_chunk_total']} code chunks, {event['doc_chunk_total']} doc chunks)",
            file=sys.stderr,
        )


def run_manifest(manifest_path: Path) -> dict:
    manifest = _read_manifest(manifest_path)
    repo_results: list[dict] = []

    for repo in manifest.get("repo", []):
        repo_path = Path(str(repo["path"])).expanduser().resolve()
        if not repo_path.exists():
            repo_results.append(
                {
                    "name": repo.get("name") or str(repo_path),
                    "path": str(repo_path),
                    "ok": False,
                    "error": "repo path does not exist",
                    "tasks": [],
                }
            )
            continue

        with _repo_eval_context(repo_path):
            print(f"[repo] indexing {repo.get('name') or repo_path.name}", file=sys.stderr)
            index_result = _index_project_impl(progress_callback=_print_progress)
            task_results = []
            tasks = repo.get("task", [])
            for task_index, task in enumerate(tasks, start=1):
                print(
                    f"[repo] task {task_index}/{len(tasks)}: {task['query']}",
                    file=sys.stderr,
                )
                task_results.append(_eval_task(task))

        repo_results.append(
            {
                "name": repo.get("name") or repo_path.name,
                "path": str(repo_path),
                "ok": all(task["ok"] for task in task_results),
                "index_result": index_result,
                "tasks": task_results,
            }
        )

    overall_ok = all(repo["ok"] for repo in repo_results)
    return {"ok": overall_ok, "repos": repo_results}


def _artifact_result(result: dict, manifest_path: Path) -> dict:
    artifact = dict(result)
    artifact["generated_at"] = datetime.now(UTC).isoformat()
    artifact["manifest"] = str(manifest_path.resolve())
    artifact["embedding_provider"] = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "ollama")
    artifact["embedding_model"] = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", "")
    artifact["code_embedding_model"] = os.environ.get("VIBE_RAG_CODE_EMBEDDING_MODEL", "")
    return artifact


def _write_artifact(result: dict, manifest_path: Path, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = artifact_dir / f"{manifest_path.stem}.{timestamp}.json"
    output_path.write_text(json.dumps(_artifact_result(result, manifest_path), indent=2) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local retrieval evals against real repos.")
    parser.add_argument(
        "manifest",
        nargs="?",
        default="evals/local_repos.toml",
        help="Path to a TOML manifest describing repos and retrieval tasks.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="evals/results",
        help="Directory where a timestamped JSON artifact will be written.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    result = run_manifest(manifest_path)
    artifact_path = _write_artifact(result, manifest_path, Path(args.artifact_dir))
    print(f"[artifact] wrote {artifact_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
