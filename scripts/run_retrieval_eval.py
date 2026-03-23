from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import vibe_rag.server as srv
from vibe_rag.tools import _index_project_impl, load_session_context, memory_quality_report


ELAPSED_SECONDS_RE = re.compile(r"in (?P<seconds>\d+(?:\.\d+)?)s")


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
    primary_query = str(task["query"])
    queries = [primary_query]
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
            "task_query": primary_query,
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
                "memory_quality": memory_quality_report(limit=5),
                "tasks": task_results,
            }
        )

    overall_ok = all(repo["ok"] for repo in repo_results)
    return {"ok": overall_ok, "repos": repo_results}


def _task_key(task: dict[str, Any]) -> str:
    return str(task.get("task_query") or task.get("query") or "")


def _repo_summary(repo: dict[str, Any]) -> dict[str, Any]:
    tasks = list(repo.get("tasks", []))
    memory_quality = repo.get("memory_quality") or {}
    memory_summary = memory_quality.get("summary") or {}
    passed = [task for task in tasks if task.get("ok") is True]
    failed = [task for task in tasks if task.get("ok") is not True]
    fallback_tasks = [task for task in tasks if int(task.get("attempt_count") or 1) > 1]
    noisy_tasks = [
        task
        for task in tasks
        if (task.get("irrelevant_code") or []) or (task.get("irrelevant_docs") or [])
    ]
    index_seconds = _extract_elapsed_seconds(str(repo.get("index_result") or ""))
    return {
        "name": repo.get("name"),
        "path": repo.get("path"),
        "ok": repo.get("ok") is True,
        "index_seconds": index_seconds,
        "task_total": len(tasks),
        "passed_task_total": len(passed),
        "failed_task_total": len(failed),
        "fallback_task_total": len(fallback_tasks),
        "noisy_task_total": len(noisy_tasks),
        "irrelevant_code_total": sum(len(task.get("irrelevant_code") or []) for task in tasks),
        "irrelevant_docs_total": sum(len(task.get("irrelevant_docs") or []) for task in tasks),
        "memory_cleanup_candidate_total": int(memory_summary.get("cleanup_candidate_total") or 0),
        "duplicate_auto_memory_group_total": int(memory_summary.get("duplicate_auto_memory_groups") or 0),
        "passed_queries": [_task_key(task) for task in passed],
        "failed_queries": [_task_key(task) for task in failed],
        "fallback_queries": [_task_key(task) for task in fallback_tasks],
    }


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    repos = list(result.get("repos", []))
    repo_summaries = [_repo_summary(repo) for repo in repos]
    return {
        "ok": result.get("ok") is True,
        "repo_total": len(repos),
        "passed_repo_total": sum(1 for repo in repos if repo.get("ok") is True),
        "failed_repo_total": sum(1 for repo in repos if repo.get("ok") is not True),
        "task_total": sum(item["task_total"] for item in repo_summaries),
        "passed_task_total": sum(item["passed_task_total"] for item in repo_summaries),
        "failed_task_total": sum(item["failed_task_total"] for item in repo_summaries),
        "fallback_task_total": sum(item["fallback_task_total"] for item in repo_summaries),
        "noisy_task_total": sum(item["noisy_task_total"] for item in repo_summaries),
        "irrelevant_code_total": sum(item["irrelevant_code_total"] for item in repo_summaries),
        "irrelevant_docs_total": sum(item["irrelevant_docs_total"] for item in repo_summaries),
        "memory_cleanup_candidate_total": sum(item["memory_cleanup_candidate_total"] for item in repo_summaries),
        "duplicate_auto_memory_group_total": sum(item["duplicate_auto_memory_group_total"] for item in repo_summaries),
        "total_index_seconds": round(
            sum(float(item["index_seconds"]) for item in repo_summaries if item.get("index_seconds") is not None), 1
        ),
        "repos": repo_summaries,
    }


def _result_task_map(result: dict[str, Any]) -> dict[tuple[str, str], bool]:
    task_map: dict[tuple[str, str], bool] = {}
    for repo in result.get("repos", []):
        repo_name = str(repo.get("name") or repo.get("path") or "")
        for task in repo.get("tasks", []):
            task_map[(repo_name, _task_key(task))] = task.get("ok") is True
    return task_map


def _compare_results(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_tasks = _result_task_map(previous)
    current_tasks = _result_task_map(current)
    keys = sorted(set(previous_tasks) | set(current_tasks))

    newly_passing: list[dict[str, str]] = []
    newly_failing: list[dict[str, str]] = []
    unchanged_passing: list[dict[str, str]] = []
    unchanged_failing: list[dict[str, str]] = []
    new_tasks: list[dict[str, str]] = []
    removed_tasks: list[dict[str, str]] = []

    for repo_name, query in keys:
        entry = {"repo": repo_name, "query": query}
        previous_ok = previous_tasks.get((repo_name, query))
        current_ok = current_tasks.get((repo_name, query))
        if previous_ok is None:
            new_tasks.append(entry)
        elif current_ok is None:
            removed_tasks.append(entry)
        elif previous_ok is False and current_ok is True:
            newly_passing.append(entry)
        elif previous_ok is True and current_ok is False:
            newly_failing.append(entry)
        elif current_ok is True:
            unchanged_passing.append(entry)
        else:
            unchanged_failing.append(entry)

    return {
        "previous_ok": previous.get("ok") is True,
        "current_ok": current.get("ok") is True,
        "newly_passing": newly_passing,
        "newly_failing": newly_failing,
        "unchanged_passing": unchanged_passing,
        "unchanged_failing": unchanged_failing,
        "new_tasks": new_tasks,
        "removed_tasks": removed_tasks,
        "counts": {
            "newly_passing": len(newly_passing),
            "newly_failing": len(newly_failing),
            "unchanged_passing": len(unchanged_passing),
            "unchanged_failing": len(unchanged_failing),
            "new_tasks": len(new_tasks),
            "removed_tasks": len(removed_tasks),
        },
    }


def _latest_previous_artifact(manifest_path: Path, artifact_dir: Path) -> Path | None:
    pattern = f"{manifest_path.stem}.*.json"
    candidates = sorted(artifact_dir.glob(pattern))
    return candidates[-1] if candidates else None


def _matching_artifacts(manifest_path: Path, artifact_dir: Path) -> list[Path]:
    pattern = f"{manifest_path.stem}.*.json"
    return sorted(artifact_dir.glob(pattern))


def _persistent_memory_artifact_name(repo_path: Path) -> str:
    return f"persistent-memory.{repo_path.resolve().name}"


def _matching_persistent_memory_artifacts(repo_path: Path, artifact_dir: Path) -> list[Path]:
    pattern = f"{_persistent_memory_artifact_name(repo_path)}.*.json"
    return sorted(artifact_dir.glob(pattern))


def _latest_persistent_memory_artifact(repo_path: Path, artifact_dir: Path) -> Path | None:
    candidates = _matching_persistent_memory_artifacts(repo_path, artifact_dir)
    return candidates[-1] if candidates else None


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _current_git_head(repo_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _artifact_repo_git_head(repo: dict[str, Any]) -> str | None:
    for task in repo.get("tasks", []):
        stale = task.get("stale") or {}
        metadata = stale.get("metadata") or {}
        git_head = metadata.get("git_head")
        if git_head:
            return str(git_head)
    return None


def _persistent_memory_repo_git_head(artifact: dict[str, Any]) -> str | None:
    return str(artifact.get("git_head") or "") or None


def _artifact_stale_status(artifact: dict[str, Any]) -> dict[str, Any]:
    stale_repos: list[dict[str, str]] = []
    missing_repos: list[dict[str, str]] = []
    for repo in artifact.get("repos", []):
        repo_path_raw = repo.get("path")
        if not repo_path_raw:
            continue
        repo_path = Path(str(repo_path_raw)).expanduser().resolve()
        if not repo_path.exists():
            missing_repos.append({"repo": str(repo.get("name") or repo_path), "path": str(repo_path)})
            continue
        artifact_head = _artifact_repo_git_head(repo)
        current_head = _current_git_head(repo_path)
        if artifact_head and current_head and artifact_head != current_head:
            stale_repos.append(
                {
                    "repo": str(repo.get("name") or repo_path),
                    "artifact_git_head": artifact_head,
                    "current_git_head": current_head,
                }
            )
    return {
        "is_stale": bool(stale_repos or missing_repos),
        "stale_repos": stale_repos,
        "missing_repos": missing_repos,
    }


def _persistent_memory_stale_status(artifact: dict[str, Any]) -> dict[str, Any]:
    repo_path_raw = artifact.get("repo_path")
    if not repo_path_raw:
        return {"is_stale": False, "stale_repos": [], "missing_repos": []}
    repo_path = Path(str(repo_path_raw)).expanduser().resolve()
    if not repo_path.exists():
        return {
            "is_stale": True,
            "stale_repos": [],
            "missing_repos": [{"repo": str(artifact.get("repo_name") or repo_path.name), "path": str(repo_path)}],
        }
    artifact_head = _persistent_memory_repo_git_head(artifact)
    current_head = _current_git_head(repo_path)
    if artifact_head and current_head and artifact_head != current_head:
        return {
            "is_stale": True,
            "stale_repos": [
                {
                    "repo": str(artifact.get("repo_name") or repo_path.name),
                    "artifact_git_head": artifact_head,
                    "current_git_head": current_head,
                }
            ],
            "missing_repos": [],
        }
    return {"is_stale": False, "stale_repos": [], "missing_repos": []}


def _extract_elapsed_seconds(index_result: str) -> float | None:
    match = ELAPSED_SECONDS_RE.search(index_result)
    if not match:
        return None
    try:
        return float(match.group("seconds"))
    except ValueError:
        return None


def _format_artifact_summary(artifact: dict[str, Any], artifact_path: Path) -> str:
    computed_summary = _result_summary(artifact)
    summary = dict(computed_summary)
    stored_summary = artifact.get("summary") or {}
    summary.update(stored_summary)
    if computed_summary.get("repos") and stored_summary.get("repos"):
        merged_repos: list[dict[str, Any]] = []
        stored_repos = {
            str(repo.get("name") or repo.get("path") or ""): repo
            for repo in stored_summary.get("repos", [])
        }
        for repo in computed_summary["repos"]:
            repo_name = str(repo.get("name") or repo.get("path") or "")
            merged = dict(repo)
            merged.update(stored_repos.get(repo_name, {}))
            merged_repos.append(merged)
        summary["repos"] = merged_repos
    stale_status = _artifact_stale_status(artifact)
    lines = [
        f"Artifact: {artifact_path}",
        f"Generated: {artifact.get('generated_at') or 'unknown'}",
        f"Manifest: {artifact.get('manifest') or 'unknown'}",
        f"Provider: {artifact.get('embedding_provider') or 'unknown'}",
        (
            f"Repos: {summary.get('passed_repo_total', 0)}/{summary.get('repo_total', 0)} passed"
            f" | Tasks: {summary.get('passed_task_total', 0)}/{summary.get('task_total', 0)} passed"
        ),
        (
            f"Timing: {summary.get('total_index_seconds', 0.0)}s total index"
            f" | Fallbacks: {summary.get('fallback_task_total', 0)}"
            f" | Noise: code {summary.get('irrelevant_code_total', 0)} docs {summary.get('irrelevant_docs_total', 0)}"
        ),
        (
            f"Memory: cleanup {summary.get('memory_cleanup_candidate_total', 0)}"
            f" | duplicate auto groups {summary.get('duplicate_auto_memory_group_total', 0)}"
        ),
    ]

    comparison = artifact.get("comparison")
    if isinstance(comparison, dict):
        counts = comparison.get("counts") or {}
        lines.append(
            "Changes:"
            f" +{counts.get('newly_passing', 0)} passing"
            f" -{counts.get('newly_failing', 0)} failing"
            f" | new {counts.get('new_tasks', 0)}"
            f" | removed {counts.get('removed_tasks', 0)}"
        )
    if stale_status["is_stale"]:
        lines.append(
            "Artifact freshness:"
            f" stale repos {len(stale_status['stale_repos'])}"
            f" | missing repos {len(stale_status['missing_repos'])}"
        )
        for item in stale_status["stale_repos"][:3]:
            lines.append(f"  stale: {item['repo']}")
        for item in stale_status["missing_repos"][:3]:
            lines.append(f"  missing: {item['repo']}")
    else:
        lines.append("Artifact freshness: current")

    for repo in summary.get("repos", []):
        lines.append(
            f"- {repo['name']}: {repo['passed_task_total']}/{repo['task_total']} tasks passed"
            f" | {repo.get('index_seconds', 'unknown')}s"
            f" | fallbacks {repo.get('fallback_task_total', 0)}"
            f" | memory cleanup {repo.get('memory_cleanup_candidate_total', 0)}"
            f" | dup auto {repo.get('duplicate_auto_memory_group_total', 0)}"
        )
        failed_queries = repo.get("failed_queries") or []
        if failed_queries:
            lines.append(f"  failed: {', '.join(failed_queries)}")
        fallback_queries = repo.get("fallback_queries") or []
        if fallback_queries:
            lines.append(f"  fallback: {', '.join(fallback_queries)}")

    return "\n".join(lines)


def _trend_entry(artifact: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    summary = _result_summary(artifact)
    stored_summary = artifact.get("summary") or {}
    summary.update(stored_summary)
    return {
        "path": artifact_path,
        "generated_at": str(artifact.get("generated_at") or "unknown"),
        "provider": str(artifact.get("embedding_provider") or "unknown"),
        "summary": summary,
        "comparison": artifact.get("comparison") or {},
    }


def _format_trend_summary(artifacts: list[tuple[Path, dict[str, Any]]]) -> str:
    entries = [_trend_entry(artifact, path) for path, artifact in artifacts]
    lines = [
        f"Trend Window: {len(entries)} artifacts",
        f"First: {entries[0]['generated_at']} | Last: {entries[-1]['generated_at']}",
    ]

    first = entries[0]["summary"]
    last = entries[-1]["summary"]
    lines.append(
        "Retrieval:"
        f" repos {first.get('passed_repo_total', 0)}/{first.get('repo_total', 0)} -> {last.get('passed_repo_total', 0)}/{last.get('repo_total', 0)}"
        f" | tasks {first.get('passed_task_total', 0)}/{first.get('task_total', 0)} -> {last.get('passed_task_total', 0)}/{last.get('task_total', 0)}"
    )
    lines.append(
        "Operational:"
        f" index {first.get('total_index_seconds', 0.0)}s -> {last.get('total_index_seconds', 0.0)}s"
        f" | fallbacks {first.get('fallback_task_total', 0)} -> {last.get('fallback_task_total', 0)}"
        f" | noise code {first.get('irrelevant_code_total', 0)} -> {last.get('irrelevant_code_total', 0)}"
        f" docs {first.get('irrelevant_docs_total', 0)} -> {last.get('irrelevant_docs_total', 0)}"
    )
    lines.append(
        "Memory:"
        f" cleanup {first.get('memory_cleanup_candidate_total', 0)} -> {last.get('memory_cleanup_candidate_total', 0)}"
        f" | duplicate auto groups {first.get('duplicate_auto_memory_group_total', 0)} -> {last.get('duplicate_auto_memory_group_total', 0)}"
    )

    latest_comparison = entries[-1]["comparison"]
    if isinstance(latest_comparison, dict):
        counts = latest_comparison.get("counts") or {}
        lines.append(
            "Latest delta:"
            f" +{counts.get('newly_passing', 0)} passing"
            f" -{counts.get('newly_failing', 0)} failing"
            f" | new {counts.get('new_tasks', 0)}"
            f" | removed {counts.get('removed_tasks', 0)}"
        )

    lines.append("Artifacts:")
    for entry in entries:
        summary = entry["summary"]
        lines.append(
            f"- {entry['generated_at']}: repos {summary.get('passed_repo_total', 0)}/{summary.get('repo_total', 0)}"
            f" | tasks {summary.get('passed_task_total', 0)}/{summary.get('task_total', 0)}"
            f" | index {summary.get('total_index_seconds', 0.0)}s"
            f" | fallbacks {summary.get('fallback_task_total', 0)}"
            f" | memory cleanup {summary.get('memory_cleanup_candidate_total', 0)}"
            f" | dup auto {summary.get('duplicate_auto_memory_group_total', 0)}"
        )
    return "\n".join(lines)


def _persistent_memory_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    report = artifact.get("memory_quality") or {}
    summary = dict(report.get("summary") or artifact.get("summary") or {})
    summary.setdefault("total_memories", 0)
    summary.setdefault("current_project_memories", 0)
    summary.setdefault("stale_memories", 0)
    summary.setdefault("superseded_memories", 0)
    summary.setdefault("cleanup_candidate_total", 0)
    summary.setdefault("duplicate_auto_memory_groups", 0)
    return summary


def _compare_persistent_memory(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_summary = _persistent_memory_summary(previous)
    current_summary = _persistent_memory_summary(current)
    keys = (
        "total_memories",
        "current_project_memories",
        "stale_memories",
        "superseded_memories",
        "cleanup_candidate_total",
        "duplicate_auto_memory_groups",
    )
    deltas = {
        key: int(current_summary.get(key, 0)) - int(previous_summary.get(key, 0))
        for key in keys
    }
    return {
        "deltas": deltas,
    }


def _persistent_memory_artifact(repo_path: Path, artifact_dir: Path, previous_artifact_path: Path | None = None) -> dict[str, Any]:
    report = memory_quality_report(limit=10)
    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_name": repo_path.resolve().name,
        "repo_path": str(repo_path.resolve()),
        "project_id": report.get("project_id"),
        "git_head": _current_git_head(repo_path.resolve()),
        "memory_quality": report,
    }
    previous_path = previous_artifact_path or _latest_persistent_memory_artifact(repo_path, artifact_dir)
    artifact["previous_artifact"] = str(previous_path.resolve()) if previous_path else None
    artifact["summary"] = _persistent_memory_summary(artifact)
    if previous_path and previous_path.exists():
        artifact["comparison"] = _compare_persistent_memory(_load_artifact(previous_path), artifact)
    return artifact


def _write_persistent_memory_artifact(repo_path: Path, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    previous_artifact_path = _latest_persistent_memory_artifact(repo_path, artifact_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = artifact_dir / f"{_persistent_memory_artifact_name(repo_path)}.{timestamp}.json"
    artifact = _persistent_memory_artifact(repo_path, artifact_dir, previous_artifact_path)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return output_path


def _format_persistent_memory_summary(artifact: dict[str, Any], artifact_path: Path) -> str:
    summary = _persistent_memory_summary(artifact)
    stale_status = _persistent_memory_stale_status(artifact)
    lines = [
        f"Artifact: {artifact_path}",
        f"Generated: {artifact.get('generated_at') or 'unknown'}",
        f"Repo: {artifact.get('repo_name') or 'unknown'}",
        f"Project id: {artifact.get('project_id') or 'unknown'}",
        (
            f"Memory: total {summary.get('total_memories', 0)}"
            f" | current {summary.get('current_project_memories', 0)}"
            f" | stale {summary.get('stale_memories', 0)}"
            f" | superseded {summary.get('superseded_memories', 0)}"
        ),
        (
            f"Cleanup: {summary.get('cleanup_candidate_total', 0)}"
            f" | duplicate auto groups {summary.get('duplicate_auto_memory_groups', 0)}"
        ),
    ]
    comparison = artifact.get("comparison") or {}
    deltas = comparison.get("deltas") or {}
    if deltas:
        lines.append(
            "Delta:"
            f" total {deltas.get('total_memories', 0):+d}"
            f" | stale {deltas.get('stale_memories', 0):+d}"
            f" | cleanup {deltas.get('cleanup_candidate_total', 0):+d}"
            f" | dup auto {deltas.get('duplicate_auto_memory_groups', 0):+d}"
        )
    if stale_status["is_stale"]:
        lines.append(
            "Artifact freshness:"
            f" stale repos {len(stale_status['stale_repos'])}"
            f" | missing repos {len(stale_status['missing_repos'])}"
        )
    else:
        lines.append("Artifact freshness: current")
    return "\n".join(lines)


def _format_persistent_memory_trend_summary(artifacts: list[tuple[Path, dict[str, Any]]]) -> str:
    entries = []
    for path, artifact in artifacts:
        entries.append(
            {
                "path": path,
                "generated_at": str(artifact.get("generated_at") or "unknown"),
                "summary": _persistent_memory_summary(artifact),
                "comparison": artifact.get("comparison") or {},
            }
        )
    first = entries[0]["summary"]
    last = entries[-1]["summary"]
    lines = [
        f"Persistent Memory Trend Window: {len(entries)} artifacts",
        f"First: {entries[0]['generated_at']} | Last: {entries[-1]['generated_at']}",
        (
            f"Memory: total {first.get('total_memories', 0)} -> {last.get('total_memories', 0)}"
            f" | stale {first.get('stale_memories', 0)} -> {last.get('stale_memories', 0)}"
            f" | superseded {first.get('superseded_memories', 0)} -> {last.get('superseded_memories', 0)}"
        ),
        (
            f"Cleanup: {first.get('cleanup_candidate_total', 0)} -> {last.get('cleanup_candidate_total', 0)}"
            f" | duplicate auto groups {first.get('duplicate_auto_memory_groups', 0)} -> {last.get('duplicate_auto_memory_groups', 0)}"
        ),
    ]
    latest_deltas = (entries[-1]["comparison"] or {}).get("deltas") or {}
    if latest_deltas:
        lines.append(
            "Latest delta:"
            f" total {latest_deltas.get('total_memories', 0):+d}"
            f" | stale {latest_deltas.get('stale_memories', 0):+d}"
            f" | cleanup {latest_deltas.get('cleanup_candidate_total', 0):+d}"
            f" | dup auto {latest_deltas.get('duplicate_auto_memory_groups', 0):+d}"
        )
    lines.append("Artifacts:")
    for entry in entries:
        summary = entry["summary"]
        lines.append(
            f"- {entry['generated_at']}: total {summary.get('total_memories', 0)}"
            f" | stale {summary.get('stale_memories', 0)}"
            f" | cleanup {summary.get('cleanup_candidate_total', 0)}"
            f" | dup auto {summary.get('duplicate_auto_memory_groups', 0)}"
        )
    return "\n".join(lines)


def _format_release_evidence(
    *,
    manifest_path: Path,
    artifact_dir: Path,
    repo_path: Path,
    trend_limit: int,
) -> str:
    lines = ["## Release Evidence"]
    latest_eval = _latest_previous_artifact(manifest_path, artifact_dir)
    if latest_eval and latest_eval.exists():
        lines.append("")
        lines.append("### Retrieval Snapshot")
        lines.append(_format_artifact_summary(_load_artifact(latest_eval), latest_eval))
    matching_eval = _matching_artifacts(manifest_path, artifact_dir)
    if matching_eval:
        lines.append("")
        lines.append("### Retrieval Trends")
        selected = matching_eval[-max(1, trend_limit) :]
        lines.append(_format_trend_summary([(path, _load_artifact(path)) for path in selected]))

    latest_memory = _latest_persistent_memory_artifact(repo_path, artifact_dir)
    if latest_memory and latest_memory.exists():
        lines.append("")
        lines.append("### Persistent Memory Snapshot")
        lines.append(_format_persistent_memory_summary(_load_artifact(latest_memory), latest_memory))
    matching_memory = _matching_persistent_memory_artifacts(repo_path, artifact_dir)
    if matching_memory:
        lines.append("")
        lines.append("### Persistent Memory Trends")
        selected = matching_memory[-max(1, trend_limit) :]
        lines.append(_format_persistent_memory_trend_summary([(path, _load_artifact(path)) for path in selected]))

    return "\n".join(lines)


def _artifact_result(
    result: dict,
    manifest_path: Path,
    artifact_dir: Path,
    previous_artifact_path: Path | None = None,
) -> dict:
    artifact = dict(result)
    artifact["generated_at"] = datetime.now(UTC).isoformat()
    artifact["manifest"] = str(manifest_path.resolve())
    artifact["embedding_provider"] = os.environ.get("VIBE_RAG_EMBEDDING_PROVIDER", "ollama")
    artifact["embedding_model"] = os.environ.get("VIBE_RAG_EMBEDDING_MODEL", "")
    artifact["code_embedding_model"] = os.environ.get("VIBE_RAG_CODE_EMBEDDING_MODEL", "")
    artifact["summary"] = _result_summary(result)

    previous_path = previous_artifact_path or _latest_previous_artifact(manifest_path, artifact_dir)
    artifact["previous_artifact"] = str(previous_path.resolve()) if previous_path else None
    if previous_path and previous_path.exists():
        previous_payload = json.loads(previous_path.read_text())
        artifact["comparison"] = _compare_results(previous_payload, artifact)
    return artifact


def _write_artifact(result: dict, manifest_path: Path, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    previous_artifact_path = _latest_previous_artifact(manifest_path, artifact_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = artifact_dir / f"{manifest_path.stem}.{timestamp}.json"
    artifact = _artifact_result(result, manifest_path, artifact_dir, previous_artifact_path)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the latest saved artifact summary for the manifest instead of running evals.",
    )
    parser.add_argument(
        "--artifact",
        default="",
        help="Specific artifact JSON path to summarize. Implies --summary.",
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Print a cross-artifact trend summary for the manifest.",
    )
    parser.add_argument(
        "--trend-limit",
        type=int,
        default=5,
        help="How many matching artifacts to include in the trend view.",
    )
    parser.add_argument(
        "--persistent-memory",
        action="store_true",
        help="Write a persistent-memory snapshot artifact for a real repo path instead of running evals.",
    )
    parser.add_argument(
        "--persistent-memory-summary",
        action="store_true",
        help="Print the latest persistent-memory snapshot summary for the repo path.",
    )
    parser.add_argument(
        "--persistent-memory-trends",
        action="store_true",
        help="Print a trend summary across persistent-memory snapshots for the repo path.",
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Repo path for persistent-memory snapshot/summary/trend operations.",
    )
    parser.add_argument(
        "--release-evidence",
        action="store_true",
        help="Render a compact release-evidence report from the latest eval and persistent-memory artifacts.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    artifact_dir = Path(args.artifact_dir)
    repo_path = Path(args.repo_path).expanduser().resolve()

    if args.release_evidence:
        print(
            _format_release_evidence(
                manifest_path=manifest_path,
                artifact_dir=artifact_dir,
                repo_path=repo_path,
                trend_limit=max(1, int(args.trend_limit)),
            )
        )
        return 0

    if args.persistent_memory_trends:
        matching = _matching_persistent_memory_artifacts(repo_path, artifact_dir)
        if not matching:
            print("No persistent-memory artifacts found for trend summary.", file=sys.stderr)
            return 1
        limit = max(1, int(args.trend_limit))
        selected = matching[-limit:]
        payloads = [(path, _load_artifact(path)) for path in selected]
        print(_format_persistent_memory_trend_summary(payloads))
        return 0

    if args.persistent_memory_summary:
        artifact_path = _latest_persistent_memory_artifact(repo_path, artifact_dir)
        if artifact_path is None or not artifact_path.exists():
            print("No persistent-memory artifact found to summarize.", file=sys.stderr)
            return 1
        artifact = _load_artifact(artifact_path)
        print(_format_persistent_memory_summary(artifact, artifact_path))
        return 0

    if args.persistent_memory:
        old_cwd = Path.cwd()
        try:
            os.chdir(repo_path)
            artifact_path = _write_persistent_memory_artifact(repo_path, artifact_dir)
        finally:
            os.chdir(old_cwd)
        print(f"[artifact] wrote {artifact_path}", file=sys.stderr)
        artifact = _load_artifact(artifact_path)
        print(json.dumps(artifact, indent=2))
        return 0

    if args.trends:
        matching = _matching_artifacts(manifest_path, artifact_dir)
        if not matching:
            print("No artifacts found for trend summary.", file=sys.stderr)
            return 1
        limit = max(1, int(args.trend_limit))
        selected = matching[-limit:]
        payloads = [(path, _load_artifact(path)) for path in selected]
        print(_format_trend_summary(payloads))
        return 0

    if args.summary or args.artifact:
        artifact_path = Path(args.artifact) if args.artifact else _latest_previous_artifact(manifest_path, artifact_dir)
        if artifact_path is None or not artifact_path.exists():
            print("No artifact found to summarize.", file=sys.stderr)
            return 1
        artifact = _load_artifact(artifact_path)
        print(_format_artifact_summary(artifact, artifact_path))
        return 0

    result = run_manifest(manifest_path)
    artifact_path = _write_artifact(result, manifest_path, artifact_dir)
    print(f"[artifact] wrote {artifact_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
