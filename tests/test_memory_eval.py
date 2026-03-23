from __future__ import annotations

import vibe_rag.server as srv
from vibe_rag.tools import (
    load_session_context,
    remember,
    remember_structured,
    save_session_memory,
    save_session_summary,
    supersede_memory,
)


class MemoryEvalEmbedder:
    """Small keyword embedder to make memory usefulness scenarios deterministic."""

    KEYWORDS = (
        "auth",
        "gateway",
        "refresh",
        "constraint",
        "deploy",
        "blue",
        "green",
        "todo",
        "pipeline",
        "letter",
        "decision",
        "roles",
        "session",
        "context",
        "project",
        "id",
    )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        counts = [float(lowered.count(keyword)) for keyword in self.KEYWORDS]
        total = sum(counts) or 1.0
        normalized = [count / total for count in counts]
        return normalized + [0.0] * (1024 - len(normalized))

    def embed_text_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_code_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


def _clear_all_memories() -> None:
    project_db = srv._get_db()
    user_db = srv._get_user_db()
    for db in (project_db, user_db):
        conn = db._get_conn()
        conn.executescript("DELETE FROM memories_vec; DELETE FROM memories;")
        conn.commit()


def _memory_summaries(payload: dict) -> list[str]:
    return [str(item.get("summary") or "") for item in payload.get("memories", [])]


def test_memory_eval_structured_decision_beats_freeform_note(tmp_db):
    old_embedder = srv._embedder
    old_project_id = srv._project_id
    srv._embedder = MemoryEvalEmbedder()
    srv._project_id = "eval-repo"
    try:
        _clear_all_memories()
        remember("rough auth note about gateway ownership")
        remember_structured(
            summary="auth decision",
            details="Gateway owns auth token validation and the auth service owns refresh issuance.",
            memory_kind="decision",
        )

        payload = load_session_context("where should auth token validation live", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._embedder = old_embedder
        srv._project_id = old_project_id

    assert payload["ok"] is True
    assert payload["memories"][0]["memory_kind"] == "decision"
    assert payload["memories"][0]["provenance"]["source_type"] == "manual_structured"
    assert "Gateway owns auth token validation" in payload["memories"][0]["content"]


def test_memory_eval_constraint_and_todo_are_recoverable(tmp_db):
    old_embedder = srv._embedder
    old_project_id = srv._project_id
    srv._embedder = MemoryEvalEmbedder()
    srv._project_id = "eval-repo"
    try:
        _clear_all_memories()
        remember_structured(
            summary="deployment constraint",
            details="Use blue green deployment for the API gateway.",
            memory_kind="constraint",
        )
        remember_structured(
            summary="pipeline todo",
            details="Finish the warning-letter enrichment stage before backfill.",
            memory_kind="todo",
        )

        deploy_payload = load_session_context("what deployment constraint do we have for the gateway", memory_limit=4, code_limit=0, docs_limit=0)
        todo_payload = load_session_context("what is still open on the warning letter pipeline", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._embedder = old_embedder
        srv._project_id = old_project_id

    assert deploy_payload["memories"][0]["memory_kind"] == "constraint"
    assert "blue green deployment" in deploy_payload["memories"][0]["content"].lower()
    assert todo_payload["memories"][0]["memory_kind"] == "todo"
    assert "warning-letter enrichment stage" in todo_payload["memories"][0]["content"]


def test_memory_eval_superseded_memory_stays_out_of_the_way(tmp_db):
    old_embedder = srv._embedder
    old_project_id = srv._project_id
    srv._embedder = MemoryEvalEmbedder()
    srv._project_id = "eval-repo"
    try:
        _clear_all_memories()
        old_id = srv._get_user_db().remember_structured(
            summary="auth decision",
            content="auth decision\n\nAuth service validates every API token.",
            embedding=srv._get_embedder().embed_text_sync(["auth decision\n\nAuth service validates every API token."])[0],
            project_id="eval-repo",
            memory_kind="decision",
            metadata={"capture_kind": "manual"},
        )
        supersede_memory(
            old_memory_id=str(old_id),
            summary="auth decision",
            details="Gateway validates API tokens and the auth service only issues refresh tokens.",
            memory_kind="decision",
        )

        payload = load_session_context("where should auth token validation live now", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._embedder = old_embedder
        srv._project_id = old_project_id

    summaries = _memory_summaries(payload)
    assert payload["memories"][0]["provenance"]["is_superseded"] is False
    assert "Gateway validates API tokens" in payload["memories"][0]["content"]
    assert summaries.count("auth decision") == 1


def test_memory_eval_current_project_memory_beats_cross_project_memory(tmp_db):
    old_embedder = srv._embedder
    old_project_id = srv._project_id
    srv._embedder = MemoryEvalEmbedder()
    try:
        _clear_all_memories()
        srv._project_id = "sink-repo"
        srv._get_user_db().remember_structured(
            summary="auth constraint",
            content="Auth constraint for sink repo: gateway validates tokens.",
            embedding=srv._get_embedder().embed_text_sync(["Auth constraint for sink repo: gateway validates tokens."])[0],
            project_id="sink-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )
        srv._get_user_db().remember_structured(
            summary="auth constraint",
            content="Auth constraint for source repo: auth service validates tokens.",
            embedding=srv._get_embedder().embed_text_sync(["Auth constraint for source repo: auth service validates tokens."])[0],
            project_id="source-repo",
            memory_kind="constraint",
            metadata={"capture_kind": "manual"},
        )

        payload = load_session_context("what auth constraint do we have", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._embedder = old_embedder
        srv._project_id = old_project_id

    assert payload["memories"][0]["provenance"]["is_current_project"] is True
    assert payload["memories"][0]["provenance"]["is_stale"] is False
    assert len(payload["memories"]) == 1


def test_memory_eval_duplicate_and_low_signal_auto_memories_do_not_interfere(tmp_db):
    old_embedder = srv._embedder
    old_project_id = srv._project_id
    srv._embedder = MemoryEvalEmbedder()
    srv._project_id = "eval-repo"
    try:
        _clear_all_memories()
        save_session_summary(
            task="hi",
            turns=[{"user": "hi", "assistant": "hello"}],
            source_session_id="sess-hi",
            source_message_id="msg-hi",
        )
        first = save_session_memory(
            task="Reply with only the project id loaded in session context, or NONE if no session context was loaded.",
            response="eval-repo",
            source_session_id="sess-project-1",
            source_message_id="msg-project-1",
        )
        second = save_session_memory(
            task="Reply with only the project id loaded in session context, or NONE if no session context was loaded.",
            response="eval-repo",
            source_session_id="sess-project-2",
            source_message_id="msg-project-2",
        )
        remember_structured(
            summary="auth decision",
            details="Gateway validates API tokens and the auth service issues refresh tokens.",
            memory_kind="decision",
        )

        payload = load_session_context("where should auth token validation live", memory_limit=4, code_limit=0, docs_limit=0)
    finally:
        srv._embedder = old_embedder
        srv._project_id = old_project_id

    assert first["ok"] is True
    assert first["skipped"] is True
    assert second["ok"] is True
    assert second["skipped"] is True
    assert payload["memories"][0]["memory_kind"] == "decision"
    assert all(
        item["provenance"]["capture_kind"] != "session_rollup"
        for item in payload["memories"]
    )
