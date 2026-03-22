from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path

from watchfiles import awatch, Change

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 200
KEEP_ROLES = {"user", "assistant"}


def parse_session_messages(jsonl_path: Path) -> str:
    lines: list[str] = []
    for raw_line in jsonl_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in KEEP_ROLES and content:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
    return "\n\n".join(lines)


def chunk_session_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + MAX_CHUNK_CHARS // 2:
                end = boundary + 2
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


def find_completed_sessions(log_dir: Path) -> list[dict]:
    completed = []
    if not log_dir.exists():
        return completed
    for session_dir in sorted(log_dir.iterdir()):
        meta_path = session_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("end_time"):
            completed.append({
                "session_id": meta.get("session_id", session_dir.name),
                "session_dir": session_dir,
                "meta": meta,
            })
    return completed


async def watch_session_dir(log_dir: Path, on_session_complete) -> None:
    """Watch for new completed sessions and call on_session_complete(session_info)."""
    if not log_dir.exists():
        logger.warning(f"Session log dir not found: {log_dir}. Watcher disabled.")
        return

    logger.info(f"Watching for new sessions in {log_dir}")
    async for changes in awatch(log_dir):
        for change_type, path_str in changes:
            path = Path(path_str)
            if path.name != "meta.json":
                continue
            if change_type not in (Change.added, Change.modified):
                continue
            try:
                meta = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not meta.get("end_time"):
                continue
            session_dir = path.parent
            session_info = {
                "session_id": meta.get("session_id", session_dir.name),
                "session_dir": session_dir,
                "meta": meta,
            }
            try:
                await on_session_complete(session_info)
            except Exception as e:
                logger.warning(f"Failed to index session {session_info['session_id']}: {e}")
