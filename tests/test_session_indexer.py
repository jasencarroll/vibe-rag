import json
from pathlib import Path
from vibe_memory.indexing.session_indexer import parse_session_messages, chunk_session_text, find_completed_sessions


def test_parse_session_messages(tmp_path: Path):
    messages = [
        {"role": "user", "content": "Hello, help me with auth"},
        {"role": "assistant", "content": "Sure, I can help with authentication."},
        {"role": "tool_call", "content": "read_file(src/auth.py)"},
        {"role": "tool_result", "content": "file contents here..."},
        {"role": "assistant", "content": "Here's what I found in the auth module."},
    ]
    jsonl_path = tmp_path / "messages.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(m) for m in messages))
    text = parse_session_messages(jsonl_path)
    assert "Hello, help me with auth" in text
    assert "Sure, I can help" in text
    assert "read_file" not in text


def test_chunk_session_text():
    text = "word " * 1000
    chunks = chunk_session_text(text)
    assert len(chunks) >= 2


def test_find_completed_sessions(tmp_path: Path):
    s1 = tmp_path / "session_001"
    s1.mkdir()
    (s1 / "meta.json").write_text(json.dumps({
        "session_id": "sess-001", "end_time": "2026-03-22T10:30:00Z",
        "environment": {"working_directory": "/tmp/project"},
    }))
    (s1 / "messages.jsonl").write_text("")
    s2 = tmp_path / "session_002"
    s2.mkdir()
    (s2 / "meta.json").write_text(json.dumps({
        "session_id": "sess-002", "environment": {"working_directory": "/tmp/project"},
    }))
    (s2 / "messages.jsonl").write_text("")
    completed = find_completed_sessions(tmp_path)
    assert len(completed) == 1
    assert completed[0]["session_id"] == "sess-001"
