"""Tests for vibe_rag.indexing.code_chunker module."""
from __future__ import annotations

from vibe_rag.indexing.code_chunker import (
    chunk_code_sliding_window,
    chunk_code,
    _subsplit_large_chunks,
    _try_tree_sitter_chunk,
    MAX_CHUNK_LINES,
)


# --- Existing tests (preserved) ---

def test_sliding_window_basic():
    lines = [f"line {i}" for i in range(120)]
    content = "\n".join(lines)
    chunks = chunk_code_sliding_window(content, "test.py", window=60, overlap=10)
    assert len(chunks) == 3
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 60
    assert chunks[1]["start_line"] == 51
    assert chunks[1]["end_line"] == 110
    assert chunks[2]["start_line"] == 101
    assert chunks[2]["end_line"] == 120
    assert max(chunk["end_line"] - chunk["start_line"] + 1 for chunk in chunks) <= 60


def test_sliding_window_small_file():
    content = "x = 1\ny = 2\n"
    chunks = chunk_code_sliding_window(content, "small.py", window=60, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["content"] == content


def test_chunk_code_returns_chunks():
    code = '''
def hello():
    print("hi")

def world():
    print("world")

class Foo:
    def bar(self):
        return 42
'''
    chunks = chunk_code(code, "example.py", "python")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "file_path" in chunk
        assert "content" in chunk
        assert "start_line" in chunk
        assert "end_line" in chunk


def test_chunk_code_unsupported_language_falls_back():
    code = "\n".join([f"line {i}" for i in range(120)])
    chunks = chunk_code(code, "test.weird", "weirdlang")
    assert len(chunks) >= 1


# --- New edge case tests ---

class TestChunkCodeSlidingWindow:
    def test_empty_content(self):
        assert chunk_code_sliding_window("", "empty.py") == []

    def test_single_line(self):
        chunks = chunk_code_sliding_window("x = 1\n", "one.py")
        assert len(chunks) == 1
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["language"] is None
        assert chunks[0]["symbol"] is None

    def test_chunk_indices_sequential(self):
        lines = ["a\n" for _ in range(150)]
        content = "".join(lines)
        chunks = chunk_code_sliding_window(content, "f.py")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_custom_window_and_overlap(self):
        lines = [f"x{i}\n" for i in range(30)]
        content = "".join(lines)
        chunks = chunk_code_sliding_window(content, "f.py", window=10, overlap=2)
        assert len(chunks) >= 2

    def test_last_chunk_does_not_absorb_past_window_size(self):
        lines = [f"line {i}\n" for i in range(119)]
        chunks = chunk_code_sliding_window("".join(lines), "f.py", window=60, overlap=10)
        assert max(chunk["end_line"] - chunk["start_line"] + 1 for chunk in chunks) <= 60


class TestTryTreeSitterChunk:
    def test_unknown_language_returns_none(self):
        result = _try_tree_sitter_chunk("x = 1", "f.xyz", "unknown_lang")
        assert result is None

    def test_no_mapped_language_returns_none(self):
        result = _try_tree_sitter_chunk("x = 1", "f.txt", "plaintext")
        assert result is None

    def test_supported_language_returns_symbol_chunks(self):
        code = """
def hello():
    return 1

class Greeter:
    def greet(self):
        return hello()
"""
        result = _try_tree_sitter_chunk(code, "f.py", "python")
        assert result is not None
        assert len(result) >= 2
        assert {chunk["symbol"] for chunk in result} >= {"hello", "Greeter"}

    def test_supported_language_falls_back_to_safe_parser_when_get_parser_breaks(self, monkeypatch):
        import tree_sitter_languages

        code = """
def hello():
    return 1
"""

        monkeypatch.setattr(
            tree_sitter_languages,
            "get_parser",
            lambda _lang: (_ for _ in ()).throw(TypeError("broken wrapper")),
        )

        result = _try_tree_sitter_chunk(code, "f.py", "python")
        assert result is not None
        assert {chunk["symbol"] for chunk in result} >= {"hello"}

    def test_falls_back_to_sliding_window_when_tree_sitter_library_is_untrusted(self, tmp_path, monkeypatch):
        import tree_sitter_languages

        code = "def hello():\n    return 1\n"
        fake_pkg = tmp_path / "tree_sitter_languages"
        fake_pkg.mkdir()
        fake_pkg.joinpath("__init__.py").write_text("__version__ = '0.0.0'\n")
        fake_pkg.joinpath("languages.so").touch()

        monkeypatch.setattr(tree_sitter_languages, "__file__", str(fake_pkg / "__init__.py"), raising=False)
        monkeypatch.setattr(
            tree_sitter_languages,
            "get_parser",
            lambda _lang: (_ for _ in ()).throw(TypeError("shimmed parser")),
        )
        monkeypatch.setattr(
            tree_sitter_languages,
            "get_language",
            lambda _lang: (_ for _ in ()).throw(TypeError("shimmed language")),
        )

        chunks = chunk_code(code, "f.py", "python")
        assert len(chunks) == 1
        assert chunks[0]["content"] == code

    def test_supported_language_falls_back_to_sliding_window_when_get_parser_and_get_language_breaks(self, monkeypatch):
        import tree_sitter_languages

        code = "def hello():\n    return 1\n"
        monkeypatch.setattr(
            tree_sitter_languages,
            "get_parser",
            lambda _lang: (_ for _ in ()).throw(TypeError("broken parser")),
        )
        monkeypatch.setattr(
            tree_sitter_languages,
            "get_language",
            lambda _lang: (_ for _ in ()).throw(TypeError("broken language")),
        )

        chunks = chunk_code(code, "f.py", "python")
        assert len(chunks) == 1
        assert chunks[0]["content"] == code


class TestSubsplitLargeChunks:
    def test_small_chunks_unchanged(self):
        chunks = [
            {"file_path": "f.py", "chunk_index": 0, "content": "a\nb\nc\n",
             "language": "python", "symbol": "foo", "start_line": 1, "end_line": 3},
        ]
        result = _subsplit_large_chunks(chunks)
        assert len(result) == 1
        assert result[0]["chunk_index"] == 0

    def test_large_chunk_gets_split(self):
        big_content = "\n".join([f"line {i}" for i in range(MAX_CHUNK_LINES + 50)])
        chunks = [
            {"file_path": "f.py", "chunk_index": 0, "content": big_content,
             "language": "python", "symbol": "big_func", "start_line": 1,
             "end_line": MAX_CHUNK_LINES + 50},
        ]
        result = _subsplit_large_chunks(chunks)
        assert len(result) >= 2
        for sc in result:
            assert sc["language"] == "python"
            assert sc["symbol"] == "big_func"
        assert [c["chunk_index"] for c in result] == list(range(len(result)))

    def test_empty_list(self):
        assert _subsplit_large_chunks([]) == []


class TestChunkCode:
    def test_no_language_uses_sliding_window(self):
        content = "x = 1\ny = 2\n"
        chunks = chunk_code(content, "f.py", language=None)
        assert len(chunks) >= 1
        assert chunks[0]["file_path"] == "f.py"

    def test_empty_content(self):
        chunks = chunk_code("", "f.py")
        assert chunks == []
