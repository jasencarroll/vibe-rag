"""Tests for vibe_rag.chunking module."""
from __future__ import annotations

from pathlib import Path

from vibe_rag.chunking import (
    chunk_markdown,
    chunk_plain_text,
    chunk_doc,
    collect_files,
    _should_include_file,
)


class TestChunkMarkdown:
    def test_single_short_section(self):
        text = "## Intro\n\nHello world."
        chunks = chunk_markdown(text, "readme.md")
        assert len(chunks) == 1
        assert chunks[0]["file_path"] == "readme.md"
        assert chunks[0]["chunk_index"] == 0
        assert "Hello world" in chunks[0]["content"]

    def test_multiple_sections(self):
        text = "## A\n\nFirst.\n\n## B\n\nSecond."
        chunks = chunk_markdown(text, "doc.md")
        assert len(chunks) == 2
        assert "First" in chunks[0]["content"]
        assert "Second" in chunks[1]["content"]

    def test_long_section_splits_by_paragraph(self):
        # Create a section longer than 2000 chars
        para = "word " * 300  # ~1500 chars per paragraph
        text = f"## Big\n\n{para}\n\n{para}"
        chunks = chunk_markdown(text, "big.md")
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_markdown("", "empty.md")
        assert chunks == []

    def test_no_headers(self):
        text = "Just plain text without headers."
        chunks = chunk_markdown(text, "plain.md")
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        text = "## A\n\nOne.\n\n## B\n\nTwo.\n\n## C\n\nThree."
        chunks = chunk_markdown(text, "f.md")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestChunkPlainText:
    def test_short_text_single_chunk(self):
        text = "Hello world."
        chunks = chunk_plain_text(text, "file.txt")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello world."

    def test_long_text_multiple_chunks_with_overlap(self):
        text = "a" * 4000
        chunks = chunk_plain_text(text, "long.txt")
        assert len(chunks) >= 2
        # First chunk is 2000 chars
        assert len(chunks[0]["content"]) == 2000
        # Overlap: second chunk starts at 1800
        assert chunks[1]["content"][:200] == chunks[0]["content"][-200:]

    def test_empty_text(self):
        chunks = chunk_plain_text("", "empty.txt")
        assert chunks == []

    def test_exactly_2000_chars(self):
        text = "x" * 2000
        chunks = chunk_plain_text(text, "exact.txt")
        assert len(chunks) == 1


class TestChunkDoc:
    def test_markdown_extension_uses_markdown_chunker(self):
        text = "## Section\n\nContent here."
        chunks = chunk_doc(text, "readme.md")
        assert len(chunks) >= 1
        assert "Content here" in chunks[0]["content"]

    def test_txt_extension_uses_plain_text_chunker(self):
        text = "Plain text content."
        chunks = chunk_doc(text, "notes.txt")
        assert len(chunks) == 1

    def test_unknown_extension_uses_plain_text(self):
        chunks = chunk_doc("data", "file.rst")
        assert len(chunks) == 1


class TestCollectFiles:
    def test_collects_code_and_doc_files(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print(1)")
        (tmp_path / "readme.md").write_text("# Hi")
        code, docs = collect_files([tmp_path])
        code_names = {f.name for f in code}
        doc_names = {f.name for f in docs}
        assert "main.py" in code_names
        assert "readme.md" in doc_names

    def test_skips_node_modules(self, tmp_path: Path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("module.exports = 1;")
        code, docs = collect_files([tmp_path])
        assert len(code) == 0

    def test_skips_git_dir(self, tmp_path: Path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config.json").write_text("{}")
        code, _ = collect_files([tmp_path])
        assert len(code) == 0

    def test_empty_directory(self, tmp_path: Path):
        code, docs = collect_files([tmp_path])
        assert code == []
        assert docs == []

    def test_multiple_roots(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "one.py").write_text("x = 1")
        (b / "two.py").write_text("y = 2")
        code, _ = collect_files([a, b])
        names = {f.name for f in code}
        assert names == {"one.py", "two.py"}


class TestShouldIncludeFile:
    def test_normal_file(self, tmp_path: Path):
        f = tmp_path / "ok.py"
        f.write_text("x = 1")
        assert _should_include_file(f) is True

    def test_too_large(self, tmp_path: Path):
        f = tmp_path / "big.py"
        f.write_text("x" * 200_000)
        assert _should_include_file(f) is False

    def test_directory_excluded(self, tmp_path: Path):
        assert _should_include_file(tmp_path) is False

    def test_skip_dir_in_path(self, tmp_path: Path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        f = venv / "lib.py"
        f.write_text("import os")
        assert _should_include_file(f) is False

    def test_symlink_rejected(self, tmp_path: Path):
        target = tmp_path / "real.py"
        target.write_text("x = 1\n")
        link = tmp_path / "link.py"
        link.symlink_to(target)
        assert _should_include_file(link) is False
