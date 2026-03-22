from vibe_rag.indexing.doc_chunker import chunk_markdown, chunk_plain_text


def test_chunk_markdown_splits_on_headers():
    md = """## Section 1\n\nSome content here.\n\n## Section 2\n\nMore content here.\n"""
    chunks = chunk_markdown(md)
    assert len(chunks) == 2
    assert "Section 1" in chunks[0]
    assert "Section 2" in chunks[1]


def test_chunk_markdown_subsplits_large_sections():
    large_section = "## Big Section\n\n" + ("word " * 600) + "\n\nAnother paragraph.\n"
    chunks = chunk_markdown(large_section)
    assert len(chunks) >= 2


def test_chunk_plain_text():
    text = "word " * 1000
    chunks = chunk_plain_text(text)
    assert len(chunks) >= 2
