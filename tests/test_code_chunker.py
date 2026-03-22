from vibe_memory.indexing.code_chunker import chunk_code, chunk_code_sliding_window


def test_sliding_window_basic():
    lines = [f"line {i}" for i in range(120)]
    content = "\n".join(lines)
    chunks = chunk_code_sliding_window(content, "test.py", window=60, overlap=10)
    assert len(chunks) == 2
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 60
    assert chunks[1]["start_line"] == 51
    assert chunks[1]["end_line"] == 120


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
