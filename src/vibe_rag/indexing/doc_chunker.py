from __future__ import annotations
import re

MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 200


def chunk_markdown(text: str) -> list[str]:
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]
    chunks: list[str] = []
    for section in sections:
        if len(section) <= MAX_CHUNK_CHARS:
            chunks.append(section)
        else:
            paragraphs = re.split(r"\n\n+", section)
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and current:
                    chunks.append(current.strip())
                    current = current[-OVERLAP_CHARS:] + "\n\n" + para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append(current.strip())
    return chunks


def chunk_plain_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += MAX_CHUNK_CHARS - OVERLAP_CHARS
    return chunks
