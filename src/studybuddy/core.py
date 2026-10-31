"""Text chunking primitives for retrieval."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chunk_text(
    doc_id: str,
    title: str,
    text: str,
    size: int = 180,
    overlap: int = 35,
) -> list[Chunk]:
    if not text.strip():
        raise ValueError("text cannot be empty")
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    words = re.findall(r"\S+", text)
    step = size - overlap
    return [
        Chunk(
            f"{doc_id}:{start // step}",
            doc_id,
            title,
            " ".join(words[start : start + size]),
        )
        for start in range(0, len(words), step)
    ]
