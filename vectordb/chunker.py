"""Split a long note into overlapping windows.

A single embedding of a whole article gets pulled toward every topic in
it. Smaller windows stay specific, and the overlap keeps a sentence that
sits on a boundary intact inside one of the windows.
"""

from __future__ import annotations


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Return character windows of `size` that share `overlap` characters.

    The input is stripped once. Windows themselves are not stripped, so the
    overlap between neighbors stays exact.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += step
    return chunks
