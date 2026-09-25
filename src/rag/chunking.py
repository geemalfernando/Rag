"""Split text into overlapping chunks, preferring paragraph and sentence boundaries."""

import re

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    pieces = []
    for part in text.split(sep):
        part = part + sep if sep.strip() == "." else part
        pieces.extend(_split(part, size, rest) if len(part) > size else [part])
    return pieces


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for piece in _split(text, size, _SEPARATORS):
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        # Carry the tail of the previous chunk forward so context isn't lost at the seam.
        tail = current[-overlap:] if overlap else ""
        if tail and " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        current = f"{tail}\n{piece}" if tail and len(tail) + len(piece) + 1 <= size else piece
    if current:
        chunks.append(current)
    return chunks
