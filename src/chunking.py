from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+|(?<=\.)\n+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        if sep == "":
            splits = list(current_text)
        else:
            splits = current_text.split(sep)

        if len(splits) <= 1:
            return self._split(current_text, next_seps)

        chunks: list[str] = []
        buffer = ""
        for piece in splits:
            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split(piece, next_seps))
            else:
                candidate = (buffer + sep + piece) if buffer else piece
                if len(candidate) <= self.chunk_size:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = piece
        if buffer:
            chunks.append(buffer)
        return [c for c in chunks if c]


class HeadingChunker:
    """
    Split Markdown text into chunks by heading boundaries.

    Each chunk starts at a Markdown heading line (``#``, ``##``, ``###``, …)
    and contains all content until the next heading of the same or higher
    level.  The heading itself is preserved as the first line of the chunk
    so that the chunk remains self-contained (e.g. "## 2. Thời gian phản
    hồi\\n…").

    If the document has no headings, the whole text is returned as a single
    chunk.

    Parameters
    ----------
    min_heading_level : int
        Only treat heading lines whose ``#``-depth is ≤ this value as split
        points.  Default 3 means ``#``, ``##``, and ``###`` all trigger a
        split; ``####`` and deeper are kept inside the current chunk.
    """

    def __init__(self, min_heading_level: int = 3) -> None:
        self.min_heading_level = max(1, min_heading_level)
        self._heading_re = re.compile(
            r"^(#{1," + str(self.min_heading_level) + r"})\s+.+", re.MULTILINE
        )

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        lines = text.splitlines(keepends=True)
        chunks: list[str] = []
        current: list[str] = []

        for line in lines:
            if self._heading_re.match(line.rstrip("\n\r")):
                # flush the previous section
                section = "".join(current).strip()
                if section:
                    chunks.append(section)
                current = [line]
            else:
                current.append(line)

        # flush the last section
        section = "".join(current).strip()
        if section:
            chunks.append(section)

        # If no headings were found, return the whole text as one chunk
        if not chunks:
            return [text.strip()] if text.strip() else []

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(y * y for y in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
            "by_headings": HeadingChunker(min_heading_level=3).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {
                "chunks": chunks,
                "count": count,
                "avg_length": avg_length,
            }
        return result
