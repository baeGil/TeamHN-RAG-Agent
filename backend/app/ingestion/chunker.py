"""Sentence-aware recursive chunker with overlap. Preserves page/section metadata."""
import re
from dataclasses import dataclass
from typing import Optional

from .loaders import Block

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!;:])\s+|\n{2,}")


@dataclass
class Chunk:
    text: str
    page: Optional[int]
    section: Optional[str]


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        s = sent
        # hard-split very long sentences
        while len(s) > max_chars:
            head, s = s[:max_chars], s[max_chars:]
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            chunks.append(head)
        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            # build overlap tail
            tail, tlen = [], 0
            for prev in reversed(cur):
                if tlen + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                tlen += len(prev) + 1
            cur, cur_len = tail[:], tlen
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]


def chunk_blocks(
    blocks: list[Block], max_chars: int = 900, overlap_chars: int = 150
) -> list[Chunk]:
    out: list[Chunk] = []
    for page, section, text in blocks:
        for piece in _pack(_split_sentences(text), max_chars, overlap_chars):
            out.append(Chunk(text=piece, page=page, section=section))
    return out
