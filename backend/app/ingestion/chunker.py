"""Section-aware chunker with sentence packing, overlap and atomic formulas.

- Chunks never cross a section boundary (each block already carries its section).
- Formula lines and table rows are kept as ATOMIC units: they are never split
  mid-expression, so retrieved chunks contain whole formulas (correct citations
  and grounded answers).
- Prose is sentence-split and packed up to max_chars with sentence-level overlap.
"""
import re
from dataclasses import dataclass
from typing import Optional

from .loaders import Block

_SENT_SPLIT = re.compile(
    r"(?<=[\?\!])\s+"
    r"|(?<=[\.\;])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴĐÉÈẺẼÊỀẾỆỂỄÍÌỊỈĨÓÒỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠƯỜỨỰỬỮÚÙỤỦŨƯỪỨỰỬỮÝỲỴỶỸ])"
    r"|\n{2,}"
)

# A line is treated as a formula (atomic) when it carries math notation.
_MATH_SYMBOLS = "∑√∏∫≤≥≠⇒⇐→←±·×÷∈∀∅∩∪⊆⊇≡∇∂"
_FORMULA_RE = re.compile(
    r"[" + _MATH_SYMBOLS + r"]"
    r"|\)/\("
    r"|\$"
)


@dataclass
class Chunk:
    text: str
    page: Optional[int]
    section: Optional[str]


_ABBREV_RE = re.compile(
    r"\b[A-ZĐÀÁẠẢÃ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]*\.\s"
)


def _protect_abbreviations(text: str) -> str:
    return _ABBREV_RE.sub(lambda m: m.group().replace(". ", ".\x00"), text)


def _restore_abbreviations(text: str) -> str:
    return text.replace("\x00", " ")


def _split_sentences(text: str) -> list[str]:
    protected = _protect_abbreviations(text)
    parts = [p.strip() for p in _SENT_SPLIT.split(protected) if p and p.strip()]
    return [_restore_abbreviations(p) for p in parts] or ([text.strip()] if text.strip() else [])


def _is_atomic_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("[BẢNG]") or " | " in s:
        return True
    return bool(_FORMULA_RE.search(s))


def _split_units(text: str) -> list[tuple[str, bool]]:
    """Split block text into (unit, is_atomic) preserving formulas/tables whole."""
    units: list[tuple[str, bool]] = []
    prose: list[str] = []
    atomic: list[str] = []

    def _flush_prose() -> None:
        if prose:
            joined = " ".join(prose)
            for sent in _split_sentences(joined):
                units.append((sent, False))
            prose.clear()

    def _flush_atomic() -> None:
        if atomic:
            units.append(("\n".join(atomic), True))
            atomic.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            _flush_prose()
            _flush_atomic()
            continue
        if _is_atomic_line(line):
            _flush_prose()
            atomic.append(line)
        else:
            _flush_atomic()
            prose.append(line)
    _flush_prose()
    _flush_atomic()
    return units


def _overlap_tail(cur: list[str], overlap_chars: int) -> tuple[list[str], int]:
    tail: list[str] = []
    tlen = 0
    for prev in reversed(cur):
        if tlen + len(prev) > overlap_chars:
            break
        tail.insert(0, prev)
        tlen += len(prev) + 1
    return tail, tlen


def _smart_split_long(text: str, max_chars: int) -> list[str]:
    """Split a very long prose sentence at logical boundaries (comma, semicolon, paren)."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    last_boundary = 0
    i = 0
    while i < len(text):
        if text[i] in ",;)" and i - start >= max_chars * 0.4:
            last_boundary = i + 1
        if i - start + 1 >= max_chars:
            if last_boundary > start:
                parts.append(text[start:last_boundary].strip())
                start = last_boundary
                last_boundary = start
            else:
                parts.append(text[start:start + max_chars].strip())
                start += max_chars
        i += 1
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            parts.append(remaining)
    return [p for p in parts if p]


def _pack(units: list[tuple[str, bool]], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def _flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(" ".join(cur))
            cur, cur_len = _overlap_tail(cur, overlap_chars)

    for text, atomic in units:
        if atomic:
            # Keep whole; start a fresh chunk if it won't fit alongside current.
            if cur and cur_len + len(text) + 1 > max_chars:
                _flush()
            cur.append(text)
            cur_len += len(text) + 1
            continue
        if len(text) > max_chars:
            _flush()
            for frag in _smart_split_long(text, max_chars):
                if cur and cur_len + len(frag) + 1 > max_chars:
                    _flush()
                cur.append(frag)
                cur_len += len(frag) + 1
            continue
        if cur and cur_len + len(text) + 1 > max_chars:
            _flush()
        cur.append(text)
        cur_len += len(text) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]


def chunk_blocks(
    blocks: list[Block], max_chars: int = 1000, overlap_chars: int = 200
) -> list[Chunk]:
    out: list[Chunk] = []
    for page, section, text in blocks:
        for piece in _pack(_split_units(text), max_chars, overlap_chars):
            out.append(Chunk(text=piece, page=page, section=section))
    return out
