"""Math-aware local PDF extraction with PyMuPDF (fitz).

Born-digital PDFs (LaTeX papers, theses) carry correct characters, but naive
text extraction mangles math. This module reconstructs readable text + formulas
deterministically, with no LLM cost:

- Font-aware operator remap: large operators (∑, √, ∏, ∫) live in math fonts
  (e.g. CMEX10) whose glyphs get mis-mapped to ASCII by extractors. We remap
  them by font, so a real "X" in prose is never touched.
- Geometric space reconstruction: some Vietnamese fonts drop the space glyph
  but leave a real horizontal gap; we re-insert spaces from char box gaps.
- Fraction reconstruction: fraction bars are vector drawings; we pair the spans
  above (numerator) and below (denominator) within the bar's x-range into a/b.
- Heading detection: numbered headings (e.g. "1.1 Title") become section paths.

Each page yields Blocks: (page, section, text). Returns ([], reason) when a page
has no extractable text so the caller can fall back to VLM transcription.
"""
import re
from typing import Optional

import fitz

from .block import Block
from .vn_text import clean_artifacts

# Glyphs from cmex10 (LaTeX large operators) commonly mis-decoded to ASCII.
_CMEX_REMAP = {"X": "∑", "P": "∑", "q": "√", "Y": "∏", "Z": "∫", "R": "∫"}
_MATH_OP_FONT_HINTS = ("CMEX", "CMMI", "CMSY", "MSAM", "MSBM")

_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s+(\S.*)$")
# Roman-numeral headings ("I.", "II.", "VI. Câu hỏi trắc nghiệm") and step/keyword
# headings ("Bước 4: ...", "Mục lục", "Phụ lục") used in Vietnamese course docs.
_ROMAN_HEADING_RE = re.compile(r"^([IVXLCDM]{1,5})\.\s*(\S.*)$")
_STEP_HEADING_RE = re.compile(r"^(Bước\s+\d+)\s*[:.\-]?\s*(\S.*)$", re.IGNORECASE)
_KEYWORD_HEADING_RE = re.compile(
    r"^(Mục\s*lục|Phụ\s*lục|Tài\s*liệu\s*tham\s*khảo|Kết\s*luận|Giới\s*thiệu|Tổng\s*kết)\b",
    re.IGNORECASE,
)
_MATH_HINT_RE = re.compile(r"[∑√∏∫≤≥≠⇒⇐→←±·×÷∈∀∅∩∪⊆≡]|\)/\(|/\{|\^\{|_\{|=")


def _heading_label(stripped: str) -> Optional[str]:
    """Return a section label if the line looks like a heading, else None.
    Supports numbered (1.2), Roman (I., VI.), step (Bước N:) and keyword headings."""
    m = _HEADING_RE.match(stripped)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()
    m = _ROMAN_HEADING_RE.match(stripped)
    if m:
        return f"{m.group(1)}. {m.group(2)}".strip()
    m = _STEP_HEADING_RE.match(stripped)
    if m:
        return f"{m.group(1)}: {m.group(2)}".strip()
    if _KEYWORD_HEADING_RE.match(stripped):
        return stripped
    return None


def _is_math_font(font: str) -> bool:
    fu = font.upper()
    return any(h in fu for h in _MATH_OP_FONT_HINTS)


def _remap_char(c: str, font: str) -> str:
    if _is_math_font(font):
        return _CMEX_REMAP.get(c, c)
    return c


def _collect_chars(page: "fitz.Page") -> list[dict]:
    chars: list[dict] = []
    raw = page.get_text("rawdict")
    for b in raw.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = span.get("size", 0.0) or 0.0
                for ch in span.get("chars", []):
                    x0, y0, x1, y1 = ch["bbox"]
                    chars.append(
                        {
                            "c": _remap_char(ch["c"], font),
                            "x0": x0,
                            "x1": x1,
                            "cx": (x0 + x1) / 2,
                            "cy": (y0 + y1) / 2,
                            "top": y0,
                            "bottom": y1,
                            "font": font,
                            "size": size,
                            "used": False,
                        }
                    )
    return chars


def _fraction_bars(page: "fitz.Page") -> list[tuple[float, float, float]]:
    """Short horizontal rules = fraction bars. Full-width rules (page/watermark
    borders) are excluded by width."""
    pw = page.rect.width
    max_w = pw * 0.6
    bars: list[tuple[float, float, float]] = []

    def _add(x0: float, x1: float, y: float) -> None:
        x0, x1 = sorted((x0, x1))
        if 14.0 <= (x1 - x0) <= max_w:
            bars.append((x0, x1, y))

    for d in page.get_drawings():
        for it in d.get("items", []):
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.8:
                    _add(p1.x, p2.x, (p1.y + p2.y) / 2)
            elif it[0] == "re":
                r = it[1]
                if r.height < 1.6:
                    _add(r.x0, r.x1, (r.y0 + r.y1) / 2)
    return bars


def _space_width(chars: list[dict]) -> float:
    widths = [c["x1"] - c["x0"] for c in chars if c["c"] == " " and c["x1"] > c["x0"]]
    if widths:
        widths.sort()
        return widths[len(widths) // 2]
    return 3.5


def _join_line(items: list[dict], space_w: float) -> str:
    items = sorted(items, key=lambda c: c["x0"])
    out: list[str] = []
    prev: Optional[dict] = None
    thresh = space_w * 0.6
    for c in items:
        ch = c["c"]
        if prev is not None and ch != " " and prev["c"] != " ":
            gap = c["x0"] - prev["x1"]
            if gap > thresh:
                out.append(" ")
        out.append(ch)
        prev = c
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


def _build_fractions(chars: list[dict], bars: list[tuple[float, float, float]]) -> list[dict]:
    """Consume numerator/denominator chars around each bar; return fraction tokens."""
    fracs: list[dict] = []
    for x0, x1, by in bars:
        pad = 2.0
        # Numerator/denominator rows sit within about half a line-height of the
        # vinculum. A wider band would swallow accents (overlines) and adjacent
        # text lines, so keep it tight.
        band = 11.0
        num = [
            c for c in chars
            if not c["used"] and (x0 - pad) <= c["cx"] <= (x1 + pad)
            and (by - band) < c["cy"] < (by - 0.3)
        ]
        den = [
            c for c in chars
            if not c["used"] and (x0 - pad) <= c["cx"] <= (x1 + pad)
            and (by + 0.3) < c["cy"] < (by + band)
        ]
        if not num or not den or len(num) > 40 or len(den) > 40:
            continue
        # Guard against false bars sitting inside a single text line: a real
        # fraction has its denominator row clearly below the numerator row.
        num_cy = sum(c["cy"] for c in num) / len(num)
        den_cy = sum(c["cy"] for c in den) / len(den)
        if (den_cy - num_cy) < 6.0:
            continue
        # Reject accents (overlines) misread as bars: a real numerator is
        # isolated, but text grabbed from a running line has a close neighbour
        # hugging its left/right edge on the same row.
        sw = _space_width(chars)
        n_left = min(c["x0"] for c in num)
        n_right = max(c["x1"] for c in num)
        embedded = any(
            not c["used"] and c not in num and abs(c["cy"] - num_cy) < 3.0
            and (0 <= (n_left - c["x1"]) < sw or 0 <= (c["x0"] - n_right) < sw)
            for c in chars
        )
        if embedded:
            continue
        for c in num + den:
            c["used"] = True
        anchor_x = min(c["x0"] for c in num)
        nt = _join_line(num, _space_width(chars))
        dt = _join_line(den, _space_width(chars))
        fracs.append(
            {
                "c": f"({nt})/({dt})",
                "x0": anchor_x,
                "x1": x1,
                "cx": anchor_x,
                "cy": by,
                "top": by,
                "font": "FRAC",
                "size": 12.0,
                "used": False,
                "atom": True,
            }
        )
    return fracs


def _cluster_rows(items: list[dict]) -> list[list[dict]]:
    items = sorted(items, key=lambda c: (round(c["cy"], 1), c["x0"]))
    rows: list[list[dict]] = []
    for c in items:
        placed = False
        for row in rows:
            if abs(row[0]["cy"] - c["cy"]) <= 4.0:
                row.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    rows.sort(key=lambda r: min(x["cy"] for x in r))
    return rows


def _median_font_size(chars: list[dict]) -> float:
    sizes = sorted(c.get("size", 0.0) for c in chars if c.get("size", 0.0) > 0)
    if not sizes:
        return 12.0
    return sizes[len(sizes) // 2]


def _page_lines(page: "fitz.Page") -> tuple[list[tuple[str, float]], float]:
    """Return ([(line_text, max_body_font_size)], median_font_size) in reading order."""
    chars = _collect_chars(page)
    if not chars:
        return [], 12.0
    bars = [] #_fraction_bars(page) - DISABLED TO PREVENT HANGS ON CAD VECTOR GRAPHICS
    fracs = _build_fractions(chars, bars)
    space_w = _space_width(chars)
    remaining = [c for c in chars if not c["used"]]
    rows = _cluster_rows(remaining + fracs)
    median_size = _median_font_size(chars)
    lines: list[tuple[str, float]] = []
    for row in rows:
        parts = sorted(row, key=lambda c: c["x0"])
        text_parts: list[str] = []
        prev: Optional[dict] = None
        max_size = 0.0
        for c in parts:
            if not c.get("atom"):
                max_size = max(max_size, c.get("size", 0.0))
            if c.get("atom"):
                if text_parts and not text_parts[-1].endswith(" "):
                    text_parts.append(" ")
                text_parts.append(c["c"])
                prev = None
                continue
            ch = c["c"]
            if prev is not None and ch != " " and prev["c"] != " ":
                if (c["x0"] - prev["x1"]) > space_w * 0.6:
                    text_parts.append(" ")
            text_parts.append(ch)
            prev = c
        line = re.sub(r"[ \t]+", " ", "".join(text_parts)).strip()
        if line:
            lines.append((line, max_size))
    return lines, median_size


def _split_sections(
    page_no: int, lines: list[tuple[str, float]], current: dict, median_size: float
) -> list[Block]:
    """Walk lines; larger-font numbered headings open a new section."""
    blocks: list[Block] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            body = "\n".join(buf).strip()
            if body:
                blocks.append(Block(page=page_no, section=current["section"], text=body))
            buf.clear()

    heading_min = max(median_size * 1.15, 11.0)
    for line, size in lines:
        stripped = line.strip()
        label = _heading_label(stripped)
        is_heading = (
            label is not None
            and size >= heading_min
            and len(stripped) <= 80
            and not _MATH_HINT_RE.search(stripped)
        )
        if is_heading:
            _flush()
            current["section"] = label
            continue
        buf.append(line)
    _flush()
    return blocks


def _boilerplate_lines(pages_lines: list[list[tuple[str, float]]]) -> set[str]:
    """Lines repeated (verbatim) on most pages are running headers/footers."""
    from collections import Counter

    n_pages = len(pages_lines)
    if n_pages < 3:
        return set()
    counts: Counter = Counter()
    for lines in pages_lines:
        for t in {t for t, _ in lines if 0 < len(t) <= 60}:
            counts[t] += 1
    threshold = max(3, int(n_pages * 0.5))
    return {t for t, c in counts.items() if c >= threshold}


def extract_pdf_pages(data: bytes) -> tuple[list[Block], list[int]]:
    """Return (blocks, empty_pages). empty_pages lists 1-based page numbers with
    no extractable text (candidates for VLM fallback)."""
    blocks: list[Block] = []
    empty: list[int] = []
    current = {"section": None}
    with fitz.open(stream=data, filetype="pdf") as doc:
        per_page: list[tuple[int, list[tuple[str, float]], float]] = []
        for i, page in enumerate(doc, start=1):
            lines, ms = _page_lines(page)
            cleaned = [(clean_artifacts(t).strip(), s) for t, s in lines]
            cleaned = [(t, s) for t, s in cleaned if t]
            per_page.append((i, cleaned, ms))

        boilerplate = _boilerplate_lines([c for _, c, _ in per_page])
        for i, cleaned, ms in per_page:
            cleaned = [(t, s) for t, s in cleaned if t not in boilerplate]
            if sum(len(t) for t, _ in cleaned) < 3:
                empty.append(i)
                continue
            blocks.extend(_split_sections(i, cleaned, current, ms))
    return blocks, empty
