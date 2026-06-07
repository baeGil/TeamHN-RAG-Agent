"""Section-aware chunker for MinerU markdown output.

MinerU produces clean structured markdown with:
  - Heading lines: ## Section, ### Subsection (no leading # for H1)
  - HTML tables: <table>...</table>
  - LaTeX math blocks: $$...$$
  - Regular paragraphs

Strategy:
  1. Detect section boundaries from markdown headings → Block.section
  2. Atomic units: <table> blocks, $$ math blocks (never split)
  3. Prose within a section is sentence-packed up to max_chars with NO overlap
     (overlap-free is required for Relevant Segment Extraction)
  4. Page numbers come from content_list JSON when available.

Design notes:
  - No overlap between adjacent chunks (unlike the standard chunker.py).
    RSE re-assembles contiguous context windows at query time, so overlap
    would create duplicate content inside a segment.
  - The chunker reuses the sentence splitter from chunker.py to stay consistent.
"""
from __future__ import annotations

import re
from typing import Optional

from .block import Block
from .chunker import _split_sentences, _smart_split_long

# Heading patterns in MinerU markdown
_H2_RE = re.compile(r"^#{1,4}\s+(.+)$")
_TABLE_RE = re.compile(r"<table[\s>]", re.IGNORECASE)
_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)
_MATH_BLOCK_START = re.compile(r"^\$\$")
_MATH_BLOCK_END = re.compile(r"\$\$$")

# Page marker inserted by some MinerU configs:  <!-- Page 3 -->  or  \n---\n
_PAGE_MARKER_RE = re.compile(r"<!--\s*[Pp]age\s+(\d+)\s*-->")

# Default max chars per chunk (can be overridden per call)
_DEFAULT_MAX_CHARS = 1000


def _extract_page_map(content_list: list) -> dict[str, int]:
    """Build a rough map of section title → page number from content_list JSON.

    Supports two formats from MinerU:
      - doc_content_list.json: flat list of dicts with {type, text/content, page_idx}
      - doc_content_list_v2.json: list of pages, each page is a list of block dicts
    """
    page_map: dict[str, int] = {}

    def _process_item(item: dict, page_no: Optional[int]) -> None:
        tp = item.get("type", "")
        if tp not in ("title", "section_title", "heading"):
            return
        # v2: content is a dict with title_content list
        content = item.get("content", "")
        if isinstance(content, dict):
            parts = content.get("title_content") or []
            text = " ".join(p.get("content", "") for p in parts if isinstance(p, dict))
        elif isinstance(content, str):
            text = content
        else:
            text = item.get("text", "")
        text = text.strip()
        if not text:
            return
        # Page number: from explicit field or caller-supplied index
        pg = item.get("page_idx") or item.get("page") or page_no
        if pg is not None:
            page_map[text.lower()] = int(pg) + 1  # 0-based → 1-based

    for page_idx, page_or_item in enumerate(content_list):
        if isinstance(page_or_item, list):
            # v2 format: each entry is a page (list of blocks)
            for block in page_or_item:
                if isinstance(block, dict):
                    _process_item(block, page_idx)
        elif isinstance(page_or_item, dict):
            # flat format
            _process_item(page_or_item, None)

    return page_map


def _section_page(section: Optional[str], page_map: dict[str, int]) -> Optional[int]:
    if not section:
        return None
    return page_map.get(section.lower())


def _pack_sentences(
    sentences: list[str],
    max_chars: int,
    current_page: Optional[int],
    section: Optional[str],
) -> list[Block]:
    """Pack sentences into blocks of at most max_chars, no overlap."""
    blocks: list[Block] = []
    buf: list[str] = []
    buf_len = 0

    def _flush() -> None:
        if buf:
            blocks.append(Block(page=current_page, section=section, text=" ".join(buf)))
            buf.clear()

    for sent in sentences:
        if len(sent) > max_chars:
            _flush()
            for frag in _smart_split_long(sent, max_chars):
                if buf_len + len(frag) + 1 > max_chars and buf:
                    _flush()
                    buf_len = 0
                buf.append(frag)
                buf_len += len(frag) + 1
            continue
        if buf and buf_len + len(sent) + 1 > max_chars:
            _flush()
            buf_len = 0
        buf.append(sent)
        buf_len += len(sent) + 1

    _flush()
    return blocks


def parse_mineru_markdown(
    md_text: str,
    content_list: Optional[list[dict]] = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[Block]:
    """Parse MinerU markdown output into a list of Blocks.

    Each Block carries (page, section, text).  Atomic units (tables, math
    display blocks) are kept whole.  Prose is sentence-packed up to max_chars
    with NO overlap between adjacent blocks.

    Args:
        md_text: Full text of the .md file produced by MinerU.
        content_list: Parsed doc_content_list_v2.json (optional, for page numbers).
        max_chars: Maximum characters per prose block.

    Returns:
        list[Block] ordered as they appear in the document.
    """
    page_map = _extract_page_map(content_list or [])
    lines = md_text.splitlines()

    blocks: list[Block] = []
    current_section: Optional[str] = None
    current_page: Optional[int] = None

    # Buffer for prose lines
    prose_lines: list[str] = []

    # State for multi-line atomic blocks
    in_table = False
    table_buf: list[str] = []
    in_math = False
    math_buf: list[str] = []

    def _flush_prose() -> None:
        if not prose_lines:
            return
        joined = " ".join(line.strip() for line in prose_lines if line.strip())
        if joined:
            sentences = _split_sentences(joined)
            blocks.extend(
                _pack_sentences(sentences, max_chars, current_page, current_section)
            )
        prose_lines.clear()

    def _flush_atomic(text: str) -> None:
        if text.strip():
            blocks.append(Block(page=current_page, section=current_section, text=text))

    for line in lines:
        # Detect page markers (some MinerU configs emit these)
        pm = _PAGE_MARKER_RE.search(line)
        if pm:
            current_page = int(pm.group(1))
            continue

        # --- Inside a multi-line table ---
        if in_table:
            table_buf.append(line)
            if _TABLE_END_RE.search(line):
                _flush_prose()
                _flush_atomic("\n".join(table_buf))
                table_buf.clear()
                in_table = False
            continue

        # --- Inside a display math block ---
        if in_math:
            math_buf.append(line)
            stripped = line.strip()
            # End of math block: a line that ends with $$ but is NOT the opening line
            if len(math_buf) > 1 and _MATH_BLOCK_END.search(stripped):
                _flush_prose()
                _flush_atomic("\n".join(math_buf))
                math_buf.clear()
                in_math = False
            continue

        stripped = line.strip()

        # --- Heading line → new section ---
        m = _H2_RE.match(stripped)
        if m:
            _flush_prose()
            current_section = m.group(1).strip()
            page_from_map = _section_page(current_section, page_map)
            if page_from_map is not None:
                current_page = page_from_map
            continue

        # --- Start of HTML table ---
        if _TABLE_RE.search(stripped):
            in_table = True
            table_buf = [line]
            # Single-line table?
            if _TABLE_END_RE.search(stripped):
                _flush_prose()
                _flush_atomic(stripped)
                table_buf.clear()
                in_table = False
            continue

        # --- Display math block ($$ ... $$) ---
        if _MATH_BLOCK_START.match(stripped):
            single_line = stripped.count("$$") >= 2 and len(stripped) > 2
            if single_line:
                _flush_prose()
                _flush_atomic(stripped)
            else:
                in_math = True
                math_buf = [line]
            continue

        # --- Empty line → paragraph break ---
        if not stripped:
            _flush_prose()
            continue

        # --- Regular prose ---
        prose_lines.append(stripped)

    # Flush anything remaining
    _flush_prose()
    if table_buf:
        _flush_atomic("\n".join(table_buf))
    if math_buf:
        _flush_atomic("\n".join(math_buf))

    return [b for b in blocks if b.text.strip()]
