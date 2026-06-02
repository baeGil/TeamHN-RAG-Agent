"""Optional VLM fallback: transcribe PDF pages to Markdown + LaTeX.

Used only for pages where local extraction fails (scanned/image pages) or when
VLM_PARSE=on. Results are cached on disk by image hash so re-ingest / eval
re-runs cost nothing.
"""
import base64
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

import fitz

from .vn_text import normalize_structure

from .block import Block

_SYSTEM = (
    "Bạn là công cụ trích xuất tài liệu học thuật tiếng Việt sang Markdown. "
    "Chuyển ảnh trang tài liệu thành văn bản trung thực, giữ nguyên bố cục."
)
_INSTRUCTION = (
    "Chuyển trang này thành Markdown:\n"
    "- Giữ nguyên toàn bộ văn bản tiếng Việt và thứ tự đọc.\n"
    "- MỌI công thức toán phải viết bằng LaTeX: nội dòng $...$, tách dòng $$...$$.\n"
    "- Tiêu đề mục dùng #, ##; bảng dùng cú pháp bảng Markdown.\n"
    "- KHÔNG thêm giải thích, KHÔNG bịa nội dung. Chỉ trả về Markdown của trang."
)


class _Cache:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS vlm (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT v FROM vlm WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, val: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO vlm(k, v) VALUES (?,?)", (key, val))
        self.conn.commit()


def transcribe_pages(
    data: bytes, page_numbers: list[int], cache_path: Optional[Path], model: str, dpi: int = 200
) -> list[Block]:
    """Transcribe the given 1-based page numbers; returns Blocks (page, None, text)."""
    if not page_numbers:
        return []
    from ..agent.llm import LLM

    llm = LLM()
    cache = _Cache(cache_path) if cache_path is not None else None
    out: list[Block] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for pno in page_numbers:
            if pno < 1 or pno > len(doc):
                continue
            pix = doc[pno - 1].get_pixmap(dpi=dpi)
            png = pix.tobytes("png")
            b64 = base64.b64encode(png).decode("ascii")
            key = hashlib.sha256((model + ":").encode() + png).hexdigest()
            md = cache.get(key) if cache else None
            if md is None:
                md = llm.chat_vision(_SYSTEM, _INSTRUCTION, b64)
                if cache:
                    cache.put(key, md)
            md = normalize_structure(md)
            if md.strip():
                out.append(Block(page=pno, section=None, text=md))
    return out
