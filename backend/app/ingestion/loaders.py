"""Document loaders: PDF, URL, plain text. Each returns (title, [(page, section, text)])."""
import re
from pathlib import Path
from typing import Optional

from ..config import get_settings
from . import pdf_extract
from .vn_text import normalize_structure

Block = tuple[Optional[int], Optional[str], str]


def load_pdf(
    data: bytes, filename: str, cache_dir: Optional[Path] = None
) -> tuple[str, list[Block]]:
    """Local math-aware extraction (PyMuPDF); optional VLM fallback for scanned
    pages (VLM_PARSE=auto) or all pages (VLM_PARSE=on)."""
    settings = get_settings()
    mode = settings.vlm_parse

    blocks: list[Block] = []
    empty: list[int] = []
    if mode != "on":
        blocks, empty = pdf_extract.extract_pdf_pages(data)

    vlm_pages: list[int] = []
    if mode == "on":
        with pdf_extract.fitz.open(stream=data, filetype="pdf") as doc:
            vlm_pages = list(range(1, len(doc) + 1))
    elif mode == "auto" and empty:
        vlm_pages = empty

    if vlm_pages and settings.has_openai:
        from . import pdf_vlm

        cache_path = (cache_dir / "parse_cache.db") if cache_dir else None
        vlm_blocks = pdf_vlm.transcribe_pages(
            data, vlm_pages, cache_path, settings.vlm_model
        )
        blocks = sorted(blocks + vlm_blocks, key=lambda b: (b[0] or 0))

    if not blocks:
        raise ValueError("Không trích xuất được nội dung từ PDF.")
    title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return title, blocks


def load_url(url: str) -> tuple[str, list[Block]]:
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Không tải được nội dung từ URL: {url}")
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        output_format="markdown",
    )
    if not extracted:
        raise ValueError(f"Không trích xuất được văn bản từ URL: {url}")
    meta = trafilatura.extract_metadata(downloaded)
    title = (meta.title if meta and meta.title else url)
    blocks: list[Block] = [(None, None, normalize_structure(extracted))]
    return title, blocks


def load_text(text: str, title: str = "Văn bản") -> tuple[str, list[Block]]:
    text = normalize_structure(text)
    if not text:
        raise ValueError("Văn bản rỗng.")
    return title, [(None, None, text)]
