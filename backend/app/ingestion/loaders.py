"""Document loaders: PDF, URL, plain text. Each returns (title, [(page, section, text)])."""
import io
import re
from typing import Optional

from .vn_text import normalize

Block = tuple[Optional[int], Optional[str], str]


def load_pdf(data: bytes, filename: str) -> tuple[str, list[Block]]:
    import pdfplumber

    blocks: list[Block] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = normalize(text)
            if text:
                blocks.append((i, None, text))
    title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return title, blocks


def load_url(url: str) -> tuple[str, list[Block]]:
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Không tải được nội dung từ URL: {url}")
    extracted = trafilatura.extract(
        downloaded, include_comments=False, include_tables=True, favor_recall=True
    )
    if not extracted:
        raise ValueError(f"Không trích xuất được văn bản từ URL: {url}")
    meta = trafilatura.extract_metadata(downloaded)
    title = (meta.title if meta and meta.title else url)
    blocks: list[Block] = [(None, None, normalize(extracted))]
    return title, blocks


def load_text(text: str, title: str = "Văn bản") -> tuple[str, list[Block]]:
    text = normalize(text)
    if not text:
        raise ValueError("Văn bản rỗng.")
    return title, [(None, None, text)]
