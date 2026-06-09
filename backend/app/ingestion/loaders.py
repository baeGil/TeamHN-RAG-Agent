"""Document loaders: PDF, URL, plain text. Each returns (title, [Block, ...])."""
import gzip
import logging
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from ..config import get_settings
from . import pdf_extract
from .block import Block
from .vn_text import normalize_structure

logger = logging.getLogger(__name__)

_URL_TIMEOUT_SECONDS = 20
_MAX_URL_DOWNLOAD_BYTES = 80 * 1024 * 1024
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)


@dataclass
class _FetchedUrl:
    data: bytes
    text: str
    final_url: str
    content_type: str


def load_pdf(
    data: bytes, filename: str, cache_dir: Optional[Path] = None
) -> tuple[str, list[Block]]:
    """Load PDF using the configured parser.

    Parser selection (via REDUCTO_PARSE env var):

    - "off":     markitdown (fast, local) with PyMuPDF fallback
    - "default": Reducto default parse (1 credit/page, good quality)
    - "agentic": Reducto agentic parse (2 credits/page, best quality)
    """
    settings = get_settings()

    # --- Reducto path ---
    if settings.reducto_parse not in ("off",) and settings.reducto_api_key:
        try:
            return _load_pdf_reducto(data, filename, settings)
        except Exception as exc:
            logger.warning("[Reducto] parse failed (%s), falling back to local", exc)
    elif settings.reducto_parse not in ("off",) and not settings.reducto_api_key:
        logger.warning("[Reducto] REDUCTO_PARSE=%s but no API key, falling back to markitdown",
                        settings.reducto_parse)

    # --- markitdown fast path (primary local parser) ---
    try:
        return _load_pdf_markitdown(data, filename)
    except Exception as exc:
        logger.warning("[markitdown] parse failed (%s), raising exception", exc)
        raise exc


def _load_pdf_markitdown(
    data: bytes, filename: str
) -> tuple[str, list[Block]]:
    """Fast local PDF parsing via microsoft/markitdown → Markdown text."""
    import tempfile

    from markitdown import MarkItDown

    md = MarkItDown()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = md.convert(str(tmp_path))
        text = (result.text_content or "").strip()
    finally:
        tmp_path.unlink(missing_ok=True)

    if not text:
        raise ValueError("markitdown returned empty content")

    from .vn_text import normalize_structure
    text = normalize_structure(text)

    # Split into page-sized blocks (~3000-char chunks) so the embedder
    # doesn't receive one massive string.
    _BLOCK_CHARS = 3000
    raw_blocks: list[Block] = []
    for i in range(0, len(text), _BLOCK_CHARS):
        chunk = text[i : i + _BLOCK_CHARS].strip()
        if chunk:
            raw_blocks.append(Block(text=chunk))

    if not raw_blocks:
        raise ValueError("markitdown: no blocks after chunking")

    title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    logger.info("[markitdown] Parsed %s → %d blocks", filename, len(raw_blocks))
    return title, raw_blocks


def _load_pdf_local(
    data: bytes, filename: str, cache_dir: Optional[Path], settings
) -> tuple[str, list[Block]]:
    """Local math-aware extraction (PyMuPDF); optional VLM fallback for scanned
    pages (VLM_PARSE=auto) or all pages (VLM_PARSE=on)."""
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
        blocks = sorted(blocks + vlm_blocks, key=lambda b: (b.page or 0))

    if not blocks:
        raise ValueError("Không trích xuất được nội dung từ PDF.")
    title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return title, blocks


def _load_pdf_reducto(
    data: bytes, filename: str, settings
) -> tuple[str, list[Block]]:
    """Parse PDF via Reducto API (default or agentic mode)."""
    import tempfile

    from .reducto_parser import parse_pdf_reducto

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        blocks, meta = parse_pdf_reducto(
            pdf_path=tmp_path,
            api_key=settings.reducto_api_key,
            mode=settings.reducto_parse,
            chunk_mode=settings.reducto_chunk_mode,
            chunk_size=settings.reducto_chunk_size,
            filter_blocks=settings.reducto_filter_blocks,
            table_format=settings.reducto_table_format,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    for b in blocks:
        b.text = normalize_structure(b.text)
        if b.embed_text:
            b.embed_text = normalize_structure(b.embed_text)

    logger.info(
        "[Reducto] Parsed %s: %d blocks, %.1fs, %.0f credits, agentic=%s",
        filename, len(blocks),
        meta.get("total_elapsed", 0),
        meta.get("total_credits", 0),
        meta.get("used_agentic", False),
    )

    title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return title, blocks


def load_url(url: str) -> tuple[str, list[Block]]:
    import trafilatura

    clean_url = _validate_url(url)
    fetched = _fetch_url(clean_url)

    if fetched.data and _looks_like_pdf(fetched):
        filename = _url_filename(fetched.final_url, default="document.pdf")
        return load_pdf(fetched.data, filename)

    html = fetched.text
    if not html:
        html = trafilatura.fetch_url(clean_url) or ""
    if not html:
        raise ValueError(f"Không tải được nội dung từ URL: {url}")

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        output_format="markdown",
    )
    if not extracted:
        extracted = _visible_text(html)
    if not extracted:
        raise ValueError(f"Không trích xuất được văn bản từ URL: {url}")

    meta = trafilatura.extract_metadata(html)
    title = meta.title if meta and meta.title else _html_title(html, fetched.final_url)
    blocks: list[Block] = [Block(text=normalize_structure(extracted))]
    return title, blocks


def _validate_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL phải bắt đầu bằng http:// hoặc https://.")
    return url


def _fetch_url(url: str) -> _FetchedUrl:
    req = Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/pdf,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with urlopen(req, timeout=_URL_TIMEOUT_SECONDS) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_URL_DOWNLOAD_BYTES:
                raise ValueError("URL content is too large to ingest.")

            raw = resp.read(_MAX_URL_DOWNLOAD_BYTES + 1)
            if len(raw) > _MAX_URL_DOWNLOAD_BYTES:
                raise ValueError("URL content is too large to ingest.")

            data = _decompress_response(raw, resp.headers.get("Content-Encoding"))
            content_type = resp.headers.get("Content-Type", "")
            text = (
                _decode_response(data, content_type)
                if _is_text_response(content_type, data)
                else ""
            )
            return _FetchedUrl(
                data=data,
                text=text,
                final_url=resp.geturl(),
                content_type=content_type,
            )
    except Exception:
        return _FetchedUrl(data=b"", text="", final_url=url, content_type="")


def _decompress_response(data: bytes, encoding: Optional[str]) -> bytes:
    encoding = (encoding or "").lower()
    if "gzip" in encoding:
        return gzip.decompress(data)
    if "deflate" in encoding:
        try:
            return zlib.decompress(data)
        except zlib.error:
            return zlib.decompress(data, -zlib.MAX_WBITS)
    return data


def _is_text_response(content_type: str, data: bytes) -> bool:
    ctype = content_type.split(";", 1)[0].strip().lower()
    if ctype.startswith("text/"):
        return True
    if ctype in {"application/xhtml+xml", "application/xml", "application/json"}:
        return True
    sample = data[:256].lstrip().lower()
    return sample.startswith((b"<!doctype html", b"<html", b"{", b"["))


def _decode_response(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    if match:
        charset = match.group(1)
    return data.decode(charset, errors="replace")


def _looks_like_pdf(fetched: _FetchedUrl) -> bool:
    ctype = fetched.content_type.split(";", 1)[0].strip().lower()
    path = urlparse(fetched.final_url).path.lower()
    return (
        ctype == "application/pdf"
        or fetched.data.startswith(b"%PDF")
        or path.endswith(".pdf")
    )


def _url_filename(url: str, default: str = "document") -> str:
    name = Path(unquote(urlparse(url).path)).name.strip()
    return name or default


def _html_title(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.get_text(strip=True):
        return normalize_structure(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return normalize_structure(h1.get_text(" ", strip=True))
    return fallback


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(
        ("script", "style", "noscript", "nav", "aside", "footer", "header")
    ):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    parts: list[str] = []
    for tag in root.find_all(
        ("h1", "h2", "h3", "p", "li", "blockquote", "td", "th", "pre")
    ):
        text = tag.get_text("\n" if tag.name == "pre" else " ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if text:
            parts.append(text)
    if not parts:
        parts.append(root.get_text("\n", strip=True))
    return normalize_structure("\n\n".join(parts))


def load_text(text: str, title: str = "Văn bản") -> tuple[str, list[Block]]:
    text = normalize_structure(text)
    if not text:
        raise ValueError("Văn bản rỗng.")
    return title, [Block(text=text)]
