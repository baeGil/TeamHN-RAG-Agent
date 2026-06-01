"""Document loaders: PDF, URL, plain text. Each returns (title, [Block, ...])."""
import logging
import re
from pathlib import Path
from typing import Optional

from ..config import get_settings
from . import pdf_extract
from .block import Block
from .vn_text import normalize_structure

logger = logging.getLogger(__name__)


def load_pdf(
    data: bytes, filename: str, cache_dir: Optional[Path] = None
) -> tuple[str, list[Block]]:
    """Load PDF using the configured parser.

    Parser selection (via REDUCTO_PARSE env var):

    - "off":     PyMuPDF + optional VLM fallback (local, free)
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
        logger.warning("[Reducto] REDUCTO_PARSE=%s but no API key, falling back to local",
                        settings.reducto_parse)

    # --- Local PyMuPDF + VLM path ---
    return _load_pdf_local(data, filename, cache_dir, settings)


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
    blocks: list[Block] = [Block(text=normalize_structure(extracted))]
    return title, blocks


def load_text(text: str, title: str = "Văn bản") -> tuple[str, list[Block]]:
    text = normalize_structure(text)
    if not text:
        raise ValueError("Văn bản rỗng.")
    return title, [Block(text=text)]
