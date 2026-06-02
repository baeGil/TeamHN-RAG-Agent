"""Reducto-based PDF parser.

Modes (controlled by REDUCTO_PARSE env var):
  - "off":     Skip Reducto entirely, use PyMuPDF+VLM pipeline (free, local)
  - "default": Reducto default parse — 1 credit/page, good quality
  - "agentic":  Reducto agentic (text+table VLM review) — 2 credits/page, best quality
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from reducto import Reducto
from reducto.types import EnhanceParam, FormattingParam, RetrievalParam

from .block import Block

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared config builder
# ---------------------------------------------------------------------------


def _build_params(
    mode: str = "default",
    chunk_mode: str = "page_sections",
    chunk_size: int = 1200,
    filter_blocks: list[str] | None = None,
    table_format: str = "dynamic",
) -> dict[str, Any]:
    """Build Reducto API params.

    mode="default": agentic=[{scope: table, mode: auto}] — cheap, good quality
    mode="agentic": agentic=[{scope: text}, {scope: table, mode: auto}] — best quality
    """
    if filter_blocks is None:
        filter_blocks = ["Header", "Footer", "Page Number"]

    if mode == "agentic":
        agentic = [
            {"scope": "text"},
            {"scope": "table", "mode": "auto"},
        ]
    else:
        agentic = [{"scope": "table", "mode": "auto"}]

    return {
        "enhance": EnhanceParam(agentic=agentic, summarize_figures=True),
        "retrieval": RetrievalParam(
            chunking={"chunk_mode": chunk_mode, "chunk_size": chunk_size},
            embedding_optimized=True,
            filter_blocks=filter_blocks,
        ),
        "formatting": FormattingParam(
            table_output_format=table_format,
            add_page_markers=True,
        ),
    }


# ---------------------------------------------------------------------------
# Convert Reducto result to backend Blocks
# ---------------------------------------------------------------------------


def _extract_section_from_blocks(blocks: list[Any]) -> str | None:
    """Find the first Section Header or Title block in a chunk."""
    for block in blocks:
        if block.type in ("Section Header", "Title"):
            return block.content.strip() if block.content else None
    return None


def reducto_result_to_blocks(result: Any) -> list[Block]:
    """Convert a Reducto parse result to the backend Block format.

    Returns ``list[Block]`` where each Block carries:
      - page, section, text (full content)
      - embed_text (Reducto's optimized embedding-ready summary, or None)
    """
    blocks_out: list[Block] = []

    for chunk in result.result.chunks:
        if not chunk.blocks:
            continue

        pages = set()
        for block in chunk.blocks:
            if block.bbox:
                pg = getattr(block.bbox, "original_page", None) or block.bbox.page
                if pg is not None:
                    pages.add(pg)
        primary_page = min(pages) if pages else None
        section = _extract_section_from_blocks(chunk.blocks)
        content = chunk.content or ""
        embed = chunk.embed or None
        if not content.strip():
            continue

        if "[[START OF PAGE" in content:
            import re
            parts = re.split(r"\[\[START OF PAGE (\d+)\]\]\n?", content)
            i = 1
            while i < len(parts) - 1:
                pg_num = int(parts[i])
                pg_content = parts[i + 1].strip()
                if pg_content:
                    blocks_out.append(Block(page=pg_num, section=section, text=pg_content,
                                            embed_text=embed))
                i += 2
        else:
            blocks_out.append(Block(page=primary_page, section=section, text=content,
                                   embed_text=embed))

    return blocks_out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_pdf_reducto(
    pdf_path: str | Path,
    api_key: str,
    mode: str = "default",
    chunk_mode: str = "page_sections",
    chunk_size: int = 1200,
    filter_blocks: list[str] | None = None,
    table_format: str = "dynamic",
) -> tuple[list[Block], dict[str, Any]]:
    """Parse a PDF using Reducto.

    Args:
        pdf_path: Path to PDF file.
        api_key: Reducto API key.
        mode: "default" (cheap, good) or "agentic" (expensive, best).
        chunk_mode: Chunking mode for Reducto.
        chunk_size: Target chunk size in characters.
        filter_blocks: Block types to filter out of content.
        table_format: "dynamic", "html", "md", "json", or "csv".

    Returns:
        (blocks, metadata) where blocks is ``list[(page, section, text)]``
        and metadata contains parsing stats.
    """
    client = Reducto(api_key=api_key)
    pdf_path = Path(pdf_path)
    meta: dict[str, Any] = {
        "parser": "reducto",
        "mode": mode,
        "file": pdf_path.name,
    }

    logger.info("[Reducto] Uploading %s (mode=%s) ...", pdf_path.name, mode)
    t0 = time.time()
    upload = client.upload(file=pdf_path)
    file_id = upload.file_id
    upload_elapsed = time.time() - t0
    logger.info("[Reducto] Uploaded in %.1fs: %s", upload_elapsed, file_id)

    params = _build_params(
        mode=mode,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        filter_blocks=filter_blocks,
        table_format=table_format,
    )

    logger.info("[Reducto] Parsing (mode=%s) ...", mode)
    t1 = time.time()
    result = client.parse.run(input=file_id, **params)
    parse_elapsed = time.time() - t1

    blocks = reducto_result_to_blocks(result)
    total_elapsed = time.time() - t0

    meta["total_pages"] = result.usage.num_pages
    meta["total_credits"] = result.usage.credits
    meta["total_elapsed"] = round(total_elapsed, 2)
    meta["used_agentic"] = mode == "agentic"
    meta["n_blocks"] = len(blocks)
    meta["job_id"] = result.job_id
    meta["studio_link"] = result.studio_link

    logger.info(
        "[Reducto] Done: %.1fs, %d pages, %.0f credits, %d blocks, agentic=%s",
        total_elapsed, result.usage.num_pages, result.usage.credits,
        len(blocks), mode == "agentic",
    )

    return blocks, meta