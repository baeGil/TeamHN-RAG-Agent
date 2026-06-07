"""MinerU-based PDF parser.

MinerU converts PDFs to structured markdown using a hybrid pipeline:
  - Layout detection (PP-DocLayoutV2)
  - VLM-based text recognition (Qwen2-VL / MLX on Apple Silicon)
  - OCR fallback (PaddleOCR)
  - Math formula recognition (UniMERNet)

The CLI output includes:
  - <name>.md          — clean structured markdown (primary output used here)
  - doc_content_list_v2.json — typed block list with page numbers

MINERU_PARSE env var: "off" | "on" | "auto"
  - "off":  skip MinerU (default)
  - "on":   always use MinerU, raise on failure
  - "auto": try MinerU, fall back to PyMuPDF+VLM on any error

MINERU_CMD env var: path to the mineru binary.
  Auto-detection order:
    1. MINERU_CMD env var (if set)
    2. <project_root>/.venv_parser/bin/mineru
    3. "mineru" on system PATH
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .block import Block

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENV_PARSER_MINERU = _PROJECT_ROOT / ".venv_parser" / "bin" / "mineru"


def _resolve_mineru_cmd(configured: str = "") -> str:
    if configured.strip():
        return configured.strip()
    if _VENV_PARSER_MINERU.exists():
        return str(_VENV_PARSER_MINERU)
    if shutil.which("mineru"):
        return "mineru"
    raise FileNotFoundError(
        "MinerU binary not found. Options:\n"
        "  1. Set MINERU_CMD=/path/to/mineru in .env\n"
        "  2. Install MinerU in .venv_parser: pip install mineru[all]\n"
        "  3. Install MinerU globally: pip install mineru[all]"
    )


def parse_pdf_mineru(
    pdf_path: Path,
    mineru_cmd: str = "",
    output_dir: Optional[Path] = None,
) -> tuple[list[Block], dict[str, Any]]:
    """Parse a PDF with MinerU CLI and return (blocks, metadata).

    Args:
        pdf_path: Path to the PDF file.
        mineru_cmd: Override MinerU binary path (empty = auto-detect).
        output_dir: If provided, MinerU output is written here (not cleaned up).
                    If None, a temp directory is used and cleaned up after parsing.

    Returns:
        (blocks, meta) where blocks carry page, section, text.
    """
    cmd = _resolve_mineru_cmd(mineru_cmd)
    pdf_path = Path(pdf_path)
    meta: dict[str, Any] = {"parser": "mineru", "file": pdf_path.name}

    use_tmp = output_dir is None
    tmp_ctx = tempfile.TemporaryDirectory() if use_tmp else None

    try:
        out_dir = Path(tmp_ctx.name) if use_tmp else output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[MinerU] Parsing %s → %s", pdf_path.name, out_dir)
        result = subprocess.run(
            [cmd, "-p", str(pdf_path), "-o", str(out_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU exited with code {result.returncode}:\n{result.stderr[-2000:]}"
            )

        # MinerU writes to: <out_dir>/<stem>/auto/<stem>.md
        # or: <out_dir>/<stem>/hybrid_auto/<stem>.md
        # Walk to find the .md file
        md_files = sorted(out_dir.rglob("*.md"))
        if not md_files:
            raise RuntimeError(
                f"MinerU produced no .md file under {out_dir}. stderr:\n{result.stderr[-2000:]}"
            )
        md_path = md_files[0]
        md_text = md_path.read_text(encoding="utf-8")

        # Try to load structured content list for page numbers
        content_list: list[dict] = []
        for cand in ("doc_content_list_v2.json", "doc_content_list.json"):
            cl_path = md_path.parent / cand
            if cl_path.exists():
                try:
                    content_list = json.loads(cl_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
                break

        from .markdown_chunker import parse_mineru_markdown
        blocks = parse_mineru_markdown(md_text, content_list=content_list)

        meta["n_blocks"] = len(blocks)
        meta["md_path"] = str(md_path)
        logger.info("[MinerU] Done: %d blocks from %s", len(blocks), pdf_path.name)
        return blocks, meta

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
