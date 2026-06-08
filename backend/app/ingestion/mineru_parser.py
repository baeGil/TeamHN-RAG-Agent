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
    2. <backend>/.venv/bin/mineru  (same venv as backend)
    3. "mineru" on system PATH

Device selection: CUDA → MPS → CPU (auto).
  Override with MINERU_DEVICE_MODE env var if needed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .block import Block

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_VENV_MINERU = _BACKEND_ROOT / ".venv" / "bin" / "mineru"


def _detect_device() -> str:
    """Auto-detect best available device: cuda → mps → cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _resolve_mineru_cmd(configured: str = "") -> str:
    if configured.strip():
        return configured.strip()
    if _VENV_MINERU.exists():
        return str(_VENV_MINERU)
    if shutil.which("mineru"):
        return "mineru"
    raise FileNotFoundError(
        "MinerU binary not found. Options:\n"
        "  1. Set MINERU_CMD=/path/to/mineru in .env\n"
        "  2. Install MinerU: uv pip install 'mineru[pipeline]'\n"
        "  3. Install MinerU globally: pip install 'mineru[pipeline]'"
    )


def _resolve_backend(device: str) -> list[str]:
    """Choose MinerU CLI backend flag based on device.

    - cuda / mps: use hybrid-auto-engine (local VLM + layout model)
    - cpu: use pipeline (CPU-only, no MLX/VLM deps needed)
    """
    if device in ("cuda", "mps"):
        return ["-b", "hybrid-auto-engine"]
    # CPU: pipeline backend is most compatible
    return ["-b", "pipeline"]


def _try_mineru(cmd: str, pdf_path: Path, out_dir: Path, env: dict,
                backend_flags: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run MinerU with given backend, return CompletedProcess."""
    cmd_args = [cmd, "-p", str(pdf_path), "-o", str(out_dir)] + backend_flags
    logger.info("[MinerU] Running: %s", " ".join(cmd_args))
    return subprocess.run(cmd_args, capture_output=True, text=True, timeout=timeout, env=env)


def parse_pdf_mineru(
    pdf_path: Path,
    mineru_cmd: str = "",
    output_dir: Optional[Path] = None,
) -> tuple[list[Block], dict[str, Any]]:
    """Parse a PDF with MinerU CLI and return (blocks, metadata).

    Device selection: CUDA → MPS → CPU (auto). Override with MINERU_DEVICE_MODE.
    Backend fallback: hybrid-auto-engine → pipeline (if hybrid fails/timeout).

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

    device = os.environ.get("MINERU_DEVICE_MODE") or _detect_device()
    primary_backend = _resolve_backend(device)
    # Fallback: pipeline (CPU-only) always works if torch is installed
    fallback_backend = ["-b", "pipeline"]
    logger.info("[MinerU] device=%s primary_backend=%s", device, " ".join(primary_backend))

    use_tmp = output_dir is None
    tmp_ctx = tempfile.TemporaryDirectory() if use_tmp else None

    try:
        out_dir = Path(tmp_ctx.name) if use_tmp else output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "MINERU_DEVICE_MODE": device}

        # Try primary backend first (e.g. hybrid-auto-engine for GPU/MPS)
        result = None
        used_backend = primary_backend
        try:
            result = _try_mineru(cmd, pdf_path, out_dir, env, primary_backend)
        except subprocess.TimeoutExpired:
            logger.warning("[MinerU] primary backend timed out, falling back to pipeline")
        except Exception as exc:
            logger.warning("[MinerU] primary backend failed (%s), falling back to pipeline", exc)

        # Fallback to pipeline if primary failed
        if result is None or result.returncode != 0:
            if primary_backend != fallback_backend:
                logger.info("[MinerU] Trying fallback backend: pipeline")
                # Clean output dir for fresh attempt
                import shutil
                for child in out_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                used_backend = fallback_backend
                env["MINERU_DEVICE_MODE"] = "cpu"
                try:
                    result = _try_mineru(cmd, pdf_path, out_dir, env, fallback_backend)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("MinerU pipeline backend also timed out")
            else:
                # Primary was already pipeline and it failed
                if result is not None and result.returncode != 0:
                    raise RuntimeError(
                        f"MinerU exited with code {result.returncode}:\n{result.stderr[-2000:]}"
                    )
        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU exited with code {result.returncode}:\n{result.stderr[-2000:]}"
            )

        # MinerU writes to: <out_dir>/<stem>/auto/<stem>.md
        # or: <out_dir>/<stem>/hybrid_auto/<stem>.md
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
        meta["device"] = device
        logger.info("[MinerU] Done: %d blocks from %s (device=%s)", len(blocks), pdf_path.name, device)
        return blocks, meta

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
