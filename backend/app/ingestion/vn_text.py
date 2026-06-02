"""Vietnamese text utilities for BM25 tokenization (underthesea word segmentation)."""
import re
import threading
import unicodedata
from functools import lru_cache

_word_tokenize = None
_word_tokenize_lock = threading.Lock()

# A compact Vietnamese stopword list. Removing these sharpens BM25 keyword matching.
_STOPWORDS = {
    "và", "là", "của", "có", "các", "được", "cho", "trong", "với", "khi", "này",
    "đó", "một", "những", "đến", "từ", "theo", "để", "ra", "vào", "thì", "mà",
    "hay", "hoặc", "nếu", "vì", "do", "bởi", "nên", "cũng", "đã", "sẽ", "đang",
    "rồi", "tại", "trên", "dưới", "về", "như", "thế", "ở", "bằng", "lại", "nó",
    "rằng", "phải", "không", "chỉ", "còn", "đây", "kia", "ấy",
}


def _load():
    global _word_tokenize
    if _word_tokenize is None:
        with _word_tokenize_lock:
            if _word_tokenize is None:
                from underthesea import word_tokenize  # lazy import (heavy)
                # Force model loading on the main thread before any parallel calls.
                # underthesea's word_tokenize uses a module-level global (word_tokenize_model)
                # that is lazily initialized on first call. Multiple threads calling
                # word_tokenize simultaneously can race on this global, causing
                # 'NoneType' object has no attribute 'process' errors.
                word_tokenize("init", format="text")
                _word_tokenize = word_tokenize
    return _word_tokenize


_CID_RE = re.compile(r"\(cid:\d+\)")
# Large-operator glyphs from LaTeX math fonts that some extractors mis-map to ASCII.
_GLYPH_FIX = {"\uf8e6": "√"}


def clean_artifacts(text: str) -> str:
    """Remove extraction artifacts: (cid:NN) glyphs and known broken glyphs."""
    text = _CID_RE.sub(" ", text)
    for bad, good in _GLYPH_FIX.items():
        text = text.replace(bad, good)
    return text


def normalize(text: str) -> str:
    """Aggressive single-line normalization (used for BM25 tokenization)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_structure(text: str) -> str:
    """Structure-preserving normalization for chunking.

    Collapses runs of spaces/tabs but keeps newlines (so headings, paragraphs
    and multi-line formulas survive into the chunker).
    """
    text = unicodedata.normalize("NFC", text)
    text = clean_artifacts(text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln:
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _subword_parts(tok: str) -> list[str]:
    """Split an underscore/camelCase compound into unigram parts so a spaced query
    ("compute loss") matches an underscore term ("compute_loss"), and also matches
    underthesea compounds like "hàm_compute_loss" -> hàm, compute, loss."""
    parts: list[str] = []
    for piece in _CAMEL_SPLIT_RE.sub(" ", tok).split():
        parts.extend(p for p in piece.split("_") if len(p) > 1)
    return [p.lower() for p in parts]


@lru_cache(maxsize=4096)
def _tok_cached(text: str) -> tuple[str, ...]:
    seg = _load()(text, format="text")
    tokens = []
    for tok in seg.split():
        low = tok.lower()
        # underscore-joined compound -> keep as single token, also strip pure punctuation
        if re.fullmatch(r"[\W_]+", low) and "_" not in low:
            continue
        bare = low.replace("_", " ")
        if bare in _STOPWORDS:
            continue
        tokens.append(low)
        # Also emit subword tokens for compounds (underscore or camelCase) so spaced
        # queries match joined terms (e.g. "compute loss" -> "compute_loss").
        if "_" in low or _CAMEL_SPLIT_RE.search(tok):
            for part in _subword_parts(tok):
                if part not in _STOPWORDS and part != low:
                    tokens.append(part)
    return tuple(tokens)


def tokenize(text: str) -> list[str]:
    """Word-segment Vietnamese text and return BM25-ready lowercase tokens."""
    text = normalize(text)
    if not text:
        return []
    return list(_tok_cached(text))
