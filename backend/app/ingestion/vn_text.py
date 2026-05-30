"""Vietnamese text utilities for BM25 tokenization (underthesea word segmentation)."""
import re
import unicodedata
from functools import lru_cache

_word_tokenize = None

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
        from underthesea import word_tokenize  # lazy import (heavy)

        _word_tokenize = word_tokenize
    return _word_tokenize


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    return tuple(tokens)


def tokenize(text: str) -> list[str]:
    """Word-segment Vietnamese text and return BM25-ready lowercase tokens."""
    text = normalize(text)
    if not text:
        return []
    return list(_tok_cached(text))
