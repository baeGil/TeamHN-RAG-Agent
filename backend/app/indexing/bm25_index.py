"""BM25 keyword index over Vietnamese word-segmented tokens."""
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from ..ingestion.vn_text import tokenize


class BM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[int] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = False

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    def add(self, chunk_id: int, text: str) -> None:
        self._chunk_ids.append(chunk_id)
        self._corpus_tokens.append(tokenize(text))
        self._dirty = True

    def remove(self, ids: set[int]) -> None:
        keep = [
            (cid, toks)
            for cid, toks in zip(self._chunk_ids, self._corpus_tokens)
            if cid not in ids
        ]
        self._chunk_ids = [c for c, _ in keep]
        self._corpus_tokens = [t for _, t in keep]
        self._dirty = True

    def _rebuild(self) -> None:
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None
        self._dirty = False

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._dirty:
            self._rebuild()
        if self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(int(cid), float(sc)) for cid, sc in ranked[:k] if sc > 0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "corpus_tokens": self._corpus_tokens}, f
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._chunk_ids = data["chunk_ids"]
        obj._corpus_tokens = data["corpus_tokens"]
        obj._dirty = True
        return obj
