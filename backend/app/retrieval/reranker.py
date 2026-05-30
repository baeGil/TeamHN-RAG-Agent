"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3). Local, no token cost.

Loaded lazily and degrades gracefully if the model/weights are unavailable.
"""
import threading
from typing import Optional


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = False
        self._lock = threading.Lock()
        # The HF tokenizer / model forward is not thread-safe; the agent retrieves
        # sub-questions in parallel, so reranker calls must be serialized.
        self._infer_lock = threading.Lock()

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from FlagEmbedding import FlagReranker

                    self._model = FlagReranker(self.model_name, use_fp16=True)
                except Exception:
                    self._failed = True
                    self._model = None
        return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    def rerank(
        self, query: str, candidates: list[tuple[int, str]], top_k: int
    ) -> Optional[list[tuple[int, float]]]:
        """Return [(chunk_id, score)] sorted desc, or None if reranker unavailable."""
        model = self._load()
        if model is None or not candidates:
            return None
        pairs = [[query, text] for _, text in candidates]
        with self._infer_lock:
            scores = model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(int(cid), float(sc)) for cid, sc in ranked[:top_k]]
