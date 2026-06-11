"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3). Local, no token cost.

Loaded lazily and degrades gracefully if the model/weights are unavailable.
"""
import os
import logging
import threading
import time
from typing import Optional

from ..config import get_settings

logger = logging.getLogger("rag.flow")


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.settings = get_settings()
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
                started = time.perf_counter()
                try:
                    os.environ.setdefault("USE_TF", "0")
                    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                    from FlagEmbedding import FlagReranker

                    device = self._resolve_device()
                    self._model = FlagReranker(
                        self.model_name,
                        use_fp16=device != "cpu",
                        devices=device,
                        batch_size=self.settings.reranker_batch_size,
                    )
                    logger.info(
                        "RAG_FLOW inference_load node=retrieve component=reranker model=%s device=%s batch_size=%s duration_ms=%.1f",
                        self.model_name,
                        device,
                        self.settings.reranker_batch_size,
                        (time.perf_counter() - started) * 1000,
                    )
                except Exception:
                    logger.exception(
                        "RAG_FLOW inference_load_error node=retrieve component=reranker model=%s duration_ms=%.1f",
                        self.model_name,
                        (time.perf_counter() - started) * 1000,
                    )
                    self._failed = True
                    self._model = None
        return self._model

    def _resolve_device(self) -> str:
        configured = (self.settings.reranker_device or "auto").strip().lower()
        if configured and configured != "auto":
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

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
            started = time.perf_counter()
            scores = model.compute_score(pairs, normalize=True)
            logger.info(
                "RAG_FLOW inference node=retrieve component=reranker model=%s duration_ms=%.1f pairs=%s",
                self.model_name,
                (time.perf_counter() - started) * 1000,
                len(pairs),
            )
        if not isinstance(scores, list):
            scores = [scores]
        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(int(cid), float(sc)) for cid, sc in ranked[:top_k]]
