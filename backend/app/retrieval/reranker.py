"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3). Local, no token cost.

Loaded lazily and degrades gracefully if the model/weights are unavailable.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("rag.flow")


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
                started = time.perf_counter()
                try:
                    # Monkey-patch prepare_for_model on tokenizer base class if it is missing (compat with transformers v5+)
                    import transformers
                    if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "prepare_for_model"):
                        def custom_prepare_for_model(self_tok, ids, pair_ids=None, add_special_tokens=True, padding=False, truncation=None, max_length=None, **kwargs):
                            bos = self_tok.bos_token_id if self_tok.bos_token_id is not None else 0
                            eos = self_tok.eos_token_id if self_tok.eos_token_id is not None else 2
                            if pair_ids is not None:
                                if max_length is not None:
                                    max_pair_len = max_length - len(ids) - 4
                                    pair_ids = pair_ids[:max(0, max_pair_len)]
                                input_ids = [bos] + ids + [eos, eos] + pair_ids + [eos]
                            else:
                                if max_length is not None:
                                    ids = ids[:max(0, max_length - 2)]
                                input_ids = [bos] + ids + [eos]
                            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
                        transformers.tokenization_utils_base.PreTrainedTokenizerBase.prepare_for_model = custom_prepare_for_model

                    from FlagEmbedding import FlagReranker
                    import torch

                    if not torch.cuda.is_available():
                        import os
                        threads = int(os.getenv("TORCH_NUM_THREADS", "8"))
                        torch.set_num_threads(threads)

                    self._model = FlagReranker(self.model_name, use_fp16=torch.cuda.is_available())
                    logger.info(
                        "RAG_FLOW inference_load node=retrieve component=reranker model=%s duration_ms=%.1f",
                        self.model_name,
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
