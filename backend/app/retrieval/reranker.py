"""Reranker abstraction with local (FlagEmbedding) and API (Jina) backends.

Default: Jina API (fast, multilingual, no local GPU needed).
Fallback: BAAI/bge-reranker-v2-m3 local (no API cost, but slow on CPU).

Config:
  RERANKER_TYPE = "jina" | "local" (default: "jina" if JINA_API_KEY set, else "local")
  RERANKER_MODEL = model name (e.g. "jina-reranker-v3" or "BAAI/bge-reranker-v2-m3")
  JINA_API_KEY = API key for Jina reranker
"""
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger("rag.flow")


class LocalReranker:
    """Cross-encoder reranker using FlagEmbedding (BAAI/bge-reranker-v2-m3).

    Local, no token cost, but slow on CPU (~2-4s per 15 pairs).
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = False
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is None and not self._failed:
                started = time.perf_counter()
                try:
                    from FlagEmbedding import FlagReranker

                    self._model = FlagReranker(self.model_name, use_fp16=True)
                    logger.info(
                        "RAG_FLOW inference_load node=retrieve component=reranker "
                        "backend=local model=%s duration_ms=%.1f",
                        self.model_name,
                        (time.perf_counter() - started) * 1000,
                    )
                except Exception:
                    logger.exception(
                        "RAG_FLOW inference_load_error node=retrieve component=reranker "
                        "backend=local model=%s duration_ms=%.1f",
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
        model = self._load()
        if model is None or not candidates:
            return None
        pairs = [[query, text] for _, text in candidates]
        with self._infer_lock:
            started = time.perf_counter()
            scores = model.compute_score(pairs, normalize=True)
            logger.info(
                "RAG_FLOW inference node=retrieve component=reranker "
                "backend=local model=%s duration_ms=%.1f pairs=%s",
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


class JinaReranker:
    """Jina Reranker API (jina-reranker-v3). Fast, multilingual, API-based.

    API: POST https://api.jina.ai/v1/rerank
    Typically ~200-500ms for 15 pairs (vs 2-4s local).
    Supports 100+ languages including Vietnamese.
    """

    def __init__(self, model_name: str = "jina-reranker-v3", api_key: str = "") -> None:
        self.model_name = model_name
        self.api_key = api_key
        self._failed = False

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self._failed

    def rerank(
        self, query: str, candidates: list[tuple[int, str]], top_k: int
    ) -> Optional[list[tuple[int, float]]]:
        if not self.api_key or not candidates:
            return None
        import httpx

        documents = [text for _, text in candidates]
        payload = {
            "model": self.model_name,
            "query": query,
            "top_n": min(top_k, len(candidates)),
            "documents": documents,
            "return_documents": False,
        }
        started = time.perf_counter()
        try:
            resp = httpx.post(
                "https://api.jina.ai/v1/rerank",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "RAG_FLOW inference node=retrieve component=reranker "
                "backend=jina model=%s duration_ms=%.1f pairs=%s",
                self.model_name,
                duration_ms,
                len(candidates),
            )
            ranked = []
            for r in results:
                idx = r.get("index", 0)
                score = r.get("relevance_score", 0.0)
                cid = candidates[idx][0]
                ranked.append((int(cid), float(score)))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "RAG_FLOW inference_error node=retrieve component=reranker "
                "backend=jina model=%s duration_ms=%.1f pairs=%s",
                self.model_name,
                duration_ms,
                len(candidates),
            )
            self._failed = True
            return None


def create_reranker(
    reranker_type: str = "auto",
    model_name: str = "",
    jina_api_key: str = "",
) -> LocalReranker | JinaReranker:
    """Factory: pick reranker backend based on config.

    reranker_type="auto": use Jina if API key present, else local.
    reranker_type="jina": always use Jina API.
    reranker_type="local": always use local FlagEmbedding.
    """
    if reranker_type == "jina" or (
        reranker_type == "auto" and jina_api_key
    ):
        model = model_name if model_name and not model_name.startswith("BAAI") else "jina-reranker-v3"
        logger.info("RAG_FLOW reranker_init backend=jina model=%s", model)
        return JinaReranker(model_name=model, api_key=jina_api_key)
    else:
        model = model_name or "BAAI/bge-reranker-v2-m3"
        logger.info("RAG_FLOW reranker_init backend=local model=%s", model)
        return LocalReranker(model_name=model)


# Backward-compatible alias
Reranker = LocalReranker
