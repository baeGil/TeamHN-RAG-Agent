"""Pairwise contradiction detection for reranked RAG context.

The notebook prototype used Hugging Face Inference API with
FacebookAI/roberta-large-mnli. This module keeps that behavior optional and
lazy so retrieval still works when the token/model/API is unavailable.
"""
import logging
import threading
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

logger = logging.getLogger("rag.flow")


@dataclass
class ConflictPair:
    pair_index: int
    conflict: bool
    raw_label: str
    confidence: float
    chunk_a_label: int
    chunk_b_label: int
    chunk_a_id: int
    chunk_b_id: int
    chunk_a_title: str
    chunk_b_title: str
    chunk_a_page: Optional[int]
    chunk_b_page: Optional[int]
    text_a_preview: str
    text_b_preview: str


class ConflictDetector:
    def __init__(
        self,
        enabled: bool,
        model_name: str,
        api_key: str,
        min_confidence: float = 0.75,
        max_pairs: int = 45,
        max_chars: int = 180,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.api_key = api_key
        self.min_confidence = min_confidence
        self.max_pairs = max_pairs
        self.max_chars = max_chars
        self._client = None
        self._failed = False
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        msg = str(exc)
        if "expanded size of the tensor" in msg or "must match the existing size" in msg:
            return "input_too_long"
        if "401" in msg or "Unauthorized" in msg:
            return "hf_unauthorized"
        if "429" in msg or "Too Many Requests" in msg:
            return "hf_rate_limited"
        if "Bad request" in msg:
            return "hf_bad_request"
        return "hf_inference_error"

    def _load(self):
        if not self.enabled or not self.api_key or self._failed:
            return None
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None or self._failed:
                return self._client
            started = time.perf_counter()
            try:
                from huggingface_hub import InferenceClient

                try:
                    self._client = InferenceClient(provider="hf-inference", api_key=self.api_key)
                except TypeError:
                    self._client = InferenceClient(provider="hf-inference", token=self.api_key)
                logger.info(
                    "RAG_FLOW inference_load node=conflict component=nli model=%s duration_ms=%.1f",
                    self.model_name,
                    (time.perf_counter() - started) * 1000,
                )
            except Exception:
                logger.exception(
                    "RAG_FLOW inference_load_error node=conflict component=nli model=%s duration_ms=%.1f",
                    self.model_name,
                    (time.perf_counter() - started) * 1000,
                )
                self._failed = True
                self._client = None
        return self._client

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _translate_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, str]:
        import json
        try:
            from ..agent.llm import LLM
            llm = LLM()
        except Exception as e:
            logger.warning("Could not initialize LLM for conflict translation: %s", e)
            return {}

        unique_texts = list(set(str(c.get("text", "")) for c in chunks))
        if not unique_texts:
            return {}
            
        prompt = (
            "You are a professional translator. Translate the following list of Vietnamese texts into English. "
            "Keep the exact same meaning, numbers, and context. "
            "Format the output as a JSON object with a single key 'translations' containing a list of translated strings in the exact same order.\n\n"
            f"Texts to translate:\n{json.dumps(unique_texts, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ]
        try:
            res = llm.chat_json(messages, fast=True, node="conflict_translate")
            translations = res.get("translations", [])
            if len(translations) == len(unique_texts):
                return dict(zip(unique_texts, translations))
        except Exception as e:
            logger.warning("Failed to translate chunks for NLI conflict check: %s", e)
        return {}

    def check(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "available": False, "conflicts": [], "checked_pairs": 0}
        client = self._load()
        if client is None:
            reason = "missing_hf_token" if not self.api_key else "client_unavailable"
            return {
                "enabled": True,
                "available": False,
                "reason": reason,
                "conflicts": [],
                "checked_pairs": 0,
            }
        if len(chunks) < 2:
            return {"enabled": True, "available": True, "conflicts": [], "checked_pairs": 0}

        started = time.perf_counter()
        try:
            # Try to translate texts to English for the English NLI model
            translations = self._translate_chunks(chunks)
            chunks_en = []
            for c in chunks:
                c_en = dict(c)
                orig_text = str(c.get("text", ""))
                c_en["text"] = translations.get(orig_text, orig_text)
                chunks_en.append(c_en)

            pairs_en = list(combinations(chunks_en, 2))[: self.max_pairs]
            pairs_orig = list(combinations(chunks, 2))[: self.max_pairs]

            raw = None
            used_chars = self.max_chars
            retry_budgets = [self.max_chars, 120, 80]
            for budget in retry_budgets:
                batch_inputs = [
                    f"{self._text(c1_en, budget)} </s></s> {self._text(c2_en, budget)}"
                    for c1_en, c2_en in pairs_en
                ]
                try:
                    with self._infer_lock:
                        raw = client.text_classification(text=batch_inputs, model=self.model_name)
                    used_chars = budget
                    break
                except Exception as e:
                    if self._friendly_error(e) == "input_too_long" and budget != retry_budgets[-1]:
                        logger.warning(
                            "RAG_FLOW inference_retry node=conflict component=nli model=%s reason=input_too_long chars=%s",
                            self.model_name,
                            budget,
                        )
                        continue
                    raise
            responses = self._normalize_responses(raw, len(pairs_orig))
            conflicts: list[ConflictPair] = []
            for idx, (c1_orig, c2_orig) in enumerate(pairs_orig):
                pred = self._best_prediction(responses[idx] if idx < len(responses) else None)
                if not pred:
                    continue
                label = str(pred.get("label", ""))
                score = float(pred.get("score", 0.0) or 0.0)
                is_conflict = label.upper() == "CONTRADICTION" and score >= self.min_confidence
                if not is_conflict:
                    continue
                conflicts.append(
                    ConflictPair(
                        pair_index=idx + 1,
                        conflict=True,
                        raw_label=label,
                        confidence=score,
                        chunk_a_label=int(c1_orig.get("label", 0) or 0),
                        chunk_b_label=int(c2_orig.get("label", 0) or 0),
                        chunk_a_id=int(c1_orig.get("chunk_id", 0) or 0),
                        chunk_b_id=int(c2_orig.get("chunk_id", 0) or 0),
                        chunk_a_title=str(c1_orig.get("doc_title", "")),
                        chunk_b_title=str(c2_orig.get("doc_title", "")),
                        chunk_a_page=c1_orig.get("page"),
                        chunk_b_page=c2_orig.get("page"),
                        text_a_preview=self._preview(c1_orig),
                        text_b_preview=self._preview(c2_orig),
                    )
                )
            logger.info(
                "RAG_FLOW inference node=conflict component=nli model=%s duration_ms=%.1f pairs=%s conflicts=%s",
                self.model_name,
                (time.perf_counter() - started) * 1000,
                len(pairs_orig),
                len(conflicts),
            )
            return {
                "enabled": True,
                "available": True,
                "model": self.model_name,
                "checked_pairs": len(pairs_orig),
                "input_chars": used_chars,
                "conflicts": [c.__dict__ for c in conflicts],
            }
        except Exception as e:
            reason = self._friendly_error(e)
            logger.exception(
                "RAG_FLOW inference_error node=conflict component=nli model=%s duration_ms=%.1f",
                self.model_name,
                (time.perf_counter() - started) * 1000,
            )
            # Use chunks for exception return fallback
            checked_count = len(list(combinations(chunks, 2))[: self.max_pairs])
            return {
                "enabled": True,
                "available": False,
                "reason": reason,
                "conflicts": [],
                "checked_pairs": checked_count,
            }

    def _text(self, chunk: dict[str, Any], max_chars: Optional[int] = None) -> str:
        text = " ".join(str(chunk.get("text", "")).split())
        limit = max_chars or self.max_chars
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0]

    @staticmethod
    def _preview(chunk: dict[str, Any], limit: int = 240) -> str:
        text = " ".join(str(chunk.get("text", "")).split())
        return text[:limit]

    @staticmethod
    def _best_prediction(response: Any) -> Optional[dict[str, Any]]:
        if isinstance(response, dict):
            return response
        if isinstance(response, list) and response:
            first = response[0]
            return first if isinstance(first, dict) else None
        return None

    @staticmethod
    def _normalize_responses(raw: Any, expected: int) -> list[Any]:
        if expected == 1:
            if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
                return raw
            return [raw]
        if isinstance(raw, list):
            return raw
        return []
