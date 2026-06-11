import hashlib
import json
import logging
import threading
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

logger = logging.getLogger("rag.flow")


LABELS = ["no-conflict", "factual", "temporal"]


class Stage1MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1536,
        hidden_dim_1: int = 1024,
        hidden_dim_2: int = 256,
        dropout: float = 0.2,
        num_type_labels: int = 3,
    ) -> None:
        super().__init__()
        input_dim = embedding_dim * 4

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.binary_head = nn.Linear(hidden_dim_2, 2)
        self.type_head = nn.Linear(hidden_dim_2, num_type_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.binary_head(h), self.type_head(h)


def _format_text(query: str, doc: str) -> str:
    return f"Câu hỏi: {query}\nTài liệu: {doc}"


STAGE2_SYSTEM = (
    "Bạn là chuyên gia phát hiện mâu thuẫn tri thức giữa hai đoạn tài liệu được "
    "truy hồi cho cùng một câu hỏi. Hãy phán đoán chính xác và trả về JSON."
)

STAGE2_INSTRUCTION = """\
Cho câu hỏi và hai đoạn tài liệu (A, B), xác định chúng có MÂU THUẪN với nhau khi
trả lời câu hỏi hay không, và phân loại kiểu mâu thuẫn.

Định nghĩa kiểu:
- "factual": đưa ra các khẳng định/sự kiện trái ngược nhau (số liệu, tên, kết luận khác nhau).
- "temporal": cùng chủ đề nhưng ứng với mốc thời gian khác nhau (thông tin thay đổi theo thời gian).
- "no-conflict": không mâu thuẫn (bổ sung, trùng lặp, hoặc không liên quan đến nhau).

Chỉ tính là mâu thuẫn nếu hai đoạn thực sự không thể cùng đúng khi trả lời câu hỏi.

Câu hỏi: {query}

[Tài liệu A]
{doc_i}

[Tài liệu B]
{doc_j}

Trả về JSON đúng schema sau, không thêm chữ nào khác:
{{"has_conflict": true|false, "type": "factual"|"temporal"|"no-conflict", "summary": "<mô tả ngắn gọn mâu thuẫn hoặc lý do không mâu thuẫn>", "confidence": <số thực 0..1>}}
"""


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ConflictDetector:
    """
    Stage 1 ConflictRAG detector.

    Input:
        query + list retrieved chunks

    Output:
        conflict pairs among top-K chunks.
    """

    def __init__(
        self,
        model_dir: str | Path,
        openai_api_key: str,
        openai_base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        threshold: float | None = None,
        enabled: bool = True,
        llm_model: str = "gpt-4o-mini",
        tau_c: float = 0.7,
        enable_stage2: bool = True,
    ) -> None:
        self.enabled = enabled
        self.model_dir = Path(model_dir)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
        )

        # Stage 1: ngưỡng quyết định nhị phân của MLP.
        self.threshold = threshold or self._load_threshold()
        # Stage 2: route sang LLM khi độ tự tin của Stage 1 < tau_c.
        self.llm_model = llm_model
        self.tau_c = tau_c
        self.enable_stage2 = enable_stage2
        self.label_mapping = self._load_label_mapping()

        self.model = Stage1MLP(embedding_dim=embedding_dim)
        self._load_model()
        self.model.to(self.device)
        self.model.eval()

        self._mem_cache: dict[str, np.ndarray] = {}

    def _load_threshold(self) -> float:
        path = self.model_dir / "threshold_config.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return float(data.get("best_threshold", data.get("threshold", 0.7)))
            except Exception:
                pass
        return 0.7

    def _load_label_mapping(self) -> dict[int, str]:
        path = self.model_dir / "label_mapping.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "id_to_label" in data:
                    return {int(k): v for k, v in data["id_to_label"].items()}
            except Exception:
                pass
        return {0: "no-conflict", 1: "factual", 2: "temporal"}

    def _load_model(self) -> None:
        model_path = self.model_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ConflictRAG model: {model_path}")

        state = torch.load(model_path, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.model.load_state_dict(state)

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        missing = []
        missing_idx = []

        for i, text in enumerate(texts):
            key = _hash_text(text)
            if key not in self._mem_cache:
                missing.append(text)
                missing_idx.append(i)

        if missing:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=missing,
                dimensions=self.embedding_dim,
            )

            for text, item in zip(missing, response.data):
                key = _hash_text(text)
                self._mem_cache[key] = np.array(item.embedding, dtype=np.float32)

        return [self._mem_cache[_hash_text(t)] for t in texts]

    def _make_feature(self, ei: np.ndarray, ej: np.ndarray) -> np.ndarray:
        ei = _l2_normalize(ei)
        ej = _l2_normalize(ej)
        feat = np.concatenate([ei, ej, np.abs(ei - ej), ei * ej], axis=0)
        return feat.astype(np.float32)

    @torch.no_grad()
    def predict_pair(self, query: str, doc_i: str, doc_j: str) -> dict[str, Any]:
        text_i = _format_text(query, doc_i)
        text_j = _format_text(query, doc_j)

        emb_i, emb_j = self._embed_batch([text_i, text_j])
        feat = self._make_feature(emb_i, emb_j)

        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)

        binary_logits, type_logits = self.model(x)

        binary_prob = F.softmax(binary_logits, dim=-1)[0]
        type_prob = F.softmax(type_logits, dim=-1)[0]

        conflict_probability = float(binary_prob[1].item())
        conflict = conflict_probability >= self.threshold

        type_id = int(torch.argmax(type_prob).item())
        type_label = self.label_mapping.get(type_id, LABELS[type_id])

        type_probabilities = {
            self.label_mapping.get(i, LABELS[i]): float(type_prob[i].item())
            for i in range(len(LABELS))
        }

        # Độ tự tin của quyết định nhị phân Stage 1 (khoảng [0.5, 1.0]).
        stage1_confidence = max(conflict_probability, 1.0 - conflict_probability)

        result = {
            "conflict": conflict,
            "conflict_probability": conflict_probability,
            "threshold": self.threshold,
            "type_label": type_label,
            "type_probabilities": type_probabilities,
            "stage": "stage1",
            "stage1_confidence": stage1_confidence,
        }

        # Stage 2: chỉ gọi LLM cho các cặp Stage 1 không đủ tự tin (conf < tau_c).
        if self.enable_stage2 and stage1_confidence < self.tau_c:
            refined = self._stage2_refine(query, doc_i, doc_j)
            if refined is not None:
                result.update(refined)
                result["stage"] = "stage2"

        return result

    def _stage2_refine(self, query: str, doc_i: str, doc_j: str) -> dict[str, Any] | None:
        """Tinh chỉnh bằng LLM cho cặp tài liệu Stage 1 không chắc chắn.

        Trả về None nếu gọi LLM thất bại (caller giữ nguyên quyết định Stage 1).
        """
        instruction = STAGE2_INSTRUCTION.format(
            query=query,
            doc_i=doc_i,
            doc_j=doc_j,
        )
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception:
            logger.exception("RAG_FLOW node=conflict_stage2 llm_error")
            return None

        has_conflict = bool(data.get("has_conflict", False))
        type_label = str(data.get("type", "no-conflict")).strip().lower()
        if type_label not in self.label_mapping.values():
            type_label = "factual" if has_conflict else "no-conflict"
        if has_conflict and type_label == "no-conflict":
            type_label = "factual"
        if not has_conflict:
            type_label = "no-conflict"

        try:
            llm_confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            llm_confidence = 0.0

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "RAG_FLOW node=conflict_stage2 model=%s has_conflict=%s type=%s duration_ms=%.1f",
            self.llm_model,
            has_conflict,
            type_label,
            elapsed_ms,
        )

        return {
            "conflict": has_conflict,
            "type_label": type_label,
            "stage2_summary": str(data.get("summary", "")).strip(),
            "stage2_confidence": llm_confidence,
        }

    def detect_topk(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        min_probability: float | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "has_conflict": False,
                "conflict_pairs": [],
                "num_documents": len(chunks),
                "num_pairs": 0,
            }

        started = time.perf_counter()

        selected = chunks[:top_k]
        pairs = list(combinations(range(len(selected)), 2))
        conflict_pairs = []
        num_stage2 = 0

        with self._lock:
            for i, j in pairs:
                ci = selected[i]
                cj = selected[j]

                doc_i_text = ci.get("text", "") or ""
                doc_j_text = cj.get("text", "") or ""

                if not doc_i_text.strip() or not doc_j_text.strip():
                    continue

                pred = self.predict_pair(query, doc_i_text, doc_j_text)

                if pred.get("stage") == "stage2":
                    num_stage2 += 1

                prob = pred["conflict_probability"]
                if min_probability is not None and prob < min_probability:
                    continue

                if pred["conflict"]:
                    conflict_pairs.append({
                        "doc_i_id": ci.get("chunk_id"),
                        "doc_j_id": cj.get("chunk_id"),
                        "doc_i_label": ci.get("label"),
                        "doc_j_label": cj.get("label"),
                        "doc_i_title": ci.get("doc_title"),
                        "doc_j_title": cj.get("doc_title"),
                        "doc_i_page": ci.get("page"),
                        "doc_j_page": cj.get("page"),
                        "conflict_probability": round(prob, 6),
                        "type_label": pred["type_label"],
                        "type_probabilities": pred["type_probabilities"],
                        "stage": pred.get("stage", "stage1"),
                        "stage2_summary": pred.get("stage2_summary"),
                        "stage2_confidence": pred.get("stage2_confidence"),
                        "doc_i_preview": doc_i_text[:300],
                        "doc_j_preview": doc_j_text[:300],
                    })

        conflict_pairs.sort(
            key=lambda x: x["conflict_probability"],
            reverse=True,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "RAG_FLOW node_end node=conflict_detect chunks=%s pairs=%s stage2=%s conflicts=%s duration_ms=%.1f",
            len(selected),
            len(pairs),
            num_stage2,
            len(conflict_pairs),
            elapsed_ms,
        )

        return {
            "enabled": True,
            "has_conflict": bool(conflict_pairs),
            "threshold": self.threshold,
            "tau_c": self.tau_c,
            "num_documents": len(selected),
            "num_pairs": len(pairs),
            "num_stage2_calls": num_stage2,
            "conflict_pairs": conflict_pairs,
            "duration_ms": round(elapsed_ms, 1),
        }