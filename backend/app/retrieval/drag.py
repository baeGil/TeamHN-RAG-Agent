"""DRAG: conflict-aware retrieval and answer guidance for RAG.

The implementation follows the core idea from the DRAG/CONFLICTS paper:
first predict the relationship between retrieved sources, then adapt retrieval
and answer style to the predicted conflict type.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from ..agent.llm import LLM
from ..indexing.store import KnowledgeBase, RetrievedChunk


CONFLICT_TYPES = {
    "no_conflict": (
        "Sources give equivalent or nearly equivalent answers. Answer directly "
        "and cite the supporting sources."
    ),
    "complementary_information": (
        "Sources provide different but compatible parts of the answer. Combine "
        "them into one cohesive answer."
    ),
    "conflicting_opinions": (
        "Sources disagree because of opinions, research outcomes, or debate. "
        "Present the main viewpoints neutrally."
    ),
    "freshness": (
        "Sources conflict because some are outdated or apply to older time "
        "periods/versions. Prefer the newest or time-applicable source."
    ),
    "misinformation": (
        "The question or at least one retrieved source contains a false, "
        "misleading, or contradicted claim. Correct it using reliable evidence."
    ),
}

CONFLICT_LABELS_VI = {
    "no_conflict": "Không xung đột (No conflict)",
    "misinformation": "Xung đột do thông tin sai lệch (Misinformation)",
    "freshness": "Xung đột do thông tin lỗi thời (Outdated information / Freshness)",
    "conflicting_opinions": "Xung đột quan điểm hoặc kết quả nghiên cứu (Conflicting opinions / Opinion)",
    "complementary_information": "Thông tin bổ sung (Complementary Information)",
}

CONFLICT_DISPLAY_SENTENCES = {
    "no_conflict": "DRAG không phát hiện xung đột đáng kể giữa các nguồn, nên câu trả lời có thể tổng hợp trực tiếp từ bằng chứng đã truy hồi.",
    "misinformation": "DRAG phát hiện khả năng có thông tin sai lệch hoặc tiền đề sai, nên câu trả lời cần sửa claim sai và dựa vào bằng chứng đáng tin cậy.",
    "freshness": "DRAG phát hiện xung đột do thông tin lỗi thời hoặc khác phiên bản thời gian, nên câu trả lời cần ưu tiên nguồn mới hơn hoặc đúng mốc thời gian.",
    "conflicting_opinions": "DRAG phát hiện các nguồn có quan điểm hoặc kết quả nghiên cứu trái chiều, nên câu trả lời cần trình bày trung lập các lập luận chính.",
    "complementary_information": "DRAG phát hiện các nguồn bổ sung cho nhau, nên câu trả lời cần ghép các phần thông tin tương thích thành một kết luận thống nhất.",
}

LEGACY_CONFLICT_MAP = {
    "temporal_scope": "freshness",
    "false_premise": "misinformation",
    "no_relevant_sources": "no_conflict",
}

_TOKEN_RE = re.compile(r"[\w/.-]+", re.UNICODE)
_DOC_REF_RE = re.compile(
    r"\b(?:n[đd]|nghị\s*định|nd)\s*[-_\s]*(\d+/\d{4})",
    re.IGNORECASE | re.UNICODE,
)
_TEMPORAL_CONFLICT_HINT_RE = re.compile(
    r"\b("
    r"mới\s*nhất|hiện\s*nay|hiện\s*tại|so\s*sánh|diễn\s*biến|"
    r"từ\s+các|từ\s+.+\s+đến|áp\s*dụng\s+trong\s+tháng|"
    r"phải\s+dùng|nên\s+dùng|hay\s+là|hay\s+không"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def _tokens(text: str) -> set[str]:
    return {
        tok.strip(".,;:()[]{}").lower()
        for tok in _TOKEN_RE.findall(text)
        if len(tok.strip(".,;:()[]{}")) >= 2
    }


def _text_from_chunk(chunk: Any) -> str:
    if isinstance(chunk, dict):
        parts = [
            chunk.get("doc_title", ""),
            chunk.get("doc_source", ""),
            chunk.get("text", ""),
        ]
        return " ".join(str(p) for p in parts if p)
    parts = [
        getattr(chunk, "doc_title", ""),
        getattr(chunk, "doc_source", ""),
        getattr(chunk, "text", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def _looks_answerable(question: str, chunks: list[Any]) -> bool:
    """Cheap guardrail against classifying clearly matched evidence as irrelevant."""
    q_tokens = _tokens(question)
    if not q_tokens or not chunks:
        return False

    doc_tokens = {
        tok
        for chunk in chunks[:5]
        for tok in _tokens(_text_from_chunk(chunk))
    }
    overlap = q_tokens & doc_tokens

    strong_markers = {
        tok for tok in q_tokens
        if "/" in tok or any(ch.isdigit() for ch in tok) or len(tok) >= 5
    }
    marker_hits = strong_markers & doc_tokens

    return len(overlap) >= 3 and bool(marker_hits)


def _looks_like_single_doc_fact(question: str, chunks: list[Any]) -> bool:
    refs = {m.group(1).lower() for m in _DOC_REF_RE.finditer(question)}
    if len(refs) != 1 or _TEMPORAL_CONFLICT_HINT_RE.search(question):
        return False
    ref = next(iter(refs))
    return any(ref in _text_from_chunk(chunk).lower() for chunk in chunks[:5])


def _normalize_conflict_type(label: str) -> str:
    label = (label or "no_conflict").strip()
    label = LEGACY_CONFLICT_MAP.get(label, label)
    return label if label in CONFLICT_TYPES else "no_conflict"


ASSESS_CONFLICT_SYSTEM = """You classify conflicts between retrieved RAG sources.

This DRAG pipeline uses exactly five conflict labels. Given a user question and
retrieved passages, choose exactly one label:
- no_conflict
- misinformation
- freshness
- conflicting_opinions
- complementary_information

Taxonomy and expected behavior:

1. no_conflict
Sources give equivalent or compatible answers to the query. Minor differences in
specificity are not conflicts. If the passages simply do not contain enough
evidence, still choose no_conflict and explain the insufficiency in rationale.
Expected answer: answer directly when evidence exists; otherwise say the context
does not contain enough information.

2. misinformation
The question or one retrieved source contains a false, misleading, or contradicted
claim. This includes false-premise questions. Expected answer: correct the false
claim using reliable evidence and avoid repeating it as fact.

3. freshness
Sources give incompatible factual answers because some are outdated, newer,
or apply to different dates, reporting periods, effective windows, or legal
versions. Expected answer: prefer the newest/current source, or the source whose
time scope matches the question.

4. conflicting_opinions
Sources genuinely disagree because of opinions, recommendations, debate,
historical interpretation, or research outcomes. Expected answer: neutrally
summarize the main viewpoints without forcing one side.

5. complementary_information
Sources provide different but mutually compatible aspects of the answer, or the
query is underspecified and can validly be answered from multiple perspectives.
Expected answer: merge the relevant aspects.

Decision guidance:
- Do not output any label outside the five-label taxonomy.
- Map old temporal/version conflicts to freshness.
- Map false premises and contradicted false claims to misinformation.
- Do not use conflicting_opinions for date/version differences, changing numeric
  facts, legal effective windows, or compatible partial evidence.

Return JSON with:
{
  "conflict_type": "<one of the five labels>",
  "confidence": 0.0-1.0,
  "rationale": "<short explanation>",
  "answer_policy": "<how the answer should handle the sources>"
}
"""


DRAG_ANSWER_SYSTEM = """You are a conflict-aware RAG assistant.

Answer only from the provided context. Cite sources with bracketed numbers like
[1]. Follow the DRAG conflict assessment:
- no_conflict: answer directly when evidence exists; otherwise say the context does not contain enough information.
- misinformation: correct false or misleading claims using reliable evidence.
- freshness: prioritize the newest/current or time-applicable evidence.
- conflicting_opinions: summarize viewpoints neutrally without forcing one side.
- complementary_information: merge compatible evidence.

Keep the answer concise and faithful to the context."""


@dataclass
class ConflictAssessment:
    conflict_type: str
    confidence: float
    rationale: str
    answer_policy: str

    @property
    def expected_behavior(self) -> str:
        return CONFLICT_TYPES[_normalize_conflict_type(self.conflict_type)]

    @property
    def label_vi(self) -> str:
        return CONFLICT_LABELS_VI[_normalize_conflict_type(self.conflict_type)]

    @property
    def display_sentence(self) -> str:
        return CONFLICT_DISPLAY_SENTENCES.get(
            _normalize_conflict_type(self.conflict_type),
            CONFLICT_DISPLAY_SENTENCES["no_conflict"],
        )

    def to_dict(self) -> dict[str, Any]:
        conflict_type = _normalize_conflict_type(self.conflict_type)
        return {
            "conflict_type": conflict_type,
            "label_vi": self.label_vi,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "answer_policy": self.answer_policy,
            "expected_behavior": self.expected_behavior,
            "display_sentence": self.display_sentence,
        }


class DRAGRetriever:
    """Pipeline-style DRAG retriever.

    This mirrors the pipeline prompt setup in the DRAG/CONFLICTS paper:
    first retrieve C_q once, then predict the conflict type t from [T; q; C_q],
    then pass t as guidance to answer generation. Retrieval itself is not
    expanded or re-ranked based on the predicted type.
    """

    def __init__(self, kb: KnowledgeBase, llm: Optional[LLM] = None, evidence_k: int = 8) -> None:
        self.kb = kb
        self.llm = llm or LLM()
        self.evidence_k = evidence_k

    def assess_conflict(self, question: str, chunks: list[RetrievedChunk]) -> ConflictAssessment:
        if not chunks:
            return ConflictAssessment(
                conflict_type="no_conflict",
                confidence=1.0,
                rationale="No chunks were retrieved.",
                answer_policy="The context does not contain enough information.",
            )

        passages = []
        for i, chunk in enumerate(chunks[: self.evidence_k], start=1):
            page = f" page={chunk.page}" if chunk.page else ""
            title = f" title={chunk.doc_title}" if chunk.doc_title else ""
            source = f" source={chunk.doc_source}" if chunk.doc_source else ""
            passages.append(
                f"[{i}] chunk_id={chunk.chunk_id}{page}{title}{source}\n"
                f"{chunk.text[:1200].replace(chr(10), ' ')}"
            )
        user = f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n" + "\n\n".join(passages)

        try:
            raw = self.llm.chat_json(
                [
                    {"role": "system", "content": ASSESS_CONFLICT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                fast=True,
                node="drag_conflict_assess",
            )
        except Exception as exc:
            return self._unavailable_assessment(f"LLM assessment failed: {exc}")

        raw_label = str(raw.get("conflict_type", "no_conflict")).strip()
        label = _normalize_conflict_type(raw_label)
        if raw_label == "no_relevant_sources" and _looks_answerable(question, chunks):
            label = "no_conflict"
            raw["rationale"] = (
                "Retrieved passages share strong question markers, so DRAG treats "
                "them as answerable evidence rather than no relevant sources."
            )
        elif label == "freshness" and _looks_like_single_doc_fact(question, chunks):
            label = "no_conflict"
            raw["rationale"] = (
                "The question asks for a fact from one specific document/version, "
                "so other time-versioned sources do not create a freshness conflict."
            )
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return ConflictAssessment(
            conflict_type=label,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=str(raw.get("rationale", "")).strip(),
            answer_policy=str(raw.get("answer_policy", "")).strip() or CONFLICT_TYPES[label],
        )

    def assess_conflict_dicts(self, question: str, chunks: list[dict[str, Any]]) -> ConflictAssessment:
        if not chunks:
            return ConflictAssessment(
                conflict_type="no_conflict",
                confidence=1.0,
                rationale="No chunks were retrieved.",
                answer_policy="The context does not contain enough information.",
            )

        passages = []
        for i, chunk in enumerate(chunks[: self.evidence_k], start=1):
            page = f" page={chunk.get('page')}" if chunk.get("page") else ""
            title = f" title={chunk.get('doc_title')}" if chunk.get("doc_title") else ""
            source = f" source={chunk.get('doc_source')}" if chunk.get("doc_source") else ""
            text = str(chunk.get("text", "")).replace(chr(10), " ")[:1200]
            passages.append(
                f"[{i}] chunk_id={chunk.get('chunk_id')}{page}{title}{source}\n{text}"
            )
        user = f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n" + "\n\n".join(passages)

        try:
            raw = self.llm.chat_json(
                [
                    {"role": "system", "content": ASSESS_CONFLICT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                fast=True,
                node="drag_conflict_assess",
            )
        except Exception as exc:
            return self._unavailable_assessment(f"LLM assessment failed: {exc}")

        raw_label = str(raw.get("conflict_type", "no_conflict")).strip()
        label = _normalize_conflict_type(raw_label)
        if raw_label == "no_relevant_sources" and _looks_answerable(question, chunks):
            label = "no_conflict"
            raw["rationale"] = (
                "Retrieved passages share strong question markers, so DRAG treats "
                "them as answerable evidence rather than no relevant sources."
            )
        elif label == "freshness" and _looks_like_single_doc_fact(question, chunks):
            label = "no_conflict"
            raw["rationale"] = (
                "The question asks for a fact from one specific document/version, "
                "so other time-versioned sources do not create a freshness conflict."
            )
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return ConflictAssessment(
            conflict_type=label,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=str(raw.get("rationale", "")).strip(),
            answer_policy=str(raw.get("answer_policy", "")).strip() or CONFLICT_TYPES[label],
        )

    def retrieve(self, question: str, top_k: int = 5) -> tuple[list[RetrievedChunk], ConflictAssessment]:
        base_k = max(top_k, self.evidence_k)
        base_hits = self.kb.retrieve(question, base_k)
        assessment = self.assess_conflict(question, base_hits)
        return base_hits[:top_k], assessment

    def build_answer_messages(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        assessment: ConflictAssessment,
    ) -> list[dict[str, str]]:
        context = _build_labeled_context(chunks)
        user = (
            f"QUESTION:\n{question}\n\n"
            f"DRAG_CONFLICT_TYPE: {assessment.conflict_type}\n"
            f"DRAG_RATIONALE: {assessment.rationale}\n"
            f"DRAG_ANSWER_POLICY: {assessment.answer_policy}\n"
            f"EXPECTED_BEHAVIOR: {assessment.expected_behavior}\n\n"
            f"CONTEXT:\n{context}"
        )
        return [
            {"role": "system", "content": DRAG_ANSWER_SYSTEM},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _unavailable_assessment(reason: str) -> ConflictAssessment:
        return ConflictAssessment(
            conflict_type="no_conflict",
            confidence=0.0,
            rationale=reason,
            answer_policy=CONFLICT_TYPES["no_conflict"],
        )


def _build_labeled_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for item in chunks:
        label = item.get("label", "?")
        title = item.get("doc_title") or item.get("title") or ""
        page = item.get("page")
        loc = ", ".join(x for x in (title, f"page {page}" if page else "") if x)
        head = f"[{label}]" + (f" ({loc})" if loc else "")
        parts.append(f"{head}\n{item.get('text', '')}")
    return "\n\n".join(parts)
