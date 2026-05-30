"""Deterministic, controllable RAG agent (inspired by NirDiamant/Controllable-RAG-Agent).

Control flow (a deterministic graph acting as the agent's "brain"):

    route ─┬─ no_retrieval ───────────────────────────► answer
           ├─ simple ─► retrieve ─────────────► synthesize ─► answer
           └─ complex ► plan ► (retrieve ► distill ► verify)* ► [replan?] ► synthesize

Every node emits a trace event so the UI can monitor the agent in real time.
The final answer is streamed token-by-token. Answers are grounded in retrieved
context only; unsupported questions yield "Không tìm thấy thông tin trong tài liệu."
"""
import re
from dataclasses import asdict
from typing import Any, Iterator, Optional

from ..config import get_settings
from ..indexing.store import KnowledgeBase, RetrievedChunk
from . import prompts
from .llm import LLM

_CITE_RE = re.compile(r"\[(\d+)\]")


def _chunk_to_dict(c: RetrievedChunk) -> dict[str, Any]:
    d = asdict(c)
    return d


class Agent:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.llm = LLM()
        self.settings = get_settings()

    # ---------------- nodes ----------------
    def _route(self, question: str, history: list[dict]) -> tuple[str, str]:
        if not self.kb.vector.ready and self.kb.bm25.count == 0:
            return "no_retrieval", "Chưa có tài liệu nào được nạp."
        msgs = [
            {"role": "system", "content": prompts.ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True)
            route = data.get("route", "simple")
            if route not in {"no_retrieval", "simple", "complex"}:
                route = "simple"
            return route, data.get("reason", "")
        except Exception:
            return "simple", "Mặc định single-hop."

    def _plan(self, question: str) -> list[str]:
        msgs = [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {"role": "user", "content": question},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True)
            subqs = [s for s in data.get("subquestions", []) if isinstance(s, str) and s.strip()]
            return subqs[:4] or [question]
        except Exception:
            return [question]

    def _distill(self, subq: str, chunks: list[dict]) -> str:
        ctx = prompts.build_context(chunks, label_key="label")
        msgs = [
            {"role": "system", "content": prompts.DISTILL_SYSTEM},
            {"role": "user", "content": f"CÂU HỎI CON:\n{subq}\n\nĐOẠN TRÍCH:\n{ctx}"},
        ]
        return self.llm.chat(msgs, fast=True).strip()

    def _verify(self, claim: str, chunks: list[dict]) -> tuple[bool, str]:
        ctx = prompts.build_context(chunks, label_key="label")
        msgs = [
            {"role": "system", "content": prompts.VERIFY_SYSTEM},
            {"role": "user", "content": f"NHẬN ĐỊNH:\n{claim}\n\nNGỮ CẢNH:\n{ctx}"},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True)
            return bool(data.get("grounded", False)), data.get("reason", "")
        except Exception:
            return True, "Bỏ qua kiểm tra."

    # ---------------- orchestration ----------------
    def run(self, question: str, history: Optional[list[dict]] = None) -> Iterator[dict]:
        history = history or []
        trace: list[dict] = []

        def emit(etype: str, data: Any) -> dict:
            ev = {"type": etype, "data": data}
            if etype != "token":
                trace.append(ev)
            return ev

        route, reason = self._route(question, history)
        yield emit("route", {"route": route, "reason": reason})

        if route == "no_retrieval":
            yield from self._answer_chitchat(question, history, trace, emit)
            return

        if route == "complex":
            subqs = self._plan(question)
            yield emit("plan", {"subquestions": subqs})
        else:
            subqs = [question]

        pool: dict[int, dict[str, Any]] = {}
        steps: list[dict[str, Any]] = []

        for i, subq in enumerate(subqs):
            yield emit("subquestion", {"index": i, "subquestion": subq})
            hits = self.kb.retrieve(subq, top_k=self.settings.final_top_k)
            hit_dicts = [_chunk_to_dict(h) for h in hits]
            for hd in hit_dicts:
                pool.setdefault(hd["chunk_id"], hd)
            yield emit(
                "retrieved",
                {
                    "subquestion": subq,
                    "chunks": [
                        {
                            "chunk_id": h["chunk_id"],
                            "doc_title": h["doc_title"],
                            "page": h["page"],
                            "preview": h["text"][:200],
                            "score": round(h["score"], 4),
                            "bm25_score": h["bm25_score"],
                            "dense_score": h["dense_score"],
                            "rerank_score": h["rerank_score"],
                        }
                        for h in hit_dicts
                    ],
                },
            )

            if route == "complex" and hit_dicts:
                labeled = self._with_labels(hit_dicts)
                distilled = self._distill(subq, labeled)
                relevant = distilled != "KHÔNG_LIÊN_QUAN" and bool(distilled)
                grounded, vreason = (True, "")
                if relevant:
                    grounded, vreason = self._verify(distilled, labeled)
                yield emit(
                    "distill",
                    {"subquestion": subq, "note": distilled, "relevant": relevant},
                )
                yield emit("verify", {"subquestion": subq, "grounded": grounded, "reason": vreason})
                steps.append(
                    {"subquestion": subq, "note": distilled, "relevant": relevant, "grounded": grounded}
                )

        context_chunks = self._select_context(pool, steps, route)
        if not context_chunks:
            yield from self._emit_not_found(trace, emit)
            return

        yield emit(
            "synthesize",
            {"n_context": len(context_chunks), "labels": [c["label"] for c in context_chunks]},
        )
        yield from self._synthesize(question, history, context_chunks, steps, route, trace, emit)

    # ---------------- helpers ----------------
    @staticmethod
    def _with_labels(chunks: list[dict]) -> list[dict]:
        out = []
        for i, c in enumerate(chunks, start=1):
            d = dict(c)
            d["label"] = i
            out.append(d)
        return out

    def _select_context(self, pool, steps, route) -> list[dict]:
        chunks = sorted(pool.values(), key=lambda c: c.get("score", 0.0), reverse=True)
        limit = 8 if route == "complex" else self.settings.final_top_k
        chunks = chunks[:limit]
        return self._with_labels(chunks)

    def _build_citations(self, context_chunks: list[dict], cited_labels: set[int]) -> list[dict]:
        cits = []
        for c in context_chunks:
            cits.append(
                {
                    "label": c["label"],
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "doc_title": c["doc_title"],
                    "doc_source": c["doc_source"],
                    "page": c["page"],
                    "section": c["section"],
                    "text": c["text"],
                    "score": round(c.get("score", 0.0), 4),
                    "cited": c["label"] in cited_labels,
                }
            )
        return cits

    def _synthesize(self, question, history, context_chunks, steps, route, trace, emit):
        ctx = prompts.build_context(context_chunks, label_key="label")
        notes = ""
        if route == "complex":
            grounded_notes = [
                f"- {s['subquestion']}\n  {s['note']}"
                for s in steps
                if s.get("relevant") and s.get("grounded")
            ]
            if grounded_notes:
                notes = "\n\nGHI CHÚ ĐÃ CHẮT LỌC (đã kiểm chứng):\n" + "\n".join(grounded_notes)
        user_msg = (
            f"NGỮ CẢNH:\n{ctx}{notes}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo đúng quy tắc, trích dẫn bằng [số] tương ứng với các đoạn ngữ cảnh."
        )
        msgs = [{"role": "system", "content": prompts.ANSWER_SYSTEM}]
        for h in history[-4:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": user_msg})

        answer_parts: list[str] = []
        for tok in self.llm.stream(msgs, fast=False):
            answer_parts.append(tok)
            yield emit("token", {"text": tok})
        answer = "".join(answer_parts).strip()

        cited_labels = {int(m) for m in _CITE_RE.findall(answer)}
        citations = self._build_citations(context_chunks, cited_labels)
        yield emit(
            "final",
            {
                "answer": answer,
                "citations": citations,
                "trace": trace,
                "usage": self.llm.usage,
                "route": route,
            },
        )

    def _answer_chitchat(self, question, history, trace, emit):
        system = (
            "Bạn là trợ lý hỏi đáp tài liệu tiếng Việt. Người dùng đang chào hỏi hoặc "
            "hỏi về khả năng của bạn. Trả lời ngắn gọn, thân thiện và mời họ nạp tài liệu "
            "(PDF/URL/văn bản) rồi đặt câu hỏi. Không bịa thông tin về tài liệu."
        )
        msgs = [{"role": "system", "content": system}]
        for h in history[-4:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": question})
        parts = []
        for tok in self.llm.stream(msgs, fast=True):
            parts.append(tok)
            yield emit("token", {"text": tok})
        answer = "".join(parts).strip()
        yield emit(
            "final",
            {"answer": answer, "citations": [], "trace": trace, "usage": self.llm.usage, "route": "no_retrieval"},
        )

    def _emit_not_found(self, trace, emit):
        answer = "Không tìm thấy thông tin trong tài liệu."
        yield emit("token", {"text": answer})
        yield emit(
            "final",
            {"answer": answer, "citations": [], "trace": trace, "usage": self.llm.usage, "route": "not_found"},
        )
