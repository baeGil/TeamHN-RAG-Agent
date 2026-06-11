"""Deterministic, controllable RAG agent with replanning loop.

Control flow:

    route ─┬─ no_retrieval ───────────────────────────────────► answer (chitchat)
           ├─ simple ─► retrieve ───────────► synthesize ─► verify_answer ──┐
           └─ complex                                                           │
                  │                                                             │
                  ▼                                                             │
               plan                                                             │
                  │                                                             │
       ┌─────────▼──────────────────────────────────────┐                     │
       │  iteration 0..max_replan_iters:                 │                     │
       │                                                  │                     │
       │  retrieve (parallel)                            │                     │
       │  distill + verify (parallel)                    │                     │
       │  sufficiency_check                              │                     │
       │    │              │                             │                     │
       │ sufficient    insufficient                      │                     │
       │    │              │                             │                     │
       │    │     improved from last iter?               │                     │
       │    │      │           │                         │                     │
       │    │     yes          no → early_stop ─────┐   │                     │
       │    │      │                                │   │                     │
       │    │   replan → loop back                  │   │                     │
       │    ▼                                        ▼   ▼                     │
       │  synthesize ◄──────────────────────────────────────────────────────────┘
       │       │
       │  verify_answer
       │    │           │
       │ grounded   hallucinate → regenerate (max 1)
       │    │           │
       │    ▼           ▼
       │  final answer  final answer
       └───────────────────────────────────────────────

Every node emits a trace event so the UI can monitor the agent in real time.
The final answer is streamed token-by-token. Answers are grounded in retrieved
context only; unsupported questions yield a partial warning.
"""
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Iterator, Optional

from ..config import get_settings
from ..indexing.store import KnowledgeBase, RetrievedChunk
from ..retrieval.conflict import ConflictDetector
from . import prompts
from .llm import LLM

_CITE_RE = re.compile(r"\[(\d+)\]")
_GREETING_RE = re.compile(
    r"^[\s]*(xin\s*chào|hello|hi+|hey+|chào\s+bạn|cảm\s*ơn|thanks|thank\s*you|"
    r"tạm\s*biệt|goodbye|bye|bạn\s+là\s+ai[\?\!]*|bạn\s+có\s+thể\s+làm\s+gì[\?\!]*|"
    r"trợ\s+lý|đang\s+sẵn\s*sàng|"
    r"ok[\?\!\.]*|okay[\?\!\.]*)[\s\.\!\?]*$",
    re.IGNORECASE | re.UNICODE,
)
logger = logging.getLogger("rag.flow")


def _chunk_string(s: str, size: int) -> list[str]:
    """Split string into chunks of approximately `size` chars for streaming."""
    if not s:
        return []
    chunks = []
    for i in range(0, len(s), size):
        chunks.append(s[i:i + size])
    return chunks


def _chunk_to_dict(c: RetrievedChunk) -> dict[str, Any]:
    d = asdict(c)
    return d


class Agent:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.llm = LLM()
        self.settings = get_settings()
        self.conflict_detector = ConflictDetector(
            enabled=self.settings.enable_conflict_check,
            model_name=self.settings.conflict_model,
            api_key=self.settings.hf_token,
            min_confidence=self.settings.conflict_min_confidence,
            max_pairs=self.settings.conflict_max_pairs,
        )

    def summarize_conversation(
        self,
        session_id: str,
        old_summary: Optional[str],
        messages: list[dict[str, str]],
    ) -> str:
        if not messages:
            return old_summary or ""
        parts = []
        if old_summary:
            parts.append(f"TÓM TẮT TRƯỚC ĐÓ:\n{old_summary}")
        parts.append("HỘI THOẠI MỚI:")
        for m in messages:
            role_label = "Người dùng" if m["role"] == "user" else "Trợ lý"
            parts.append(f"{role_label}: {m['content']}")
        conv_text = "\n".join(parts)
        msgs = [
            {"role": "system", "content": prompts.SUMMARIZE_SYSTEM},
            {"role": "user", "content": conv_text},
        ]
        try:
            use_fast = self.settings.summary_model == self.settings.llm_model_fast
            summary = self.llm.chat(msgs, fast=use_fast, node="summarize").strip()
            return summary
        except Exception:
            return old_summary or "\n".join(f"{m['role']}: {m['content'][:200]}" for m in messages)

    # ---------------- nodes ----------------

    @staticmethod
    def _history_msgs(
        history: list[dict], summary: Optional[str] = None, window: int = 4,
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if summary:
            msgs.append({"role": "system", "content": f"TÓM TẮT LỊCH SỬ TRÒ CHUYỆN TRƯỚC ĐÓ:\n{summary}"})
        for h in history[-window:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        return msgs

    def _route(self, question: str, history: list[dict]) -> tuple[str, str]:
        if not self.kb.vector.ready and self.kb.bm25.count == 0:
            return "no_retrieval", "Chưa có tài liệu nào được nạp."
        if _GREETING_RE.match(question.strip()):
            return "no_retrieval", "Câu chào hỏi — không cần tra cứu."
        msgs = [
            {"role": "system", "content": prompts.ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="router")
            route = data.get("route", "simple")
            if route not in {"no_retrieval", "simple", "complex"}:
                route = "simple"
            if route == "no_retrieval":
                return "simple", "Buộc tra cứu tài liệu (tránh bỏ sót)."
            return route, data.get("reason", "")
        except Exception as e:
            return "simple", f"Lỗi định tuyến: {e}"

    def _plan(self, question: str) -> list[str]:
        msgs = [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {"role": "user", "content": question},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="planner")
            subqs = [s for s in data.get("subquestions", []) if isinstance(s, str) and s.strip()]
            return subqs[:4] or [question]
        except Exception as e:
            return [question]

    def _distill(self, subq: str, chunks: list[dict]) -> str:
        ctx = prompts.build_context(chunks, label_key="label")
        msgs = [
            {"role": "system", "content": prompts.DISTILL_SYSTEM},
            {"role": "user", "content": f"CÂU HỎI CON:\n{subq}\n\nĐOẠN TRÍCH:\n{ctx}"},
        ]
        return self.llm.chat(msgs, fast=True, node="distill").strip()

    def _generate_and_verify(
        self, question, history, context_chunks, steps, route, conflicts=None, summary=None,
    ) -> tuple[str, bool, str]:
        """Generate answer internally (no streaming) and verify grounding."""
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

        notes += self._format_conflict_notes(conflicts or [])

        user_msg = (
            f"NGỮ CẢNH:\n{ctx}{notes}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo đúng quy tắc, trích dẫn bằng [số] tương ứng với các đoạn ngữ cảnh."
        )
        msgs = [{"role": "system", "content": prompts.ANSWER_SYSTEM}]
        msgs.extend(self._history_msgs(history, summary))
        msgs.append({"role": "user", "content": user_msg})

        answer_text = self.llm.chat(msgs, fast=False, node="synthesize").strip()
        grounded, reason = self._verify_answer(answer_text, context_chunks)
        return answer_text, grounded, reason

    def _verify(self, claim: str, chunks: list[dict]) -> tuple[bool, str]:
        ctx = prompts.build_context(chunks, label_key="label")
        msgs = [
            {"role": "system", "content": prompts.VERIFY_SYSTEM},
            {"role": "user", "content": f"NHẬN ĐỊNH:\n{claim}\n\nNGỮ CẢNH:\n{ctx}"},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="verify")
            return bool(data.get("grounded", False)), data.get("reason", "")
        except Exception:
            return True, "Bỏ qua kiểm tra."

    def _check_sufficiency(self, question: str, grounded_notes: list[str]) -> tuple[bool, str]:
        if not grounded_notes:
            return False, "Không có ghi chú nào bám nguồn."
        notes_text = "\n".join(f"- {n}" for n in grounded_notes)
        msgs = [
            {"role": "system", "content": prompts.SUFFICIENCY_SYSTEM},
            {"role": "user", "content": f"CÂU HỎI GỐC:\n{question}\n\nGHI CHÚ ĐÃ KIỂM CHỨNG:\n{notes_text}"},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="sufficiency")
            return bool(data.get("sufficient", False)), data.get("reason", "")
        except Exception:
            return True, ""

    def _replan(self, question: str, failed: list[dict], grounded_notes: list[str], iteration: int) -> list[str]:
        failed_text = "\n".join(
            f'- "{f["subquestion"]}" (lý do: {f.get("reason", "không bám nguồn")})'
            for f in failed
        )
        success_text = ""
        if grounded_notes:
            success_text = "\n\nCÁC BƯỚC ĐÃ THÀNH CÔNG:\n" + "\n".join(
                f"- {n}" for n in grounded_notes[:6]
            )
        msgs = [
            {"role": "system", "content": prompts.REPLANNER_SYSTEM},
            {"role": "user", "content": (
                f"CÂU HỎI GỐC:\n{question}\n\n"
                f"CÁC BƯỚC THẤT BẠI:\n{failed_text}"
                f"{success_text}\n\n"
                f"Hãy đặt lại câu hỏi cho các bước thất bại, cụ thể hơn hoặc theo hướng khác. "
                f"Vòng lặp lại lần {iteration + 1}."
            )},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="replan")
            new = [s for s in data.get("subquestions", []) if isinstance(s, str) and s.strip()]
            return new[:4] or [question]
        except Exception:
            return [f["subquestion"] for f in failed] or [question]

    @staticmethod
    def _merge_subquestions(
        old_subqs: list[str],
        step_results: list[dict],
        new_subqs: list[str],
    ) -> list[str]:
        merged = []
        new_idx = 0
        for sq, r in zip(old_subqs, step_results):
            if r.get("grounded") and r.get("relevant"):
                merged.append(sq)
            elif new_idx < len(new_subqs):
                merged.append(new_subqs[new_idx])
                new_idx += 1
        merged.extend(new_subqs[new_idx:])
        return merged[:4]

    def _verify_answer(self, answer: str, context_chunks: list[dict]) -> tuple[bool, str]:
        ctx = prompts.build_context(context_chunks, label_key="label")
        msgs = [
            {"role": "system", "content": prompts.ANSWER_VERIFY_SYSTEM},
            {"role": "user", "content": f"CÂU TRẢ LỜI:\n{answer}\n\nNGỮ CẢNH:\n{ctx}"},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="verify_answer")
            grounded = bool(data.get("grounded", True))
            claims = data.get("ungrounded_claims", [])
            reason = data.get("reason", "")
            if claims:
                reason = "; ".join(claims)
            return grounded, reason
        except Exception:
            return True, ""

    # ---------------- orchestration ----------------
    def run(self, question: str, history: Optional[list[dict]] = None, summary: Optional[str] = None) -> Iterator[dict]:
        history = history or []
        trace: list[dict] = []
        run_started = time.perf_counter()
        logger.info(
            "RAG_FLOW run_start question_chars=%s history_messages=%s has_summary=%s",
            len(question),
            len(history),
            bool(summary),
        )

        def emit(etype: str, data: Any) -> dict:
            ev = {"type": etype, "data": data}
            if etype not in ("token", "final"):
                trace.append(ev)
            return ev

        node_started = time.perf_counter()
        yield emit("thinking", {"node": "router"})
        route, reason = self._route(question, history)
        logger.info(
            "RAG_FLOW node_end node=router duration_ms=%.1f route=%s reason_chars=%s",
            (time.perf_counter() - node_started) * 1000,
            route,
            len(reason or ""),
        )
        yield emit("route", {"route": route, "reason": reason})

        if route == "no_retrieval":
            yield from self._answer_chitchat(question, history, trace, emit, summary=summary)
            logger.info(
                "RAG_FLOW run_end duration_ms=%.1f route=%s partial=%s iterations=%s regenerated=%s",
                (time.perf_counter() - run_started) * 1000,
                route,
                False,
                0,
                False,
            )
            return

        if route == "complex":
            node_started = time.perf_counter()
            yield emit("thinking", {"node": "planner"})
            subqs = self._plan(question)
            logger.info(
                "RAG_FLOW node_end node=planner duration_ms=%.1f subquestions=%s",
                (time.perf_counter() - node_started) * 1000,
                len(subqs),
            )
            yield emit("plan", {"subquestions": subqs})
        else:
            subqs = [question]

        pool: dict[int, dict[str, Any]] = {}
        steps: list[dict[str, Any]] = []
        iteration = 0
        max_iters = self.settings.max_replan_iters
        prev_failed_count = float("inf")

        while True:
            # --- Retrieve ---
            node_started = time.perf_counter()
            yield emit("thinking", {"node": "retrieve"})
            retrieval_jobs: dict[int, list[dict]] = {}
            with ThreadPoolExecutor(max_workers=min(4, len(subqs))) as ex:
                futures = {
                    ex.submit(self.kb.retrieve, sq, self.settings.final_top_k): i
                    for i, sq in enumerate(subqs)
                }
                for fut in futures:
                    i = futures[fut]
                    hits = fut.result()
                    retrieval_jobs[i] = [_chunk_to_dict(h) for h in hits]
            logger.info(
                "RAG_FLOW node_end node=retrieve iteration=%s duration_ms=%.1f subquestions=%s hits=%s",
                iteration,
                (time.perf_counter() - node_started) * 1000,
                len(subqs),
                sum(len(v) for v in retrieval_jobs.values()),
            )

            for i, subq in enumerate(subqs):
                yield emit("subquestion", {"index": i, "subquestion": subq})
                hit_dicts = retrieval_jobs.get(i, [])
                for hd in hit_dicts:
                    pool.setdefault(hd["chunk_id"], hd)
                yield emit(
                    "retrieved",
                    {
                        "subquestion": subq,
                        "index": i,
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

            # --- Distill + Verify (complex only) ---
            if route == "complex":
                node_started = time.perf_counter()
                yield emit("thinking", {"node": "distill"})
                distill_results: dict[int, dict] = {}

                def _distill_and_verify(idx_subq):
                    i, subq = idx_subq
                    hit_dicts = retrieval_jobs.get(i, [])
                    if not hit_dicts:
                        return i, {"note": "", "relevant": False, "grounded": False, "reason": ""}
                    labeled = self._with_labels(hit_dicts)
                    try:
                        note = self._distill(subq, labeled)
                        rel = note != "KHÔNG_LIÊN_QUAN" and bool(note)
                        grounded, vreason = (True, "")
                        if rel:
                            grounded, vreason = self._verify(note, labeled)
                        return i, {"note": note, "relevant": rel, "grounded": grounded, "reason": vreason}
                    except Exception as e:
                        return i, {"note": f"Lỗi: {e}", "relevant": False, "grounded": False, "reason": str(e), "error": True}

                with ThreadPoolExecutor(max_workers=min(4, len(subqs))) as ex:
                    for i, res in ex.map(_distill_and_verify, list(enumerate(subqs))):
                        distill_results[i] = res
                logger.info(
                    "RAG_FLOW node_end node=distill_verify iteration=%s duration_ms=%.1f subquestions=%s",
                    iteration,
                    (time.perf_counter() - node_started) * 1000,
                    len(subqs),
                )

                yield emit("thinking", {"node": "verify"})
                for i, subq in enumerate(subqs):
                    r = distill_results.get(i, {})
                    is_error = r.get("error", False)
                    if is_error:
                        yield emit("error", {"node": "distill", "message": r.get("reason", "Lỗi chắt lọc")})
                    yield emit(
                        "distill",
                        {"subquestion": subq, "index": i, "note": r.get("note", ""), "relevant": r.get("relevant", False)},
                    )
                    if is_error:
                        yield emit("error", {"node": "verify", "message": r.get("reason", "Lỗi kiểm chứng")})
                    yield emit(
                        "verify",
                        {"subquestion": subq, "index": i, "grounded": r.get("grounded", True), "reason": r.get("reason", "")},
                    )
                    steps.append({"subquestion": subq, **r})

                # --- Check convergence for replanning ---
                if self.settings.enable_replan:
                    failed = [
                        {"subquestion": s["subquestion"], "reason": s.get("reason", "")}
                        for s in steps
                        if not s.get("grounded") or not s.get("relevant")
                    ]
                    if not failed:
                        yield emit("converged", {"iteration": iteration})
                        break

                    # --- Sufficiency check ---
                    if self.settings.enable_sufficiency:
                        grounded_notes = [
                            f"{s['subquestion']}: {s['note']}"
                            for s in steps
                            if s.get("relevant") and s.get("grounded") and s.get("note")
                        ]
                        node_started = time.perf_counter()
                        yield emit("thinking", {"node": "sufficiency"})
                        sufficient, s_reason = self._check_sufficiency(question, grounded_notes)
                        logger.info(
                            "RAG_FLOW node_end node=sufficiency iteration=%s duration_ms=%.1f sufficient=%s reason_chars=%s",
                            iteration,
                            (time.perf_counter() - node_started) * 1000,
                            sufficient,
                            len(s_reason or ""),
                        )
                        yield emit("sufficiency", {"sufficient": sufficient, "reason": s_reason, "iteration": iteration})
                        if sufficient:
                            break

                    # --- Early termination: no improvement ---
                    current_failed_count = len(failed)
                    if current_failed_count >= prev_failed_count:
                        yield emit("early_stop", {"iteration": iteration, "reason": "Không cải thiện so với vòng trước"})
                        break

                    # --- Max iterations reached ---
                    if iteration + 1 >= max_iters:
                        yield emit("max_iters", {"iteration": iteration, "failed": [f["subquestion"] for f in failed]})
                        break

                    # --- Replan ---
                    node_started = time.perf_counter()
                    yield emit("thinking", {"node": "replan"})
                    grounded_notes_for_replan = [
                        f"{s['subquestion']}: {s['note']}"
                        for s in steps
                        if s.get("relevant") and s.get("grounded") and s.get("note")
                    ]
                    new_subqs = self._replan(question, failed, grounded_notes_for_replan, iteration)
                    logger.info(
                        "RAG_FLOW node_end node=replan iteration=%s duration_ms=%.1f failed=%s new_subquestions=%s",
                        iteration,
                        (time.perf_counter() - node_started) * 1000,
                        len(failed),
                        len(new_subqs),
                    )
                    step_results = [steps[i] for i in range(len(subqs))] if len(steps) == len(subqs) else steps[-len(subqs):]
                    subqs = self._merge_subquestions(subqs, step_results, new_subqs)
                    yield emit("replan", {"iteration": iteration + 1, "subquestions": subqs})

                    # Reset steps for new iteration, keeping grounded steps
                    steps = [s for s in steps if s.get("grounded") and s.get("relevant")]
                    prev_failed_count = current_failed_count
                    iteration += 1
                else:
                    # Replanning disabled, just proceed
                    break
            else:
                # Simple route: no loop
                break

        # --- Build context ---
        partial_warning = False
        if route == "complex" and steps:
            grounded_count = sum(1 for s in steps if s.get("grounded") and s.get("relevant"))
            if grounded_count < len(steps):
                partial_warning = True

        context_chunks = self._select_context(pool, steps, route)
        if not context_chunks:
            yield from self._emit_not_found(trace, emit)
            logger.info(
                "RAG_FLOW run_end duration_ms=%.1f route=%s partial=%s iterations=%s regenerated=%s",
                (time.perf_counter() - run_started) * 1000,
                "not_found",
                False,
                iteration,
                False,
            )
            return

        # --- Conflict check (after rerank/context selection) ---
        conflict_report: dict[str, Any] = {"conflicts": [], "checked_pairs": 0}
        conflict_event: dict[str, Any] | None = None
        conflicts: list[dict] = []
        if self.settings.enable_conflict_check:
            node_started = time.perf_counter()
            yield emit("thinking", {"node": "conflict"})
            conflict_report = self.conflict_detector.check(context_chunks)
            conflicts = conflict_report.get("conflicts", []) or []
            logger.info(
                "RAG_FLOW node_end node=conflict duration_ms=%.1f checked_pairs=%s conflicts=%s available=%s",
                (time.perf_counter() - node_started) * 1000,
                conflict_report.get("checked_pairs", 0),
                len(conflicts),
                conflict_report.get("available", False),
            )
            conflict_event = {
                "available": conflict_report.get("available", False),
                "reason": conflict_report.get("reason", ""),
                "model": conflict_report.get("model", self.settings.conflict_model),
                "checked_pairs": conflict_report.get("checked_pairs", 0),
                "input_chars": conflict_report.get("input_chars"),
                "conflicts": conflicts,
            }
            yield emit("conflicts", conflict_event)
            if conflicts:
                partial_warning = True

        # --- Synthesize ---
        node_started = time.perf_counter()
        yield emit("thinking", {"node": "synthesize"})
        yield emit(
            "synthesize",
            {"n_context": len(context_chunks), "labels": [c["label"] for c in context_chunks]},
        )
        logger.info(
            "RAG_FLOW node_start node=synthesize route=%s context_chunks=%s",
            route,
            len(context_chunks),
        )

        # If answer verification is enabled, generate internally first to check grounding,
        # then stream the final answer (original or regenerated).
        if self.settings.enable_answer_verify and route != "no_retrieval":
            # Generate internally (no streaming) and verify grounding
            answer_text, is_grounded, v_reason = self._generate_and_verify(
                question, history, context_chunks, steps, route, conflicts=conflicts, summary=summary,
            )
            logger.info(
                "RAG_FLOW node_end node=synthesize duration_ms=%.1f answer_chars=%s grounded=%s",
                (time.perf_counter() - node_started) * 1000,
                len(answer_text),
                is_grounded,
            )

            if is_grounded:
                # Answer is grounded — stream it directly (already generated, just emit)
                yield emit("thinking", {"node": "verify_answer"})
                yield emit("verify_answer", {"grounded": True, "reason": ""})
                cited_labels = {int(m) for m in _CITE_RE.findall(answer_text)}
                citations = self._build_citations(context_chunks, cited_labels)
                for tok_chunk in _chunk_string(answer_text, 3):
                    yield emit("token", {"text": tok_chunk})
                trace_snapshot = list(trace)
                yield emit(
                    "final",
                    {
                        "answer": answer_text,
                        "citations": citations,
                        "trace": trace_snapshot,
                        "usage": self.llm.usage,
                        "route": route,
                        "partial": partial_warning,
                        "iterations": iteration,
                        "regenerated": False,
                        "conflicts": conflicts,
                        "conflict_check": conflict_event,
                    },
                )
                logger.info(
                    "RAG_FLOW run_end duration_ms=%.1f route=%s partial=%s iterations=%s regenerated=%s",
                    (time.perf_counter() - run_started) * 1000,
                    route,
                    partial_warning,
                    iteration,
                    False,
                )
            else:
                # Answer has hallucinations — regenerate with warning
                yield emit("thinking", {"node": "verify_answer"})
                yield emit("verify_answer", {"grounded": False, "reason": v_reason})
                if self.settings.max_answer_regenerations > 0:
                    yield emit("thinking", {"node": "regenerate"})
                    answer_result = yield from self._synthesize(
                        question, history, context_chunks, steps, route, trace, emit,
                        partial_warning=partial_warning, regenerate_reason=v_reason,
                        iterations=iteration, conflicts=conflicts, conflict_check=conflict_event, summary=summary,
                    )
                    yield emit("verify_answer", {"grounded": True, "reason": "Đã tạo lại câu trả lời."})
                    # _synthesize skips "final" when regenerating — emit it here.
                    cited_labels = {int(m) for m in _CITE_RE.findall(answer_result)}
                    citations = self._build_citations(context_chunks, cited_labels)
                    trace_snapshot = list(trace)
                    yield emit(
                        "final",
                        {
                            "answer": answer_result,
                            "citations": citations,
                            "trace": trace_snapshot,
                            "usage": self.llm.usage,
                            "route": route,
                            "partial": True,
                            "iterations": iteration,
                            "regenerated": True,
                            "conflicts": conflicts,
                            "conflict_check": conflict_event,
                        },
                    )
                    logger.info(
                        "RAG_FLOW run_end duration_ms=%.1f route=%s partial=%s iterations=%s regenerated=%s",
                        (time.perf_counter() - run_started) * 1000,
                        route,
                        True,
                        iteration,
                        True,
                    )
                else:
                    # No regeneration allowed — stream the existing answer with partial warning
                    cited_labels = {int(m) for m in _CITE_RE.findall(answer_text)}
                    citations = self._build_citations(context_chunks, cited_labels)
                    for tok_chunk in _chunk_string(answer_text, 3):
                        yield emit("token", {"text": tok_chunk})
                    trace_snapshot = list(trace)
                    yield emit(
                        "final",
                        {
                            "answer": answer_text,
                            "citations": citations,
                            "trace": trace_snapshot,
                            "usage": self.llm.usage,
                            "route": route,
                            "partial": True,
                            "iterations": iteration,
                            "regenerated": False,
                            "conflicts": conflicts,
                            "conflict_check": conflict_event,
                        },
                    )
                    logger.info(
                        "RAG_FLOW run_end duration_ms=%.1f route=%s partial=%s iterations=%s regenerated=%s",
                        (time.perf_counter() - run_started) * 1000,
                        route,
                        True,
                        iteration,
                        False,
                    )
        else:
            # No verification — stream directly via _synthesize
            answer_result = yield from self._synthesize(
                question, history, context_chunks, steps, route, trace, emit,
                partial_warning=partial_warning, iterations=iteration, conflicts=conflicts,
                conflict_check=conflict_event, summary=summary,
            )
            logger.info(
                "RAG_FLOW run_end duration_ms=%.1f route=%s partial=%s iterations=%s regenerated=%s answer_chars=%s",
                (time.perf_counter() - run_started) * 1000,
                route,
                partial_warning,
                iteration,
                False,
                len(answer_result or ""),
            )

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

    @staticmethod
    def _format_conflict_notes(conflicts: list[dict]) -> str:
        if not conflicts:
            return ""
        lines = []
        for c in conflicts[:8]:
            a = c.get("chunk_a_label")
            b = c.get("chunk_b_label")
            conf = float(c.get("confidence", 0.0) or 0.0)
            lines.append(f"- [{a}] mâu thuẫn với [{b}] (confidence {conf:.2%}).")
        return (
            "\n\nCẢNH BÁO XUNG ĐỘT NGUỒN:\n"
            + "\n".join(lines)
            + "\nKhi trả lời, hãy nêu rõ các nguồn đang mâu thuẫn và không tự ý trộn chúng thành một kết luận duy nhất."
        )

    def _synthesize(
        self,
        question,
        history,
        context_chunks,
        steps,
        route,
        trace,
        emit,
        partial_warning: bool = False,
        regenerate_reason: Optional[str] = None,
        iterations: int = 0,
        conflicts: Optional[list[dict]] = None,
        conflict_check: Optional[dict] = None,
        summary: Optional[str] = None,
    ):
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

        notes += self._format_conflict_notes(conflicts or [])

        if regenerate_reason:
            system = prompts.REGENERATE_SYSTEM.format(verify_reason=regenerate_reason)
        else:
            system = prompts.ANSWER_SYSTEM

        warning = ""
        if partial_warning and not regenerate_reason:
            warning = "\n\n⚠️ Lưu ý: Thông tin có thể chưa đầy đủ cho một số khía cạnh của câu hỏi. Chỉ trả lời dựa trên ngữ cảnh có sẵn."

        user_msg = (
            f"NGỮ CẢNH:\n{ctx}{notes}{warning}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo đúng quy tắc, trích dẫn bằng [số] tương ứng với các đoạn ngữ cảnh."
        )
        msgs = [{"role": "system", "content": system}]
        msgs.extend(self._history_msgs(history, summary))
        msgs.append({"role": "user", "content": user_msg})

        yield emit("thinking", {"node": "answer"})
        answer_parts: list[str] = []
        for tok in self.llm.stream(msgs, fast=regenerate_reason is not None, node="regenerate" if regenerate_reason else "answer"):
            answer_parts.append(tok)
            yield emit("token", {"text": tok})
        answer = "".join(answer_parts).strip()

        if not regenerate_reason:
            cited_labels = {int(m) for m in _CITE_RE.findall(answer)}
            citations = self._build_citations(context_chunks, cited_labels)
            trace_snapshot = list(trace)
            yield emit(
                "final",
                {
                    "answer": answer,
                    "citations": citations,
                    "trace": trace_snapshot,
                    "usage": self.llm.usage,
                    "route": route,
                    "partial": partial_warning,
                    "iterations": iterations,
                    "conflicts": conflicts or [],
                    "conflict_check": conflict_check,
                },
            )
        return answer

    def _answer_chitchat(self, question, history, trace, emit, summary=None):
        system = (
            "Bạn là trợ lý hỏi đáp tài liệu tiếng Việt. Người dùng đang chào hỏi hoặc "
            "hỏi về khả năng của bạn. Trả lời ngắn gọn, thân thiện và mời họ nạp tài liệu "
            "(PDF/URL/văn bản) rồi đặt câu hỏi. Không bịa thông tin về tài liệu."
        )
        msgs = [{"role": "system", "content": system}]
        msgs.extend(self._history_msgs(history, summary))
        msgs.append({"role": "user", "content": question})
        yield emit("thinking", {"node": "answer"})
        parts = []
        for tok in self.llm.stream(msgs, fast=True, node="answer"):
            parts.append(tok)
            yield emit("token", {"text": tok})
        answer = "".join(parts).strip()
        trace_snapshot = list(trace)
        yield emit(
            "final",
            {"answer": answer, "citations": [], "trace": trace_snapshot, "usage": self.llm.usage, "route": "no_retrieval", "partial": False, "iterations": 0},
        )

    def _emit_not_found(self, trace, emit):
        yield emit("thinking", {"node": "answer"})
        answer = "Không tìm thấy thông tin trong tài liệu."
        yield emit("token", {"text": answer})
        trace_snapshot = list(trace)
        yield emit(
            "final",
            {"answer": answer, "citations": [], "trace": trace_snapshot, "usage": self.llm.usage, "route": "not_found", "partial": False, "iterations": 0},
        )
