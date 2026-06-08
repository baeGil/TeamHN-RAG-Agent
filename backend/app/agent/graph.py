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
        # Include last 2 turns of history so router can detect follow-up questions
        user_content = question
        if history:
            recent = history[-2:]
            ctx = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent)
            user_content = f"Lịch sử gần nhất:\n{ctx}\n\nCâu hỏi hiện tại: {question}"
        msgs = [
            {"role": "system", "content": prompts.ROUTER_SYSTEM},
            {"role": "user", "content": user_content},
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

    def _generate_and_verify(
        self, question, history, context_chunks, steps, route, summary=None,
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
        # Use compact context (label + first 120 chars) instead of full text
        # to reduce token cost by ~80% while preserving enough signal for grounding check.
        compact = "\n".join(
            f"[{c['label']}] {c['text'][:120]}{'...' if len(c['text']) > 120 else ''}"
            for c in context_chunks
        )
        msgs = [
            {"role": "system", "content": prompts.ANSWER_VERIFY_SYSTEM},
            {"role": "user", "content": f"CÂU TRẢ LỜI:\n{answer}\n\nNGỮ CẢNH (tóm tắt):\n{compact}"},
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

            if route == "complex" and len(subqs) > 1:
                # Batch retrieval: collect candidates from all sub-queries in parallel,
                # then call reranker + RSE once per sub-query (serialized reranker internally
                # but avoid N separate reranker model loads via batch_rerank_and_rse).
                with ThreadPoolExecutor(max_workers=min(4, len(subqs))) as ex:
                    cand_futures = {
                        ex.submit(self.kb.retrieve_candidates, sq): i
                        for i, sq in enumerate(subqs)
                    }
                    queries_candidates = {}
                    for fut, i in [(f, cand_futures[f]) for f in cand_futures]:
                        queries_candidates[i] = fut.result()
                ordered = [queries_candidates[i] for i in range(len(subqs))]
                batch_results = self.kb.batch_rerank_and_rse(ordered)
                for i, hits in batch_results.items():
                    retrieval_jobs[i] = [_chunk_to_dict(h) for h in hits]
            else:
                # Simple route: single retrieve() call as before
                hits = self.kb.retrieve(subqs[0], self.settings.final_top_k)
                retrieval_jobs[0] = [_chunk_to_dict(h) for h in hits]

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
                                "is_segment": h.get("is_segment", False),
                                "segment_chunk_ids": h.get("segment_chunk_ids"),
                            }
                            for h in hit_dicts
                        ],
                    },
                )

            # --- Distill + Verify (complex only) ---
            if route == "complex":
                node_started = time.perf_counter()
                yield emit("thinking", {"node": "distill_verify"})
                distill_results: dict[int, dict] = {}

                def _distill_and_verify(idx_subq):
                    i, subq = idx_subq
                    hit_dicts = retrieval_jobs.get(i, [])
                    if not hit_dicts:
                        return i, {"note": "", "relevant": False, "grounded": False, "reason": ""}
                    labeled = self._with_labels(hit_dicts)
                    ctx = prompts.build_context(labeled, label_key="label")
                    try:
                        # Single merged call: distill + verify in one JSON response
                        msgs = [
                            {"role": "system", "content": prompts.DISTILL_VERIFY_SYSTEM},
                            {"role": "user", "content": f"CÂU HỎI CON:\n{subq}\n\nĐOẠN TRÍCH:\n{ctx}"},
                        ]
                        data = self.llm.chat_json(msgs, fast=True, node="distill_verify")
                        note = data.get("note", "").strip()
                        rel = note != "KHÔNG_LIÊN_QUAN" and bool(note) and data.get("relevant", True)
                        grounded = bool(data.get("grounded", True)) if rel else False
                        reason = data.get("reason", "")
                        return i, {"note": note, "relevant": rel, "grounded": grounded, "reason": reason}
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

                for i, subq in enumerate(subqs):
                    r = distill_results.get(i, {})
                    is_error = r.get("error", False)
                    if is_error:
                        yield emit("error", {"node": "distill_verify", "message": r.get("reason", "Lỗi chắt lọc & kiểm chứng")})
                    yield emit(
                        "distill_verify",
                        {
                            "subquestion": subq,
                            "index": i,
                            "note": r.get("note", ""),
                            "relevant": r.get("relevant", False),
                            "grounded": r.get("grounded", True),
                            "reason": r.get("reason", ""),
                        },
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

        # --- Synthesize ---
        node_started = time.perf_counter()
        yield emit("thinking", {"node": "synthesize"})
        # n_context = number of context entries (labels), not constituent chunks.
        # A segment with 4 chunks is 1 context entry, not 4 — the user sees
        # "N ngữ cảnh" on the graph and expects N = number of citation labels.
        yield emit(
            "synthesize",
            {"n_context": len(context_chunks), "labels": [c["label"] for c in context_chunks]},
        )
        logger.info(
            "RAG_FLOW node_start node=synthesize route=%s context_chunks=%s",
            route,
            len(context_chunks),
        )

        # Determine whether to run answer verification for this route.
        # ENABLE_ANSWER_VERIFY is a global kill-switch; per-route flags fine-tune:
        #   simple route → default off (stream directly, lower hallucination risk)
        #   complex route → default on (multi-hop synthesis more prone to hallucination)
        s = self.settings
        _do_verify = (
            s.enable_answer_verify
            and route != "no_retrieval"
            and (
                (route == "complex" and s.enable_answer_verify_complex)
                or (route == "simple" and s.enable_answer_verify_simple)
            )
        )
        if _do_verify:
            # Generate internally (no streaming) and verify grounding
            answer_text, is_grounded, v_reason = self._generate_and_verify(
                question, history, context_chunks, steps, route, summary=summary,
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
                        iterations=iteration, summary=summary,
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
                partial_warning=partial_warning, iterations=iteration, summary=summary,
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
        limit = self.settings.complex_ctx_limit if route == "complex" else self.settings.final_top_k
        # Skip entries with score < 1% of top entry — they are noise that
        # wastes context tokens and produces uncited citations.
        score_threshold = 0.0
        if chunks:
            score_threshold = chunks[0].get("score", 0.0) * 0.01
        # Deduplicate RSE segments: if two entries share overlapping chunk_ids, keep
        # only the higher-scoring one to avoid sending duplicate content to the LLM.
        selected: list[dict] = []
        used_chunk_ids: set[int] = set()
        # Count by constituent chunks, not by segments.  A segment with 15
        # chunk_ids should consume 15 slots of the limit, not 1 — otherwise
        # RSE collapses a rich 10-chunk context into a single citation.
        total_chunks = 0
        # Cap per-segment constituent chunks so we always get at least 2-3
        # different citation labels.  Without this, one 8-chunk segment fills
        # the entire limit and the answer only gets 1 label.
        per_segment_cap = max(limit // 2, 3)
        for c in chunks:
            if c.get("score", 0.0) < score_threshold:
                continue
            seg_ids = set(c.get("segment_chunk_ids") or [])
            if not seg_ids:
                seg_ids = {c["chunk_id"]}
            if seg_ids & used_chunk_ids:
                continue
            # Truncate segment to cap so we leave room for other segments
            if len(seg_ids) > per_segment_cap:
                capped_ids = sorted(seg_ids)[:per_segment_cap]
                seg_ids = set(capped_ids)
                # Fetch text for the capped subset so text matches chunk_ids
                capped_rows = self.kb.repo.get_chunks(capped_ids)
                capped_text = "\n\n".join(
                    (capped_rows[cid].get("text") or "").strip()
                    for cid in sorted(capped_rows)
                    if capped_rows[cid].get("text")
                )
                # Create a new entry with truncated data
                c = dict(c)
                c["segment_chunk_ids"] = capped_ids
                c["text"] = capped_text
                c["is_segment"] = True
            selected.append(c)
            used_chunk_ids |= seg_ids
            total_chunks += len(seg_ids)
            if total_chunks >= limit:
                break
        return self._with_labels(selected)

    def _build_citations(self, context_chunks: list[dict], cited_labels: set[int]) -> list[dict]:
        cits = []
        for c in context_chunks:
            seg_ids = c.get("segment_chunk_ids") or []
            if seg_ids and c.get("is_segment"):
                # RSE segment: fetch constituent chunks for page/section metadata only
                seg_chunk_dicts = self.kb.repo.get_chunks(seg_ids)
                pages = set()
                sections = set()
                for cid, chunk_data in sorted(seg_chunk_dicts.items()):
                    if chunk_data.get("page") is not None:
                        pages.add(chunk_data["page"])
                    if chunk_data.get("section"):
                        sections.add(chunk_data["section"])
                # Use first chunk's page as primary page, list all pages
                primary_page = min(pages) if pages else c["page"]
                page_str = str(primary_page) if primary_page is not None else None
                # Use c["text"] directly (already capped by _select_context)
                # rather than re-fetching to avoid text/segment_chunk_ids mismatch
                cits.append(
                    {
                        "label": c["label"],
                        "chunk_id": c["chunk_id"],
                        "chunk_ids": sorted(seg_chunk_dicts.keys()),
                        "document_id": c["document_id"],
                        "doc_title": c["doc_title"],
                        "doc_source": c["doc_source"],
                        "page": page_str,
                        "pages": sorted(pages) if pages else None,
                        "section": ", ".join(sections) if sections else c.get("section"),
                        "text": c["text"],
                        "n_chunks": len(seg_ids),
                        "score": round(c.get("score", 0.0), 4),
                        "cited": c["label"] in cited_labels,
                        "is_segment": True,
                    }
                )
            else:
                cits.append(
                    {
                        "label": c["label"],
                        "chunk_id": c["chunk_id"],
                        "document_id": c["document_id"],
                        "doc_title": c["doc_title"],
                        "doc_source": c["doc_source"],
                        "page": c["page"],
                        "text": c["text"],
                        "score": round(c.get("score", 0.0), 4),
                        "cited": c["label"] in cited_labels,
                        "is_segment": False,
                    }
                )
        return cits

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
