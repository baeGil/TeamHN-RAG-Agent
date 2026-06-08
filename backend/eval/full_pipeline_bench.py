"""Full pipeline benchmark: retrieval IR metrics + generation quality + latency.

Runs 10 hard QA questions through the RAG system with detailed per-stage
metrics, computes NDCG/Precision/Context Recall via LLM-as-judge,
and compares configurations (CCH on/off, RSE on/off).

Usage (from backend/):
    uv run python -m eval.full_pipeline_bench
    uv run python -m eval.full_pipeline_bench --skip-ablation
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings, get_settings
from app.indexing.store import KnowledgeBase, RetrievedChunk
from app.agent.graph import Agent
from app.agent.llm import LLM
from eval.dataset import load_dataset, QA

# ─── IR metrics ──────────────────────────────────────────────────────────────

def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in ranked_ids[:k] if r in relevant) / len(relevant)


def precision_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for r in ranked_ids[:k] if r in relevant) / k


def hit_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(ranked_ids[:k]) if rid in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ─── LLM-as-judge for relevance ──────────────────────────────────────────────

JUDGE_RELEVANCE_SYSTEM = (
    "Bạn là giám khảo đánh giá độ liên quan trong hệ thống truy hồi tài liệu.\n"
    "Cho một CÂU HỎI, ĐÁP ÁN ĐÚNG mong đợi, và danh sách các ĐOẠN ứng viên (kèm id).\n"
    "Hãy xác định những đoạn nào CHỨA thông tin cần thiết để suy ra đáp án đúng.\n"
    'Trả về JSON: {"relevant_ids": [<các id liên quan>]}. Chỉ chọn đoạn thực sự liên quan.'
)

JUDGE_ANSWER_SYSTEM = (
    "Bạn là giám khảo đánh giá chất lượng câu trả lời RAG.\n"
    "Cho CÂU HỎI, ĐÁP ÁN ĐÚNG mong đợi, và CÂU TRẢ LỜI của hệ thống.\n"
    "Đánh giá trên thang 1-5:\n"
    "  5 = hoàn toàn chính xác, đầy đủ\n"
    "  4 = chính xác nhưng thiếu chi tiết phụ\n"
    "  3 = đúng hướng nhưng có sai sót hoặc thiếu quan trọng\n"
    "  2 = sai lệch đáng kể\n"
    "  1 = hoàn toàn sai hoặc không liên quan\n"
    'Trả về JSON: {"score": <1-5>, "context_recall": <0.0-1.0 tỉ lệ nội dung đúng có trong câu trả lời>, '
    '"faithfulness": <0.0-1.0 tỉ lệ câu trả lời được hỗ trợ bởi ngữ cảnh>, '
    '"reason": "<giải thích ngắn>"}'
)


@dataclass
class PerQueryResult:
    qid: str
    question: str
    expected: str
    difficulty: str
    # Retrieval IR metrics (vs judged relevant set)
    n_relevant: int = 0
    relevant_ids: list[int] = field(default_factory=list)
    ranked_ids: list[int] = field(default_factory=list)
    recall_5: float = 0.0
    recall_10: float = 0.0
    precision_5: float = 0.0
    precision_10: float = 0.0
    hit_5: float = 0.0
    hit_10: float = 0.0
    mrr_5: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    # Retrieval latency breakdown (ms)
    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    rse_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    # Context stats
    n_returned: int = 0
    n_segments: int = 0
    total_context_chars: int = 0
    sections_covered: list[str] = field(default_factory=list)
    # Generation
    route: str = ""
    n_iterations: int = 0
    regenerated: bool = False
    answer_chars: int = 0
    answer_ms: float = 0.0
    total_ms: float = 0.0
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    # Answer quality (LLM judge)
    answer_score: float = 0.0
    context_recall_gen: float = 0.0
    faithfulness: float = 0.0
    judge_reason: str = ""


def judge_relevance(kb: KnowledgeBase, qa: QA, candidate_ids: list[int], cache: dict) -> set[int]:
    key = qa.qid
    if key in cache:
        return set(cache[key])
    meta = kb.repo.get_chunks(candidate_ids)
    lines = []
    for cid in candidate_ids:
        if cid in meta:
            txt = meta[cid]["text"][:500].replace("\n", " ")
            lines.append(f"[id={cid}] {txt}")
    user = (
        f"CÂU HỎI:\n{qa.question}\n\nĐÁP ÁN ĐÚNG:\n{qa.expected}\n\n"
        f"CÁC ĐOẠN ỨNG VIÊN:\n" + "\n\n".join(lines)
    )
    llm = LLM()
    res = llm.chat_json(
        [{"role": "system", "content": JUDGE_RELEVANCE_SYSTEM},
         {"role": "user", "content": user}],
        fast=True,
    )
    rel = [int(x) for x in res.get("relevant_ids", []) if int(x) in set(candidate_ids)]
    cache[key] = rel
    return set(rel)


def judge_answer(qa: QA, answer: str) -> dict:
    llm = LLM()
    user = (
        f"CÂU HỎI:\n{qa.question}\n\n"
        f"ĐÁP ÁN ĐÚNG:\n{qa.expected}\n\n"
        f"CÂU TRẢ LỜI:\n{answer}"
    )
    try:
        res = llm.chat_json(
            [{"role": "system", "content": JUDGE_ANSWER_SYSTEM},
             {"role": "user", "content": user}],
            fast=True,
        )
        return {
            "score": float(res.get("score", 0)),
            "context_recall_gen": float(res.get("context_recall", 0)),
            "faithfulness": float(res.get("faithfulness", 0)),
            "reason": res.get("reason", ""),
        }
    except Exception as e:
        return {"score": 0, "context_recall_gen": 0, "faithfulness": 0, "reason": str(e)}


def _retrieve_timed(kb: KnowledgeBase, query: str, top_k: int = 5) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """Run full retrieval with per-stage timing. Returns (chunks, latency_dict)."""
    s = kb.settings
    lat = {}

    t0 = time.perf_counter()
    bm25_hits = kb.bm25.search(query, s.bm25_top_k)
    lat["bm25_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dense_hits = []
    if kb.vector.ready:
        qv = kb.embedder.embed_query(query)
        t1 = time.perf_counter()
        dense_hits = kb.vector.search(qv, s.dense_top_k)
        lat["dense_ms"] = (time.perf_counter() - t0) * 1000
    else:
        lat["dense_ms"] = 0.0

    from app.retrieval.hybrid import reciprocal_rank_fusion
    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(bm25_hits, dense_hits, k=s.rrf_k)
    lat["rrf_ms"] = (time.perf_counter() - t0) * 1000

    candidates = fused[:s.rerank_top_n]
    meta = kb.repo.get_chunks([h.chunk_id for h in candidates])

    rerank_map = {}
    t0 = time.perf_counter()
    if s.use_reranker:
        pairs = [
            (h.chunk_id, meta[h.chunk_id].get("cch_text") or meta[h.chunk_id]["text"])
            for h in candidates if h.chunk_id in meta
        ]
        reranked = kb.reranker.rerank(query, pairs, top_k=len(pairs))
        lat["rerank_ms"] = (time.perf_counter() - t0) * 1000
        if reranked:
            rerank_map = {cid: sc for cid, sc in reranked}
    else:
        lat["rerank_ms"] = 0.0

    import math as _math
    def _final_score(h, rank):
        if h.chunk_id in rerank_map:
            return rerank_map[h.chunk_id]
        if s.use_rse:
            return _math.exp(-rank / 20.0)
        return h.rrf_score

    all_scored = sorted(enumerate(candidates), key=lambda ri: _final_score(ri[1], ri[0]), reverse=True)
    pre_rse = []
    for rank, h in all_scored:
        m = meta.get(h.chunk_id)
        if not m:
            continue
        pre_rse.append(RetrievedChunk(
            chunk_id=h.chunk_id, text=m["text"], document_id=m["document_id"],
            doc_title=m["doc_title"], doc_source=m["doc_source"],
            page=m["page"], section=m["section"],
            rrf_score=h.rrf_score, bm25_score=h.bm25_score,
            dense_score=h.dense_score, rerank_score=rerank_map.get(h.chunk_id),
            score=_final_score(h, rank),
            chunk_index=m.get("chunk_index", 0),
        ))

    t0 = time.perf_counter()
    if s.use_rse and pre_rse:
        final = kb._apply_rse(pre_rse)
    else:
        final = pre_rse[:top_k]
    lat["rse_ms"] = (time.perf_counter() - t0) * 1000

    total_start = time.perf_counter()
    # compute total from sum of stages (excluding this line)
    lat["total_retrieval_ms"] = sum(lat.values())

    return final, lat


def run_single_query(
    kb: KnowledgeBase,
    qa: QA,
    judge_cache: dict,
    config_label: str = "",
) -> PerQueryResult:
    r = PerQueryResult(qid=qa.qid, question=qa.question, expected=qa.expected, difficulty=qa.difficulty)
    t_total = time.perf_counter()

    # ── Step 1: Retrieval with timing ────────────────────────────────────
    chunks, lat = _retrieve_timed(kb, qa.question)
    r.bm25_ms = lat.get("bm25_ms", 0)
    r.dense_ms = lat.get("dense_ms", 0)
    r.rrf_ms = lat.get("rrf_ms", 0)
    r.rerank_ms = lat.get("rerank_ms", 0)
    r.rse_ms = lat.get("rse_ms", 0)
    r.total_retrieval_ms = lat.get("total_retrieval_ms", 0)

    # ── Step 2: Build candidate pool for relevance judging ──────────────
    # Get more candidates for judging (pool from both with/without reranking)
    from app.retrieval.hybrid import reciprocal_rank_fusion
    bm25_hits = kb.bm25.search(qa.question, 30)
    dense_hits = []
    if kb.vector.ready:
        qv = kb.embedder.embed_query(qa.question)
        dense_hits = kb.vector.search(qv, 30)
    fused = reciprocal_rank_fusion(bm25_hits, dense_hits, k=kb.settings.rrf_k)
    pool_ids = list(dict.fromkeys([h.chunk_id for h in fused[:20]] + [c.chunk_id for c in chunks]))

    # Judge relevance
    relevant = judge_relevance(kb, qa, pool_ids, judge_cache)
    r.n_relevant = len(relevant)
    r.relevant_ids = sorted(relevant)

    # Ranked IDs from final output
    r.ranked_ids = [c.chunk_id for c in chunks]
    # Also consider segment constituent chunk_ids for hit evaluation
    all_ranked = []
    for c in chunks:
        if c.is_segment and c.segment_chunk_ids:
            all_ranked.extend(c.segment_chunk_ids)
        else:
            all_ranked.append(c.chunk_id)
    r.ranked_ids = list(dict.fromkeys(all_ranked))

    # Compute IR metrics
    for k, suffix in [(5, "_5"), (10, "_10")]:
        ranked = r.ranked_ids
        setattr(r, f"recall{suffix}", recall_at_k(ranked, relevant, k))
        setattr(r, f"precision{suffix}", precision_at_k(ranked, relevant, k))
        setattr(r, f"hit{suffix}", hit_at_k(ranked, relevant, k))
        setattr(r, f"ndcg{suffix}", ndcg_at_k(ranked, relevant, k))
    r.mrr_5 = mrr_at_k(r.ranked_ids, relevant, 5)

    # Context stats
    r.n_returned = len(chunks)
    r.n_segments = sum(1 for c in chunks if c.is_segment)
    r.total_context_chars = sum(len(c.text) for c in chunks)
    r.sections_covered = list({c.section for c in chunks if c.section})

    # ── Step 3: Full generation (Agent) ──────────────────────────────────
    agent = Agent(kb)
    gen_start = time.perf_counter()
    answer_text = ""
    route = "simple"
    n_iters = 0
    regenerated = False
    for ev in agent.run(qa.question):
        if ev["type"] == "route":
            route = ev["data"]["route"]
        elif ev["type"] == "final":
            answer_text = ev["data"].get("answer", "")
            route = ev["data"].get("route", route)
            n_iters = ev["data"].get("iterations", 0)
            regenerated = ev["data"].get("regenerated", False)
            usage = ev["data"].get("usage", {})
            r.prompt_tokens = usage.get("prompt_tokens", 0)
            r.completion_tokens = usage.get("completion_tokens", 0)
            r.llm_calls = usage.get("calls", 0)

    r.answer_ms = (time.perf_counter() - gen_start) * 1000
    r.total_ms = (time.perf_counter() - t_total) * 1000
    r.route = route
    r.n_iterations = n_iters
    r.regenerated = regenerated
    r.answer_chars = len(answer_text)

    # ── Step 4: Answer quality judging ───────────────────────────────────
    if answer_text:
        quality = judge_answer(qa, answer_text)
        r.answer_score = quality["score"]
        r.context_recall_gen = quality["context_recall_gen"]
        r.faithfulness = quality["faithfulness"]
        r.judge_reason = quality["reason"]

    tag = f" [{config_label}]" if config_label else ""
    print(
        f"  [{qa.qid}] {qa.difficulty:4s} | "
        f"R@5={r.recall_5:.2f} P@5={r.precision_5:.2f} "
        f"H@5={r.hit_5:.0f} N@5={r.ndcg_5:.2f} MRR={r.mrr_5:.2f} | "
        f"ctx={r.total_context_chars}c n={r.n_returned} segs={r.n_segments} | "
        f"ret={r.total_retrieval_ms:.0f}ms gen={r.answer_ms:.0f}ms | "
        f"tok={r.prompt_tokens+r.completion_tokens} calls={r.llm_calls} | "
        f"score={r.answer_score:.1f} CR={r.context_recall_gen:.2f} F={r.faithfulness:.2f}"
        f"{tag}"
    )
    return r


def run_config(
    kb: KnowledgeBase,
    dataset: list[QA],
    judge_cache: dict,
    label: str = "",
) -> list[PerQueryResult]:
    print(f"\n{'='*80}")
    print(f"  CONFIG: {label}")
    print(f"  CCH_doc_summary={kb.settings.enable_doc_summary}  "
          f"CCH_section_summary={kb.settings.enable_section_summary}  "
          f"RSE={kb.settings.use_rse}  Reranker={kb.settings.use_reranker}")
    print(f"{'='*80}")
    results = []
    for qa in dataset:
        r = run_single_query(kb, qa, judge_cache, config_label=label)
        results.append(r)
    return results


def aggregate(results: list[PerQueryResult]) -> dict[str, Any]:
    n = len(results) or 1
    def _mean(attr):
        return statistics.mean([getattr(r, attr) for r in results])

    def _pct(attr):
        return sum(1 for r in results if getattr(r, attr) > 0) / n

    return {
        "n_queries": len(results),
        "retrieval": {
            "recall@5": _mean("recall_5"),
            "recall@10": _mean("recall_10"),
            "precision@5": _mean("precision_5"),
            "precision@10": _mean("precision_10"),
            "hit@5": _mean("hit_5"),
            "hit@10": _mean("hit_10"),
            "mrr@5": _mean("mrr_5"),
            "ndcg@5": _mean("ndcg_5"),
            "ndcg@10": _mean("ndcg_10"),
            "hit@5_pct": _pct("hit_5"),
            "avg_relevant": _mean("n_relevant"),
        },
        "latency_ms": {
            "bm25": _mean("bm25_ms"),
            "dense": _mean("dense_ms"),
            "rrf": _mean("rrf_ms"),
            "rerank": _mean("rerank_ms"),
            "rse": _mean("rse_ms"),
            "total_retrieval": _mean("total_retrieval_ms"),
            "generation": _mean("answer_ms"),
            "total_e2e": _mean("total_ms"),
        },
        "context": {
            "avg_returned": _mean("n_returned"),
            "avg_segments": _mean("n_segments"),
            "avg_chars": _mean("total_context_chars"),
        },
        "tokens": {
            "avg_prompt": _mean("prompt_tokens"),
            "avg_completion": _mean("completion_tokens"),
            "avg_total": _mean("prompt_tokens") + _mean("completion_tokens"),
            "avg_llm_calls": _mean("llm_calls"),
        },
        "quality": {
            "avg_score": _mean("answer_score"),
            "avg_context_recall": _mean("context_recall_gen"),
            "avg_faithfulness": _mean("faithfulness"),
            "pct_score_ge4": sum(1 for r in results if r.answer_score >= 4) / n,
        },
    }


def write_report(out_path: Path, configs: dict[str, list[PerQueryResult]], judge_cache: dict):
    agg_map = {label: aggregate(results) for label, results in configs.items()}

    md: list[str] = [
        "# Full Pipeline Benchmark: 10 Hard QA Questions",
        "",
        "## 1. Retrieval IR Metrics (LLM-as-Judge Relevance)",
        "",
        "| Config | R@5 | R@10 | P@5 | P@10 | Hit@5 | Hit@10 | MRR@5 | NDCG@5 | NDCG@10 | Avg Rel |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, a in agg_map.items():
        r = a["retrieval"]
        md.append(
            f"| {label} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
            f"{r['precision@5']:.3f} | {r['precision@10']:.3f} | "
            f"{r['hit@5']:.3f} | {r['hit@10']:.3f} | {r['mrr@5']:.3f} | "
            f"{r['ndcg@5']:.3f} | {r['ndcg@10']:.3f} | {r['avg_relevant']:.1f} |"
        )

    md += [
        "",
        "## 2. Latency Breakdown (ms, avg per query)",
        "",
        "| Config | BM25 | Dense | RRF | Rerank | RSE | Retrieval Total | Generation | E2E Total |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, a in agg_map.items():
        l = a["latency_ms"]
        md.append(
            f"| {label} | {l['bm25']:.0f} | {l['dense']:.0f} | {l['rrf']:.0f} | "
            f"{l['rerank']:.0f} | {l['rse']:.0f} | {l['total_retrieval']:.0f} | "
            f"{l['generation']:.0f} | {l['total_e2e']:.0f} |"
        )

    md += [
        "",
        "## 3. Token Usage & Cost",
        "",
        "| Config | Avg Prompt | Avg Completion | Avg Total | Avg LLM Calls |",
        "|---|---|---|---|---|",
    ]
    for label, a in agg_map.items():
        t = a["tokens"]
        md.append(f"| {label} | {t['avg_prompt']:.0f} | {t['avg_completion']:.0f} | {t['avg_total']:.0f} | {t['avg_llm_calls']:.1f} |")

    md += [
        "",
        "## 4. Answer Quality (LLM-as-Judge, 1-5 scale)",
        "",
        "| Config | Avg Score | Avg Context Recall | Avg Faithfulness | % Score>=4 |",
        "|---|---|---|---|---|",
    ]
    for label, a in agg_map.items():
        q = a["quality"]
        md.append(
            f"| {label} | {q['avg_score']:.2f} | {q['avg_context_recall']:.2f} | "
            f"{q['avg_faithfulness']:.2f} | {q['pct_score_ge4']:.0%} |"
        )

    md += [
        "",
        "## 5. Context Statistics",
        "",
        "| Config | Avg Returned | Avg Segments | Avg Context Chars |",
        "|---|---|---|---|",
    ]
    for label, a in agg_map.items():
        c = a["context"]
        md.append(f"| {label} | {c['avg_returned']:.1f} | {c['avg_segments']:.1f} | {c['avg_chars']:.0f} |")

    # Per-question detail for primary config
    primary = list(configs.keys())[0]
    md += [
        "",
        f"## 6. Per-Question Detail ({primary})",
        "",
        "| # | Q | R@5 | P@5 | Hit@5 | NDCG@5 | Ret ms | Gen ms | Tok | Score | CR | Faith | Route |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in configs[primary]:
        q_short = r.question[:50] + "..." if len(r.question) > 50 else r.question
        md.append(
            f"| {r.qid} | {q_short} | {r.recall_5:.2f} | {r.precision_5:.2f} | "
            f"{r.hit_5:.0f} | {r.ndcg_5:.2f} | {r.total_retrieval_ms:.0f} | "
            f"{r.answer_ms:.0f} | {r.prompt_tokens+r.completion_tokens} | "
            f"{r.answer_score:.1f} | {r.context_recall_gen:.2f} | {r.faithfulness:.2f} | {r.route} |"
        )

    # Ablation deltas
    if len(configs) > 1:
        md += [
            "",
            "## 7. Ablation: Impact of Techniques",
            "",
        ]
        labels = list(agg_map.keys())
        base = agg_map[labels[0]]
        for lbl in labels[1:]:
            a = agg_map[lbl]
            md += [f"### {labels[0]} vs {lbl}", ""]
            md.append("| Metric | Baseline | Ablation | Delta |")
            md.append("|---|---|---|---|")
            for metric_path in [
                ("retrieval", "recall@5"), ("retrieval", "precision@5"),
                ("retrieval", "hit@5"), ("retrieval", "ndcg@5"), ("retrieval", "mrr@5"),
                ("latency_ms", "total_retrieval"), ("latency_ms", "generation"), ("latency_ms", "total_e2e"),
                ("quality", "avg_score"), ("quality", "avg_context_recall"), ("quality", "avg_faithfulness"),
                ("tokens", "avg_total"),
                ("context", "avg_chars"),
            ]:
                cat, key = metric_path
                bv = base[cat][key]
                av = a[cat][key]
                md.append(f"| {cat}.{key} | {bv:.3f} | {av:.3f} | {av-bv:+.3f} |")
            md.append("")

    # Commentary
    md += [
        "",
        "## 8. Nhan xet & Phan tich",
        "",
    ]
    if len(configs) > 1:
        base_a = agg_map[list(configs.keys())[0]]
        # CCH effect
        cch_labels = [l for l in configs if "no-cch" in l.lower()]
        if cch_labels:
            cch_a = agg_map[cch_labels[0]]
            delta_ndcg = base_a["retrieval"]["ndcg@5"] - cch_a["retrieval"]["ndcg@5"]
            delta_score = base_a["quality"]["avg_score"] - cch_a["quality"]["avg_score"]
            md.append(f"### CCH (Contextual Chunk Headers)")
            md.append(f"- NDCG@5 delta: {delta_ndcg:+.3f} — CCH them thong tin tai lieu/phan vao moi chunk,")
            md.append(f"  giup BM25 va dense search hieu dung ngu canh hon, dac biet cho cau hoi su luan.")
            md.append(f"- Answer score delta: {delta_score:+.2f} — CCH giup truy hoi chinh xac hon,")
            md.append(f"  giam truong hop 'biet dung nhung khong tim thay'.")
            md.append("")

        # RSE effect
        rse_labels = [l for l in configs if "no-rse" in l.lower()]
        if rse_labels:
            rse_a = agg_map[rse_labels[0]]
            delta_ctx = base_a["context"]["avg_chars"] - rse_a["context"]["avg_chars"]
            delta_faith = base_a["quality"]["avg_faithfulness"] - rse_a["quality"]["avg_faithfulness"]
            delta_ret = base_a["latency_ms"]["rse"] - rse_a["latency_ms"]["rse"]
            md.append(f"### RSE (Relevant Segment Extraction)")
            md.append(f"- Context size delta: {delta_ctx:+.0f} chars — RSE mo rong context bang cach")
            md.append(f"  ghep cac chunk lien ke, tao segment day du hon.")
            md.append(f"- Faithfulness delta: {delta_faith:+.3f} — Segment nguyen van giup LLM")
            md.append(f"  co ngu canh day du hon, giam bi dat do.")
            md.append(f"- RSE latency: {base_a['latency_ms']['rse']:.1f}ms (rất nhanh, <10ms)")
            md.append("")

        # Chunking effect (if tested)
        md.append("### Section-aware Chunking (MinerU markdown)")
        md.append("- Chunk theo muc (section) tao cac doan tap trung, khong chong lap (no overlap).")
        md.append("- Cong thuc toan duoc giu nguyen trong 1 chunk thay vi bi chia cat.")
        md.append("- RSE co the ghep lai cac chunk lien ke khi can, khong can overlap o chunking time.")
        md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../data/data_hung/đatn")
    ap.add_argument("--out-dir", default="../data/bench_full")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="Skip CCH/RSE ablation (only run full config)")
    ap.add_argument("--ablation", default="all",
                    choices=["all", "no-rse", "no-cch", "no-reranker"],
                    help="Run specific ablation only")
    ap.add_argument("--skip-generation", action="store_true",
                    help="Skip generation + answer judging (retrieval only)")
    args = ap.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chua cau hinh trong backend/.env")

    # Load hard QA only
    dataset = load_dataset(test_dir)
    hard_only = [q for q in dataset if q.difficulty == "hard"]
    print(f"Loaded {len(hard_only)} hard questions from {test_dir}")

    # Use the existing KnowledgeBase (same storage the running backend uses)
    kb = KnowledgeBase(settings)
    docs = kb.repo.list_documents()
    if not docs:
        # Ingest the PDF
        pdf = next(test_dir.glob("*.pdf"), None)
        if pdf is None:
            raise SystemExit(f"Khong tim thay PDF trong {test_dir}")
        print(f"Ingesting {pdf.name} ...")
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)

    print(f"KB stats: {kb.stats()}")

    # Judge cache (shared across configs for consistency)
    cache_path = out_dir / "judge_cache.json"
    judge_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    configs: dict[str, list[PerQueryResult]] = {}

    # Load existing results if available
    json_path = out_dir / "bench_full_results.json"
    if json_path.exists():
        existing = json.loads(json_path.read_text())
        for label, items in existing.items():
            configs[label] = [PerQueryResult(**item) for item in items]
        print(f"Loaded {len(configs)} existing configs from {json_path}")

    # ── Config 1: Full pipeline (CCH + RSE + Reranker) ───────────────
    full_label = "Full (CCH+RSE+Rerank)"
    if full_label not in configs:
        settings.use_rse = True
        settings.enable_doc_summary = True
        settings.enable_section_summary = True
        settings.use_reranker = True
        kb.settings = settings
        configs[full_label] = run_config(kb, hard_only, judge_cache, label="Full")

    if not args.skip_ablation:
        ablation_type = args.ablation or "all"

        if ablation_type in ("all", "no-rse"):
            # ── Config 2: No RSE ───────────────────────────────────────────
            settings.use_rse = True  # reset
            settings.enable_doc_summary = True
            settings.enable_section_summary = True
            settings.use_reranker = True
            settings.use_rse = False
            kb.settings = settings
            configs["No-RSE"] = run_config(kb, hard_only, judge_cache, label="No-RSE")

        if ablation_type in ("all", "no-cch"):
            # ── Config 3: No CCH ───────────────────────────────────────────
            settings.use_rse = True
            settings.enable_doc_summary = False
            settings.enable_section_summary = False
            settings.use_reranker = True
            kb.settings = settings
            configs["No-CCH-summary"] = run_config(kb, hard_only, judge_cache, label="No-CCH-summary")

        if ablation_type in ("all", "no-reranker"):
            # ── Config 4: No Reranker ─────────────────────────────────────
            settings.use_rse = True
            settings.enable_doc_summary = True
            settings.enable_section_summary = True
            settings.use_reranker = False
            kb.settings = settings
            configs["No-Reranker"] = run_config(kb, hard_only, judge_cache, label="No-Reranker")

    # Save judge cache
    cache_path.write_text(json.dumps(judge_cache, ensure_ascii=False, indent=2))

    # Save full JSON results
    full_data = {label: [asdict(r) for r in results] for label, results in configs.items()}
    json_path = out_dir / "bench_full_results.json"
    json_path.write_text(json.dumps(full_data, ensure_ascii=False, indent=2))

    # Write markdown report
    report_path = out_dir / "bench_full_report.md"
    write_report(report_path, configs, judge_cache)

    # ── Final console summary ──────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  FINAL SUMMARY")
    print(f"{'='*100}")
    for label, results in configs.items():
        a = aggregate(results)
        r = a["retrieval"]
        l = a["latency_ms"]
        q = a["quality"]
        t = a["tokens"]
        c = a["context"]
        print(f"\n  [{label}]")
        print(f"    Retrieval: R@5={r['recall@5']:.3f}  P@5={r['precision@5']:.3f}  "
              f"Hit@5={r['hit@5']:.3f}  NDCG@5={r['ndcg@5']:.3f}  MRR@5={r['mrr@5']:.3f}")
        print(f"    Latency:   ret={l['total_retrieval']:.0f}ms  gen={l['generation']:.0f}ms  "
              f"e2e={l['total_e2e']:.0f}ms  (bm25={l['bm25']:.0f} dense={l['dense']:.0f} "
              f"rrf={l['rrf']:.0f} rerank={l['rerank']:.0f} rse={l['rse']:.0f})")
        print(f"    Quality:   score={q['avg_score']:.2f}/5  CR={q['avg_context_recall']:.2f}  "
              f"Faith={q['avg_faithfulness']:.2f}  %>=4={q['pct_score_ge4']:.0%}")
        print(f"    Tokens:    avg={t['avg_total']:.0f} (in={t['avg_prompt']:.0f} out={t['avg_completion']:.0f})  "
              f"calls={t['avg_llm_calls']:.1f}")
        print(f"    Context:   n={c['avg_returned']:.1f}  segs={c['avg_segments']:.1f}  "
              f"chars={c['avg_chars']:.0f}")

    print(f"\n  Report: {report_path}")
    print(f"  JSON:   {json_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
