"""Quick test: 5 hard QA questions with local BGE reranker (not Jina).

Usage (from backend/):
    uv run python -m eval.test_bge_reranker
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force reload settings
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from app.config import Settings, get_settings
get_settings.cache_clear()

from app.indexing.store import KnowledgeBase, RetrievedChunk
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.agent.graph import Agent
from app.agent.llm import LLM
from eval.dataset import load_dataset, QA

# 5 representative hard questions (mix of formula, reasoning, multi-hop)
PICK_IDS = ["h2", "h5", "h7", "h8", "h10"]


def recall_at_k(ranked, relevant, k):
    if not relevant: return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / len(relevant)

def ndcg_at_k(ranked, relevant, k):
    if not relevant: return 0.0
    dcg = sum(1.0/math.log2(i+2) for i,r in enumerate(ranked[:k]) if r in relevant)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(relevant),k)))
    return dcg/idcg if idcg else 0.0

def hit_at_k(ranked, relevant, k):
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0

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
    'Trả về JSON: {"score": <1-5>, "context_recall": <0.0-1.0>, '
    '"faithfulness": <0.0-1.0>, "reason": "<giải thích ngắn>"}'
)


def judge_relevance(kb, qa, candidate_ids, cache):
    key = f"bge_{qa.qid}"
    if key in cache: return set(cache[key])
    meta = kb.repo.get_chunks(candidate_ids)
    lines = []
    for cid in candidate_ids:
        if cid in meta:
            txt = meta[cid]["text"][:500].replace("\n"," ")
            lines.append(f"[id={cid}] {txt}")
    user = f"CÂU HỎI:\n{qa.question}\n\nĐÁP ÁN ĐÚNG:\n{qa.expected}\n\nCÁC ĐOẠN ỨNG VIÊN:\n" + "\n\n".join(lines)
    llm = LLM()
    res = llm.chat_json([{"role":"system","content":JUDGE_RELEVANCE_SYSTEM},{"role":"user","content":user}], fast=True)
    rel = [int(x) for x in res.get("relevant_ids",[]) if int(x) in set(candidate_ids)]
    cache[key] = rel
    return set(rel)


def judge_answer(qa, answer):
    llm = LLM()
    user = f"CÂU HỎI:\n{qa.question}\n\nĐÁP ÁN ĐÚNG:\n{qa.expected}\n\nCÂU TRẢ LỜI:\n{answer}"
    try:
        res = llm.chat_json([{"role":"system","content":JUDGE_ANSWER_SYSTEM},{"role":"user","content":user}], fast=True)
        return float(res.get("score",0)), float(res.get("context_recall",0)), float(res.get("faithfulness",0)), res.get("reason","")
    except Exception as e:
        return 0, 0, 0, str(e)


def main():
    settings = get_settings()
    print(f"Reranker config: type={settings.reranker_type}  model={settings.reranker_model}")
    print(f"  use_reranker={settings.use_reranker}  jina_key={'yes' if settings.jina_api_key else 'no'}")
    print(f"  use_rse={settings.use_rse}  use_reranker={settings.use_reranker}")

    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY missing")

    test_dir = Path("../data/data_hung/đatn").resolve()
    dataset = load_dataset(test_dir)
    hard = [q for q in dataset if q.difficulty == "hard" and q.qid in PICK_IDS]
    print(f"Testing {len(hard)} hard questions: {[q.qid for q in hard]}")

    kb = KnowledgeBase(settings)
    docs = kb.repo.list_documents()
    if not docs:
        pdf = next(test_dir.glob("*.pdf"))
        print(f"Ingesting {pdf.name}...")
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)
    print(f"KB: {kb.stats()}")
    print(f"Reranker available: {kb.reranker.available}  type={type(kb.reranker).__name__}")

    # Test reranker directly first
    print("\n--- Reranker warmup ---")
    try:
        test_pairs = [(1, "Mô hình hóa di chuyển an toàn"), (2, "Công thức Risk(P)")]
        t0 = time.perf_counter()
        result = kb.reranker.rerank("Tại sao đường dài hơn ít phơi nhiễm hơn", test_pairs, top_k=2)
        warmup_ms = (time.perf_counter() - t0) * 1000
        print(f"  Warmup: {warmup_ms:.0f}ms  result={result}")
    except Exception as e:
        print(f"  Reranker FAILED: {e}")
        return

    cache_path = Path("../data/bench_full/judge_cache.json")
    judge_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    results = []
    for qa in hard:
        print(f"\n{'='*70}")
        print(f"  [{qa.qid}] {qa.question[:80]}...")
        t_total = time.perf_counter()

        # --- Retrieval with timing ---
        s = kb.settings
        t0 = time.perf_counter()
        bm25_hits = kb.bm25.search(qa.question, s.bm25_top_k)
        bm25_ms = (time.perf_counter()-t0)*1000

        t0 = time.perf_counter()
        qv = kb.embedder.embed_query(qa.question)
        dense_hits = kb.vector.search(qv, s.dense_top_k) if kb.vector.ready else []
        dense_ms = (time.perf_counter()-t0)*1000

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits, k=s.rrf_k)
        rrf_ms = (time.perf_counter()-t0)*1000

        candidates = fused[:s.rerank_top_n]
        meta = kb.repo.get_chunks([h.chunk_id for h in candidates])

        # --- Reranker (BGE local) ---
        rerank_map = {}
        t0 = time.perf_counter()
        if s.use_reranker:
            pairs = [(h.chunk_id, meta[h.chunk_id].get("cch_text") or meta[h.chunk_id]["text"])
                     for h in candidates if h.chunk_id in meta]
            reranked = kb.reranker.rerank(qa.question, pairs, top_k=len(pairs))
            rerank_ms = (time.perf_counter()-t0)*1000
            if reranked:
                rerank_map = {cid: sc for cid, sc in reranked}
                print(f"  Rerank: {rerank_ms:.0f}ms  top-3 scores: {[f'{sc:.4f}' for _,sc in reranked[:3]]}")
            else:
                print(f"  Rerank: {rerank_ms:.0f}ms  FAILED (returned None)")
                rerank_ms = 0
        else:
            rerank_ms = 0

        # Build pre-RSE
        def _score(h, rank):
            if h.chunk_id in rerank_map: return rerank_map[h.chunk_id]
            if s.use_rse: return math.exp(-rank/20.0)
            return h.rrf_score

        all_scored = sorted(enumerate(candidates), key=lambda ri: _score(ri[1],ri[0]), reverse=True)
        pre_rse = []
        for rank, h in all_scored:
            m = meta.get(h.chunk_id)
            if not m: continue
            pre_rse.append(RetrievedChunk(
                chunk_id=h.chunk_id, text=m["text"], document_id=m["document_id"],
                doc_title=m["doc_title"], doc_source=m["doc_source"],
                page=m["page"], section=m["section"],
                rrf_score=h.rrf_score, bm25_score=h.bm25_score,
                dense_score=h.dense_score, rerank_score=rerank_map.get(h.chunk_id),
                score=_score(h,rank), chunk_index=m.get("chunk_index",0),
            ))

        # --- RSE ---
        t0 = time.perf_counter()
        if s.use_rse and pre_rse:
            final_chunks = kb._apply_rse(pre_rse)
        else:
            final_chunks = pre_rse[:s.final_top_k]
        rse_ms = (time.perf_counter()-t0)*1000

        ret_total = (time.perf_counter()-t_total)*1000

        # IR metrics
        pool_ids = list(dict.fromkeys([h.chunk_id for h in fused[:20]] + [c.chunk_id for c in final_chunks]))
        relevant = judge_relevance(kb, qa, pool_ids, judge_cache)
        all_ranked = []
        for c in final_chunks:
            if c.is_segment and c.segment_chunk_ids: all_ranked.extend(c.segment_chunk_ids)
            else: all_ranked.append(c.chunk_id)
        all_ranked = list(dict.fromkeys(all_ranked))

        r5 = recall_at_k(all_ranked, relevant, 5)
        p5 = sum(1 for r in all_ranked[:5] if r in relevant)/5 if all_ranked[:5] else 0
        h5 = hit_at_k(all_ranked, relevant, 5)
        n5 = ndcg_at_k(all_ranked, relevant, 5)

        ctx_chars = sum(len(c.text) for c in final_chunks)
        n_segs = sum(1 for c in final_chunks if c.is_segment)

        print(f"  Retrieval: bm25={bm25_ms:.0f}ms dense={dense_ms:.0f}ms rrf={rrf_ms:.0f}ms "
              f"rerank={rerank_ms:.0f}ms rse={rse_ms:.0f}ms total={ret_total:.0f}ms")
        print(f"  IR: R@5={r5:.2f} P@5={p5:.2f} H@5={h5:.0f} N@5={n5:.2f} | "
              f"n_ret={len(final_chunks)} segs={n_segs} ctx={ctx_chars}c rel={len(relevant)}")

        # --- Generation ---
        agent = Agent(kb)
        gen_t0 = time.perf_counter()
        answer = ""
        route = ""
        iters = 0
        regen = False
        for ev in agent.run(qa.question):
            if ev["type"] == "route":
                route = ev["data"]["route"]
            elif ev["type"] == "final":
                answer = ev["data"].get("answer","")
                route = ev["data"].get("route",route)
                iters = ev["data"].get("iterations",0)
                regen = ev["data"].get("regenerated",False)
                usage = ev["data"].get("usage",{})
                ptok = usage.get("prompt_tokens",0)
                ctok = usage.get("completion_tokens",0)
                calls = usage.get("calls",0)
        gen_ms = (time.perf_counter()-gen_t0)*1000
        e2e_ms = (time.perf_counter()-t_total)*1000

        # Answer quality
        score, cr, faith, reason = judge_answer(qa, answer) if answer else (0,0,0,"no answer")

        print(f"  Generation: route={route} iters={iters} regen={regen} gen={gen_ms:.0f}ms")
        print(f"  Tokens: prompt={ptok} completion={ctok} calls={calls}")
        print(f"  Quality: score={score}/5 CR={cr:.2f} Faith={faith:.2f}")
        print(f"  E2E: {e2e_ms:.0f}ms")
        if answer:
            print(f"  Answer preview: {answer[:200]}...")

        results.append({
            "qid": qa.qid, "question": qa.question[:60],
            "rerank_ms": rerank_ms, "ret_ms": ret_total, "gen_ms": gen_ms, "e2e_ms": e2e_ms,
            "R@5": r5, "P@5": p5, "H@5": h5, "NDCG@5": n5,
            "n_relevant": len(relevant), "ctx_chars": ctx_chars, "n_segs": n_segs,
            "score": score, "CR": cr, "faithfulness": faith,
            "prompt_tokens": ptok, "completion_tokens": ctok, "llm_calls": calls,
            "route": route, "iters": iters, "regen": regen,
        })

    # Save judge cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(judge_cache, ensure_ascii=False, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY — BGE Reranker (local)")
    print(f"{'='*70}")
    print(f"  {'QID':>4} | {'Rerank':>7} | {'Ret':>6} | {'Gen':>6} | {'E2E':>6} | "
          f"{'R@5':>5} {'N@5':>5} | {'Score':>5} {'CR':>4} {'F':>4} | {'Tok':>6} {'Calls':>5}")
    for r in results:
        print(f"  {r['qid']:>4} | {r['rerank_ms']:>6.0f}ms | {r['ret_ms']:>5.0f}ms | "
              f"{r['gen_ms']:>5.0f}ms | {r['e2e_ms']:>5.0f}ms | "
              f"{r['R@5']:>5.2f} {r['NDCG@5']:>5.2f} | "
              f"{r['score']:>5.1f} {r['CR']:>4.2f} {r['faithfulness']:>4.2f} | "
              f"{r['prompt_tokens']+r['completion_tokens']:>6} {r['llm_calls']:>5}")

    avg_rerank = sum(r["rerank_ms"] for r in results)/len(results)
    avg_ret = sum(r["ret_ms"] for r in results)/len(results)
    avg_gen = sum(r["gen_ms"] for r in results)/len(results)
    avg_score = sum(r["score"] for r in results)/len(results)
    avg_cr = sum(r["CR"] for r in results)/len(results)
    avg_faith = sum(r["faithfulness"] for r in results)/len(results)
    print(f"\n  AVG  | {avg_rerank:>6.0f}ms | {avg_ret:>5.0f}ms | {avg_gen:>5.0f}ms | | "
          f"| {avg_score:>5.1f} {avg_cr:>4.2f} {avg_faith:>4.2f}")


if __name__ == "__main__":
    main()
