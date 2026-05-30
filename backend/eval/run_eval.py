"""Evaluate hybrid retrieval on the Vietnamese test set.

Computes Recall@5 and MRR@5 for four retrieval configurations
(BM25-only, Dense-only, Hybrid RRF, Hybrid RRF + reranker), using an
LLM-as-judge over a pooled candidate set (TREC-style pooling) to define
the relevant chunk set for each question.

Usage (from backend/):
    python -m eval.run_eval --test-dir ../test
"""
import argparse
import json
from pathlib import Path

from app.config import Settings
from app.indexing.store import KnowledgeBase
from app.retrieval.hybrid import reciprocal_rank_fusion
from .dataset import load_dataset
from .metrics import hit_at_k, mrr_at_k, recall_at_k

K = 5
POOL_PER_METHOD = 10
JUDGE_SYSTEM = (
    "Bạn là giám khảo đánh giá độ liên quan trong hệ thống truy hồi tài liệu.\n"
    "Cho một CÂU HỎI, ĐÁP ÁN ĐÚNG mong đợi, và danh sách các ĐOẠN ứng viên (kèm id).\n"
    "Hãy xác định những đoạn nào CHỨA thông tin cần thiết để suy ra đáp án đúng.\n"
    'Trả về JSON: {"relevant_ids": [<các id liên quan>]}. Chỉ chọn đoạn thực sự liên quan.'
)


def bm25_ranked(kb, q, n):
    return [cid for cid, _ in kb.bm25.search(q, n)]


def dense_ranked(kb, q, n):
    if not kb.vector.ready:
        return []
    qv = kb.embedder.embed_query(q)
    return [cid for cid, _ in kb.vector.search(qv, n)]


def _fused(kb, q):
    s = kb.settings
    bm = kb.bm25.search(q, s.bm25_top_k)
    dn = []
    if kb.vector.ready:
        dn = kb.vector.search(kb.embedder.embed_query(q), s.dense_top_k)
    return reciprocal_rank_fusion(bm, dn, k=s.rrf_k)


def hybrid_ranked(kb, q, n):
    return [h.chunk_id for h in _fused(kb, q)][:n]


def hybrid_rerank_ranked(kb, q, n):
    fused = _fused(kb, q)
    cands = fused[: kb.settings.rerank_top_n]
    meta = kb.repo.get_chunks([h.chunk_id for h in cands])
    pairs = [(h.chunk_id, meta[h.chunk_id]["text"]) for h in cands if h.chunk_id in meta]
    reranked = kb.reranker.rerank(q, pairs, top_k=len(pairs))
    if reranked is None:
        return [h.chunk_id for h in cands][:n]
    return [cid for cid, _ in reranked][:n]


METHODS = {
    "BM25": bm25_ranked,
    "Dense": dense_ranked,
    "Hybrid (RRF)": hybrid_ranked,
    "Hybrid + Rerank": hybrid_rerank_ranked,
}


def judge_relevant(kb, qa, candidate_ids, cache):
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
    data = kb.embedder  # noqa  (ensure embedder loaded)
    from app.agent.llm import LLM

    llm = LLM()
    res = llm.chat_json(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        fast=True,
    )
    rel = [int(x) for x in res.get("relevant_ids", []) if int(x) in set(candidate_ids)]
    cache[key] = rel
    return set(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--storage", default="storage_eval")
    ap.add_argument("--out-dir", default="../data/eval")
    args = ap.parse_args()

    settings = Settings()
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")
    settings.storage_dir = Path(args.storage).resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kb = KnowledgeBase(settings)

    # Ingest the PDF if storage is empty
    if not kb.repo.list_documents():
        pdf = next(test_dir.glob("*.pdf"), None)
        if pdf is None:
            raise SystemExit(f"Không tìm thấy PDF trong {test_dir}")
        print(f"Ingesting {pdf.name} ...")
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)

    dataset = load_dataset(test_dir)
    print(f"Loaded {len(dataset)} câu hỏi.")

    cache_path = out_dir / "judgments.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    per_q = []
    for qa in dataset:
        ranked = {name: fn(kb, qa.question, 20) for name, fn in METHODS.items()}
        pool = []
        for name, ids in ranked.items():
            pool += ids[:POOL_PER_METHOD]
        pool = list(dict.fromkeys(pool))  # dedupe, keep order
        relevant = judge_relevant(kb, qa, pool, cache)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

        row = {"qid": qa.qid, "difficulty": qa.difficulty, "question": qa.question,
               "n_relevant": len(relevant), "methods": {}}
        for name, ids in ranked.items():
            row["methods"][name] = {
                "recall@5": recall_at_k(ids, relevant, K),
                "mrr@5": mrr_at_k(ids, relevant, K),
                "hit@5": hit_at_k(ids, relevant, K),
                "top5": ids[:K],
            }
        per_q.append(row)
        print(f"  {qa.qid}: rel={len(relevant)} "
              + " | ".join(f"{n}:{r['hit@5']:.0f}" for n, r in row["methods"].items()))

    # aggregate
    def agg(rows):
        summary = {}
        for name in METHODS:
            r = [x["methods"][name]["recall@5"] for x in rows]
            m = [x["methods"][name]["mrr@5"] for x in rows]
            h = [x["methods"][name]["hit@5"] for x in rows]
            n = max(len(rows), 1)
            summary[name] = {
                "recall@5": sum(r) / n,
                "mrr@5": sum(m) / n,
                "hit@5": sum(h) / n,
            }
        return summary

    overall = agg(per_q)
    by_diff = {
        d: agg([x for x in per_q if x["difficulty"] == d])
        for d in {x["difficulty"] for x in per_q}
    }
    failures = [
        x for x in per_q
        if x["methods"]["Hybrid + Rerank"]["hit@5"] == 0 and x["n_relevant"] > 0
    ]

    results = {"overall": overall, "by_difficulty": by_diff, "per_question": per_q,
               "n_questions": len(per_q), "k": K}
    (out_dir / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    _write_report(out_dir / "eval_report.md", results, failures, kb)
    print(f"\nĐã ghi báo cáo: {out_dir/'eval_report.md'}")
    for name, s in overall.items():
        print(f"  {name:18s} Recall@5={s['recall@5']:.3f}  MRR@5={s['mrr@5']:.3f}  Hit@5={s['hit@5']:.3f}")


def _table(summary):
    lines = ["| Phương pháp | Recall@5 | MRR@5 | Hit@5 |", "|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['recall@5']:.3f} | {s['mrr@5']:.3f} | {s['hit@5']:.3f} |")
    return "\n".join(lines)


def _write_report(path, results, failures, kb):
    md = [f"# Báo cáo Đánh giá Retrieval (k={results['k']})", ""]
    md.append(f"- Số câu hỏi: **{results['n_questions']}**")
    md.append(f"- Tài liệu: {', '.join(d['title'] for d in kb.repo.list_documents())}")
    md.append("- Phương pháp định nhãn liên quan: LLM-as-judge trên tập ứng viên gộp (pooling).")
    md += ["", "## Kết quả tổng thể", "", _table(results["overall"]), ""]
    for diff, summ in results["by_difficulty"].items():
        md += [f"## Theo độ khó: {diff}", "", _table(summ), ""]
    md += ["## Phân tích lỗi (Hybrid + Rerank trượt @5)", ""]
    if not failures:
        md.append("Không có câu hỏi nào trượt ở top-5 với cấu hình tốt nhất. ✅")
    else:
        for f in failures:
            md.append(f"- **{f['qid']}** ({f['difficulty']}): {f['question']}")
            md.append(f"  - top5 = {f['methods']['Hybrid + Rerank']['top5']}, "
                      f"số đoạn liên quan = {f['n_relevant']}")
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
