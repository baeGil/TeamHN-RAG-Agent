"""Evaluate query transformation variants for RAG retrieval.

This script measures the metrics called out in the query transformation slide:
- Rewrite: Recall@k and MRR@k
- Step-back: context recall and answer coverage
- Decomposition: tail query success and answer completeness

Usage (from backend/):
    python -m eval.query_transformation_eval --test-dir ../test
"""
import argparse
import json
import time
from pathlib import Path

from app.agent.graph import Agent
from app.agent.llm import LLM
from app.agent.query_transform import QueryTransformer
from app.config import Settings
from app.indexing.store import KnowledgeBase
from eval.dataset import load_dataset
from eval.metrics import (
    average_score,
    context_recall_at_k,
    hit_at_k,
    mrr_at_k,
    recall_at_k,
    tail_query_success,
)
from eval.run_eval import judge_relevant

K = 5
POOL_PER_METHOD = 10

CONTEXT_COVERAGE_JUDGE = (
    "Bạn là giám khảo đánh giá ngữ cảnh truy hồi cho hệ thống RAG.\n"
    "Cho CÂU HỎI, ĐÁP ÁN MONG ĐỢI và các ĐOẠN TRUY HỒI, hãy chấm:\n"
    "- answer_coverage: ngữ cảnh có đủ thông tin để tạo đáp án mong đợi không? (0-10)\n"
    "- answer_completeness: ngữ cảnh có bao phủ đủ các ý/nhánh quan trọng của câu hỏi không? (0-10)\n"
    'Trả về JSON: {"answer_coverage": 0-10, "answer_completeness": 0-10, "reason": "..."}'
)


def _rank_original(kb: KnowledgeBase, q: str, n: int) -> list[int]:
    return [h.chunk_id for h in kb.retrieve(q, n)]


def _rank_union(kb: KnowledgeBase, queries: list[str], n: int) -> list[int]:
    by_id = {}
    for q in queries:
        for h in kb.retrieve(q, n):
            old = by_id.get(h.chunk_id)
            if old is None or h.score > old.score:
                by_id[h.chunk_id] = h
    return [
        h.chunk_id
        for h in sorted(by_id.values(), key=lambda x: x.score, reverse=True)[:n]
    ]


def _context_text(kb: KnowledgeBase, ids: list[int]) -> str:
    meta = kb.repo.get_chunks(ids)
    parts = []
    for cid in ids:
        if cid not in meta:
            continue
        txt = meta[cid]["text"][:700].replace("\n", " ")
        title = meta[cid].get("doc_title") or ""
        page = meta[cid].get("page")
        parts.append(f"[id={cid}] {title} trang {page}: {txt}")
    return "\n\n".join(parts)


def _judge_context(kb: KnowledgeBase, qa, ids: list[int], method: str, cache: dict, cache_path: Path, llm: LLM | None = None) -> dict:
    key = f"{qa.qid}:{method}:{','.join(map(str, ids[:K]))}"
    if key in cache:
        return cache[key]
    if llm is None:
        llm = LLM()
    user = (
        f"CÂU HỎI:\n{qa.question}\n\n"
        f"ĐÁP ÁN MONG ĐỢI:\n{qa.expected}\n\n"
        f"ĐOẠN TRUY HỒI:\n{_context_text(kb, ids[:K])}"
    )
    try:
        data = llm.chat_json(
            [{"role": "system", "content": CONTEXT_COVERAGE_JUDGE}, {"role": "user", "content": user}],
            fast=True,
            node="eval_context_coverage",
        )
    except Exception as e:
        data = {"answer_coverage": 0, "answer_completeness": 0, "reason": f"Lỗi judge: {e}"}
    cache[key] = data
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _score_0_1(value) -> float:
    try:
        return max(0.0, min(10.0, float(value))) / 10.0
    except (TypeError, ValueError):
        return 0.0


def _build_methods(kb: KnowledgeBase, qa, transformer: QueryTransformer, agent: Agent) -> dict[str, list[int]]:
    variants = transformer.transform(qa.question, "simple")
    rewrite_q = next((v.query for v in variants if v.kind == "rewrite"), "")
    step_q = next((v.query for v in variants if v.kind == "step_back"), "")
    subqs = agent._plan(qa.question)

    methods = {
        "Baseline": _rank_original(kb, qa.question, 20),
        "Rewrite": _rank_union(kb, [qa.question, rewrite_q] if rewrite_q else [qa.question], 20),
        "Step-back": _rank_union(kb, [qa.question, step_q] if step_q else [qa.question], 20),
        "Decomposition": _rank_union(kb, subqs or [qa.question], 20),
    }
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--storage", default="storage_eval_query_transform")
    ap.add_argument("--out-dir", default="../data/eval_query_transform")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        import shutil
        sp = Path(args.storage).resolve()
        if sp.exists():
            shutil.rmtree(sp)

    settings = Settings()
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")
    settings.storage_dir = Path(args.storage).resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.test_dir).resolve()

    kb = KnowledgeBase(settings)
    if not kb.repo.list_documents():
        pdf = next(test_dir.glob("*.pdf"), None)
        if pdf is None:
            raise SystemExit(f"Không tìm thấy PDF trong {test_dir}")
        print(f"Ingesting {pdf.name} ...")
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)

    dataset = load_dataset(test_dir)
    llm = LLM()
    transformer = QueryTransformer(llm, max_variants=3)
    agent = Agent(kb)

    relevance_cache_path = out_dir / "relevance_judgments.json"
    coverage_cache_path = out_dir / "coverage_judgments.json"
    relevance_cache = json.loads(relevance_cache_path.read_text(encoding="utf-8")) if relevance_cache_path.exists() else {}
    coverage_cache = json.loads(coverage_cache_path.read_text(encoding="utf-8")) if coverage_cache_path.exists() else {}

    rows = []
    started = time.time()
    for qa in dataset:
        ranked = _build_methods(kb, qa, transformer, agent)
        pool = []
        for ids in ranked.values():
            pool.extend(ids[:POOL_PER_METHOD])
        pool = list(dict.fromkeys(pool))
        relevant = judge_relevant(kb, qa, pool, relevance_cache, llm)
        relevance_cache_path.write_text(json.dumps(relevance_cache, ensure_ascii=False, indent=2), encoding="utf-8")

        # Capture baseline latency breakdown (single retrieve; union methods aggregate internally)
        baseline_lb = dict(kb.last_retrieve_latency)

        row = {
            "qid": qa.qid,
            "difficulty": qa.difficulty,
            "question": qa.question,
            "n_relevant": len(relevant),
            "baseline_latency_breakdown_ms": baseline_lb,
            "methods": {},
        }
        for name, ids in ranked.items():
            judged = _judge_context(kb, qa, ids, name, coverage_cache, coverage_cache_path, llm)
            row["methods"][name] = {
                f"recall@{K}": recall_at_k(ids, relevant, K),
                f"context_recall@{K}": context_recall_at_k(ids, relevant, K),
                f"mrr@{K}": mrr_at_k(ids, relevant, K),
                f"hit@{K}": hit_at_k(ids, relevant, K),
                "answer_coverage": _score_0_1(judged.get("answer_coverage", 0)),
                "answer_completeness": _score_0_1(judged.get("answer_completeness", 0)),
                f"top{K}": ids[:K],
            }
        rows.append(row)
        print(f"{qa.qid}: " + " | ".join(
            f"{name} hit={row['methods'][name][f'hit@{K}']:.0f}"
            for name in ranked
        ))

    methods = ["Baseline", "Rewrite", "Step-back", "Decomposition"]
    overall = {}
    for name in methods:
        overall[name] = {
            f"recall@{K}": average_score([r["methods"][name][f"recall@{K}"] for r in rows]),
            f"context_recall@{K}": average_score([r["methods"][name][f"context_recall@{K}"] for r in rows]),
            f"mrr@{K}": average_score([r["methods"][name][f"mrr@{K}"] for r in rows]),
            f"hit@{K}": average_score([r["methods"][name][f"hit@{K}"] for r in rows]),
            "answer_coverage": average_score([r["methods"][name]["answer_coverage"] for r in rows]),
            "answer_completeness": average_score([r["methods"][name]["answer_completeness"] for r in rows]),
            "tail_query_success": tail_query_success(rows, name, K),
        }

    steps = ["bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"]
    n_rows = max(len(rows), 1)
    avg_baseline_lb = {
        step: round(sum(r.get("baseline_latency_breakdown_ms", {}).get(step, 0.0) for r in rows) / n_rows, 1)
        for step in steps
    }

    results = {
        "k": K,
        "n_questions": len(rows),
        "duration_sec": round(time.time() - started, 2),
        "avg_baseline_latency_breakdown_ms": avg_baseline_lb,
        "llm_cost_by_node": {node: dict(cost) for node, cost in llm.node_usage.items()},
        "overall": overall,
        "per_question": rows,
    }
    (out_dir / "query_transformation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(out_dir / "query_transformation_report.md", results)
    print(f"\nĐã ghi báo cáo: {out_dir / 'query_transformation_report.md'}")


def _write_report(path: Path, results: dict) -> None:
    k = results["k"]
    md = [f"# Query Transformation Evaluation (k={k})", ""]
    md.append("| Method | Recall@k | MRR@k | Context recall | Answer coverage | Answer completeness | Tail query success |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in results["overall"].items():
        md.append(
            f"| {name} | {s[f'recall@{k}']:.3f} | {s[f'mrr@{k}']:.3f} | "
            f"{s[f'context_recall@{k}']:.3f} | {s['answer_coverage']:.3f} | "
            f"{s['answer_completeness']:.3f} | {s['tail_query_success']:.3f} |"
        )

    lb = results.get("avg_baseline_latency_breakdown_ms", {})
    if lb:
        md += [
            "",
            "## Retrieval Latency Breakdown — Baseline (avg ms per query)",
            "",
            "| BM25 | Dense+Embed | Fusion | Rerank | Total |",
            "|---:|---:|---:|---:|---:|",
            f"| {lb.get('bm25_ms',0):.1f} | {lb.get('dense_ms',0):.1f} | {lb.get('fusion_ms',0):.1f} | {lb.get('rerank_ms',0):.1f} | {lb.get('total_ms',0):.1f} |",
            "> T1/Step-back/Decomposition gọi retrieve nhiều lần — xem latency_breakdown_ms trong per_question.",
        ]

    node_cost = results.get("llm_cost_by_node", {})
    if node_cost:
        md += [
            "",
            "## LLM Token Cost by Node (cumulative)",
            "",
            "| Node | Model | Prompt | Completion | Calls | Total ms |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for node, cost in node_cost.items():
            md.append(f"| `{node}` | {cost.get('model','')} | {cost.get('prompt_tokens',0)} | {cost.get('completion_tokens',0)} | {cost.get('calls',0)} | {cost.get('total_ms',0):.1f} |")

    md += [
        "",
        "## Cách đọc metric",
        "",
        "- Rewrite nên thắng ở Recall@k và MRR@k.",
        "- Step-back nên thắng ở context recall và answer coverage.",
        "- Decomposition nên thắng ở answer completeness và tail query success.",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
