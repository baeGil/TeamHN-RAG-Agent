"""Evaluate Technique 10: Hierarchical Indices.

Pipeline:
1. Build the normal chunk index.
2. Build parent indexes by document section/page.
3. Retrieve parent nodes, expand/boost child chunks, then reuse reranking.
4. Judge pooled retrieval relevance.
5. Optionally run the full agent and judge generation quality.
6. Write JSON + Markdown reports with the roadmap metric groups.

Usage (from backend/):
    python -m eval.hierarchical_indices_eval --test-dir ../test --reset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from app.agent.graph import Agent
from app.agent.llm import LLM
from app.config import Settings
from app.indexing.store import KnowledgeBase
from eval.dataset import load_dataset
from eval.metrics import (
    average_score,
    context_precision_at_k,
    context_recall_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    redundancy_ratio,
)
from eval.run_eval import judge_relevant

K = 5
POOL_PER_METHOD = 12

GENERATION_JUDGE = (
    "Bạn là giám khảo đánh giá câu trả lời RAG tiếng Việt.\n"
    "Cho CÂU HỎI, ĐÁP ÁN MONG ĐỢI, CÂU TRẢ LỜI và CITATIONS, hãy chấm 0-10:\n"
    "- answer_correctness: trả lời có đúng ý đáp án mong đợi không\n"
    "- answer_exactness: có đủ chi tiết/số liệu/công thức quan trọng không\n"
    "- faithfulness: nội dung có bám nguồn và không bịa không\n"
    "- groundedness: lập luận có dựa trên citation/context không\n"
    "- citation_accuracy: citation có phù hợp với nội dung trả lời không\n"
    "- formula_correctness: công thức/ký hiệu toán có đúng không; nếu câu không cần công thức, cho 10\n"
    'Trả về JSON: {"answer_correctness":0-10,"answer_exactness":0-10,'
    '"faithfulness":0-10,"groundedness":0-10,"citation_accuracy":0-10,'
    '"formula_correctness":0-10,"reason":"..."}'
)


def _settings(storage: Path, *, hierarchical: bool) -> Settings:
    s = Settings()
    s.storage_dir = storage.resolve()
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    s.enable_hierarchical_indices = hierarchical
    return s


def _ensure_ingested(kb: KnowledgeBase, test_dir: Path) -> None:
    if kb.repo.list_documents():
        return
    pdf = next(test_dir.glob("*.pdf"), None)
    if pdf is None:
        raise SystemExit(f"Không tìm thấy PDF trong {test_dir}")
    print(f"Ingesting {pdf.name} into {kb.settings.storage_dir.name} ...")
    kb.ingest_pdf(pdf.read_bytes(), pdf.name)


def _rank(kb: KnowledgeBase, question: str, n: int) -> tuple[list[int], list[Any], float, dict]:
    hits = kb.retrieve(question, n)
    lb = dict(kb.last_retrieve_latency)
    return [h.chunk_id for h in hits], hits, lb.get("total_ms", 0.0), lb


def _run_agent(kb: KnowledgeBase, question: str) -> dict:
    agent = Agent(kb)
    started = time.perf_counter()
    final = None
    for ev in agent.run(question):
        if ev.get("type") == "final":
            final = ev.get("data", {})
    elapsed = (time.perf_counter() - started) * 1000
    final = final or {}
    usage = final.get("usage", {}) or {}
    answer = final.get("answer", "")
    return {
        "answer": answer,
        "citations": final.get("citations", []),
        "route": final.get("route", ""),
        "partial": bool(final.get("partial", False)),
        "latency_ms": elapsed,
        "usage": usage,
        "total_tokens": int(usage.get("prompt_tokens", 0) or 0)
        + int(usage.get("completion_tokens", 0) or 0),
        "no_answer": _is_no_answer(answer),
    }


def _is_no_answer(answer: str) -> bool:
    low = answer.lower()
    return "không tìm thấy" in low or "không có thông tin" in low or "chưa đủ thông tin" in low


def _judge_generation(qa, result: dict, cache: dict, cache_path: Path, method: str, llm: LLM | None = None) -> dict:
    answer_hash = hashlib.sha256(result.get("answer", "").encode("utf-8")).hexdigest()[:16]
    key = f"{qa.qid}:{method}:{answer_hash}"
    if key in cache:
        return cache[key]
    if llm is None:
        llm = LLM()
    user = (
        f"CÂU HỎI:\n{qa.question}\n\n"
        f"ĐÁP ÁN MONG ĐỢI:\n{qa.expected}\n\n"
        f"CÂU TRẢ LỜI:\n{result.get('answer', '')}\n\n"
        f"CITATIONS:\n{json.dumps(result.get('citations', []), ensure_ascii=False)}"
    )
    try:
        judged = llm.chat_json(
            [{"role": "system", "content": GENERATION_JUDGE}, {"role": "user", "content": user}],
            fast=True,
            node="eval_hierarchical_generation",
        )
    except Exception as e:
        judged = {
            "answer_correctness": 0,
            "answer_exactness": 0,
            "faithfulness": 0,
            "groundedness": 0,
            "citation_accuracy": 0,
            "formula_correctness": 0,
            "reason": f"Lỗi judge: {e}",
        }
    cache[key] = judged
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return judged


def _meta_metrics(kb: KnowledgeBase, ids: list[int], relevant: set[int], citations: list[dict]) -> dict:
    meta = kb.repo.get_chunks(list(dict.fromkeys(ids[:K] + list(relevant))))
    rel_pages = {meta[cid].get("page") for cid in relevant if cid in meta and meta[cid].get("page")}
    rel_sections = {
        meta[cid].get("section")
        for cid in relevant
        if cid in meta and meta[cid].get("section")
    }
    top_pages = {meta[cid].get("page") for cid in ids[:K] if cid in meta and meta[cid].get("page")}
    top_sections = {
        meta[cid].get("section")
        for cid in ids[:K]
        if cid in meta and meta[cid].get("section")
    }
    cited_pages = {c.get("page") for c in citations if c.get("page")}
    return {
        "section_hit_rate": 1.0 if rel_sections and (top_sections & rel_sections) else 0.0,
        "page_hit_rate": 1.0 if rel_pages and (top_pages & rel_pages) else 0.0,
        "page_level_citation_correctness": (
            len(cited_pages & rel_pages) / len(cited_pages)
            if cited_pages and rel_pages
            else 0.0
        ),
    }


def _hierarchical_metrics(hits: list[Any], relevant: set[int], kb: KnowledgeBase) -> dict:
    top = hits[:K]
    boosted = [h for h in top if getattr(h, "hierarchical_boosted", False)]
    relevant_parents = {
        kb.chunk_to_parent[cid]
        for cid in relevant
        if cid in kb.chunk_to_parent
    }
    top_parents = {
        h.hierarchical_parent_id or kb.chunk_to_parent.get(h.chunk_id)
        for h in top
        if h.hierarchical_parent_id or h.chunk_id in kb.chunk_to_parent
    }
    parent_sizes = [
        len(kb.parent_nodes[pid].chunk_ids)
        for pid in top_parents
        if pid in kb.parent_nodes
    ]
    return {
        "parent_hit_rate": 1.0 if relevant_parents and (top_parents & relevant_parents) else 0.0,
        "hierarchical_boosted_ratio": len(boosted) / max(len(top), 1),
        "avg_parent_size": average_score([float(x) for x in parent_sizes]),
    }


def _contains_formula(text: str) -> bool:
    return bool(re.search(r"[=∑√∏∫≤≥≠±×÷]|\\frac|\\sum|\\sqrt|\$[^$]+\$", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--storage", default="storage_eval_hierarchical")
    ap.add_argument("--out-dir", default="../data/eval_hierarchical_indices")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    storage = Path(args.storage).resolve()
    if args.reset and storage.exists():
        shutil.rmtree(storage)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.test_dir).resolve()

    baseline = KnowledgeBase(_settings(storage, hierarchical=False))
    if not baseline.settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")
    _ensure_ingested(baseline, test_dir)
    hierarchical = KnowledgeBase(_settings(storage, hierarchical=True))
    llm = LLM()

    dataset = load_dataset(test_dir)
    relevance_cache_path = out_dir / "relevance_judgments.json"
    generation_cache_path = out_dir / "generation_judgments.json"
    relevance_cache = (
        json.loads(relevance_cache_path.read_text(encoding="utf-8"))
        if relevance_cache_path.exists()
        else {}
    )
    generation_cache = (
        json.loads(generation_cache_path.read_text(encoding="utf-8"))
        if generation_cache_path.exists()
        else {}
    )

    rows = []
    methods = {"Baseline": baseline, "Hierarchical": hierarchical}
    for qa in dataset:
        ranked: dict[str, list[int]] = {}
        hit_objs: dict[str, list[Any]] = {}
        retrieval_latency: dict[str, float] = {}
        latency_breakdown: dict[str, dict] = {}
        for name, kb in methods.items():
            ids, hits, elapsed, lb = _rank(kb, qa.question, 20)
            ranked[name] = ids
            hit_objs[name] = hits
            retrieval_latency[name] = elapsed
            latency_breakdown[name] = lb

        pool = list(dict.fromkeys(ranked["Baseline"][:POOL_PER_METHOD] + ranked["Hierarchical"][:POOL_PER_METHOD]))
        relevant = judge_relevant(hierarchical, qa, pool, relevance_cache, llm)
        relevance_cache_path.write_text(json.dumps(relevance_cache, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "qid": qa.qid,
            "difficulty": qa.difficulty,
            "question": qa.question,
            "expected_has_formula": _contains_formula(qa.expected),
            "n_relevant": len(relevant),
            "methods": {},
        }
        for name, kb in methods.items():
            ids = ranked[name]
            texts = [h.text for h in hit_objs[name][:K]]
            generation_result = {}
            judged = {}
            if not args.skip_generation:
                generation_result = _run_agent(kb, qa.question)
                judged = _judge_generation(qa, generation_result, generation_cache, generation_cache_path, name, llm)
            citations = generation_result.get("citations", [])
            row["methods"][name] = {
                f"recall@{K}": recall_at_k(ids, relevant, K),
                f"mrr@{K}": mrr_at_k(ids, relevant, K),
                f"ndcg@{K}": ndcg_at_k(ids, relevant, K),
                f"context_precision@{K}": context_precision_at_k(ids, relevant, K),
                f"context_recall@{K}": context_recall_at_k(ids, relevant, K),
                f"hit@{K}": hit_at_k(ids, relevant, K),
                f"top{K}": ids[:K],
                "redundancy_ratio": redundancy_ratio(texts),
                "retrieval_latency_ms": retrieval_latency[name],
                "latency_breakdown_ms": latency_breakdown[name],
                "answer_latency_ms": generation_result.get("latency_ms", 0.0),
                "total_tokens": generation_result.get("total_tokens", 0),
                "no_answer_rate": 1.0 if generation_result.get("no_answer", False) else 0.0,
                "refusal_rate": 1.0 if generation_result.get("partial", False) else 0.0,
                "answer_correctness": float(judged.get("answer_correctness", 0)) / 10.0,
                "answer_exactness": float(judged.get("answer_exactness", 0)) / 10.0,
                "faithfulness": float(judged.get("faithfulness", 0)) / 10.0,
                "groundedness": float(judged.get("groundedness", 0)) / 10.0,
                "citation_accuracy": float(judged.get("citation_accuracy", 0)) / 10.0,
                "formula_correctness": float(judged.get("formula_correctness", 0)) / 10.0,
                **_meta_metrics(kb, ids, relevant, citations),
                **_hierarchical_metrics(hit_objs[name], relevant, kb),
            }
        rows.append(row)
        print(
            f"{qa.qid}: "
            f"base_hit={row['methods']['Baseline'][f'hit@{K}']:.0f} "
            f"hier_hit={row['methods']['Hierarchical'][f'hit@{K}']:.0f}"
        )

    metric_names = [
        f"recall@{K}",
        f"mrr@{K}",
        f"ndcg@{K}",
        f"context_precision@{K}",
        f"context_recall@{K}",
        "redundancy_ratio",
        "retrieval_latency_ms",
        "answer_latency_ms",
        "total_tokens",
        "no_answer_rate",
        "refusal_rate",
        "answer_correctness",
        "answer_exactness",
        "faithfulness",
        "groundedness",
        "citation_accuracy",
        "formula_correctness",
        "section_hit_rate",
        "page_hit_rate",
        "page_level_citation_correctness",
        "parent_hit_rate",
        "hierarchical_boosted_ratio",
        "avg_parent_size",
    ]
    overall = {
        name: {
            metric: average_score([r["methods"][name][metric] for r in rows])
            for metric in metric_names
        }
        for name in methods
    }
    steps = ["bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"]
    n_rows = max(len(rows), 1)
    avg_lb: dict[str, dict] = {
        name: {
            step: round(sum(r["methods"][name].get("latency_breakdown_ms", {}).get(step, 0.0) for r in rows) / n_rows, 1)
            for step in steps
        }
        for name in methods
    }

    results = {
        "technique": "Hierarchical Indices",
        "k": K,
        "n_questions": len(rows),
        "settings": {
            "parent_nodes": len(hierarchical.parent_nodes),
            "parent_top_k": hierarchical.settings.hierarchical_parent_top_k,
            "parent_chunk_window": hierarchical.settings.hierarchical_parent_chunk_window,
            "parent_boost": hierarchical.settings.hierarchical_parent_boost,
            "parent_max_chars": hierarchical.settings.hierarchical_parent_max_chars,
        },
        "avg_latency_breakdown_ms": avg_lb,
        "llm_cost_by_node": {node: dict(cost) for node, cost in llm.node_usage.items()},
        "overall": overall,
        "per_question": rows,
    }
    (out_dir / "hierarchical_indices_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(out_dir / "hierarchical_indices_report.md", results, metric_names)
    print(f"\nĐã ghi báo cáo: {out_dir / 'hierarchical_indices_report.md'}")


def _write_report(path: Path, results: dict, metric_names: list[str]) -> None:
    md = [
        f"# Technique 10 Evaluation: Hierarchical Indices (k={results['k']})",
        "",
        f"- Số câu hỏi: **{results['n_questions']}**",
        "- Baseline: hybrid BM25+dense+RRF+rerank trên chunk phẳng.",
        "- Hierarchical: tạo parent index theo section/page, retrieve parent trước, rồi mở rộng/boost child chunks trước rerank.",
        f"- Parent config: nodes={results['settings']['parent_nodes']}, "
        f"top_k={results['settings']['parent_top_k']}, "
        f"chunk_window={results['settings']['parent_chunk_window']}, "
        f"boost={results['settings']['parent_boost']}.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Baseline | Hierarchical | Delta |",
        "|---|---:|---:|---:|",
    ]
    base = results["overall"]["Baseline"]
    hier = results["overall"]["Hierarchical"]
    for metric in metric_names:
        delta = hier[metric] - base[metric]
        md.append(f"| {metric} | {base[metric]:.3f} | {hier[metric]:.3f} | {delta:+.3f} |")

    avg_lb = results.get("avg_latency_breakdown_ms", {})
    if avg_lb:
        md += [
            "",
            "## Retrieval Latency Breakdown (avg ms)",
            "",
            "| Method | BM25 | Dense+Embed | Fusion | Rerank | Total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, lb in avg_lb.items():
            md.append(f"| {name} | {lb.get('bm25_ms',0):.1f} | {lb.get('dense_ms',0):.1f} | {lb.get('fusion_ms',0):.1f} | {lb.get('rerank_ms',0):.1f} | {lb.get('total_ms',0):.1f} |")

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
        "## Metric Groups",
        "",
        "- Retrieval: Recall@k, MRR@k, nDCG@k, context precision/recall, redundancy ratio.",
        "- Generation: answer correctness/exactness, faithfulness, groundedness, citation accuracy.",
        "- System: retrieval latency, answer latency, token usage, no-answer/refusal rate.",
        "- Domain: formula correctness, section hit rate, page hit rate, page-level citation correctness.",
        "- Hierarchical-specific: parent hit rate, boosted chunk ratio, average parent size.",
        "",
        "## Per-question Deltas",
        "",
        "| QID | Difficulty | Recall Δ | MRR Δ | nDCG Δ | Faithfulness Δ | Citation Δ | Parent hit | Boosted ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    k = results["k"]
    for row in results["per_question"]:
        b = row["methods"]["Baseline"]
        h = row["methods"]["Hierarchical"]
        md.append(
            f"| {row['qid']} | {row['difficulty']} | "
            f"{h[f'recall@{k}'] - b[f'recall@{k}']:+.3f} | "
            f"{h[f'mrr@{k}'] - b[f'mrr@{k}']:+.3f} | "
            f"{h[f'ndcg@{k}'] - b[f'ndcg@{k}']:+.3f} | "
            f"{h['faithfulness'] - b['faithfulness']:+.3f} | "
            f"{h['citation_accuracy'] - b['citation_accuracy']:+.3f} | "
            f"{h['parent_hit_rate']:.3f} | "
            f"{h['hierarchical_boosted_ratio']:.3f} |"
        )
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
