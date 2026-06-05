"""Evaluate Technique 8: Relevant Segment Extraction (RSE).

Pipeline:
1. Ingest the benchmark PDF once.
2. Run Baseline retrieval (top-k chunks).
3. Run RSE retrieval (top candidates expanded into contiguous segments).
4. Judge relevance with pooled candidates.
5. Optionally run the full agent and judge generation quality.
6. Write JSON + Markdown reports with the metric groups from the roadmap slide.

Usage (from backend/):
    python -m eval.relevant_segment_extraction_eval --test-dir ../test --reset
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
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


@dataclass
class EvalItem:
    qid: str
    question: str
    expected: str
    difficulty: str
    answerability: str = "answerable"
    doc_id: str = ""


def _settings(storage: Path, *, rse: bool) -> Settings:
    s = Settings()
    s.storage_dir = storage.resolve()
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    s.enable_relevant_segment_extraction = rse
    s.rse_max_context_chunks = max(s.rse_max_context_chunks, K)
    return s


def _ensure_ingested(kb: KnowledgeBase, pdf_paths: list[Path]) -> None:
    if kb.repo.list_documents():
        return
    if not pdf_paths:
        raise SystemExit("Không tìm thấy PDF để ingest cho eval.")
    for pdf in pdf_paths:
        print(f"Ingesting {pdf.name} into {kb.settings.storage_dir.name} ...")
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)


def _load_inputs(test_dir: Path, benchmark_path: Path) -> tuple[list[EvalItem], list[Path]]:
    legacy = load_dataset(test_dir)
    if legacy:
        pdfs = sorted(test_dir.glob("*.pdf"))
        return [
            EvalItem(
                qid=qa.qid,
                question=qa.question,
                expected=qa.expected,
                difficulty=qa.difficulty,
            )
            for qa in legacy
        ], pdfs

    if not benchmark_path.exists():
        raise SystemExit(
            f"Không tìm thấy legacy dataset trong {test_dir} hoặc benchmark JSON tại {benchmark_path}"
        )
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    project_root = benchmark_path.parents[2] if len(benchmark_path.parents) >= 3 else Path.cwd()
    pdfs: list[Path] = []
    for doc in raw.get("documents", []):
        local_pdf = Path(doc.get("local_pdf", ""))
        pdf = local_pdf if local_pdf.is_absolute() else project_root / local_pdf
        if pdf.exists():
            pdfs.append(pdf)
    items = [
        EvalItem(
            qid=str(item.get("id", "")),
            question=str(item.get("question", "")),
            expected=str(item.get("expected_answer", "")),
            difficulty=str(item.get("difficulty", "unknown")),
            answerability=str(item.get("answerability", "answerable")),
            doc_id=str(item.get("doc_id", "")),
        )
        for item in raw.get("items", [])
        if item.get("question")
    ]
    return items, pdfs


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
    key = f"{qa.qid}:{method}:{hash(result.get('answer', ''))}"
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
            node="eval_rse_generation",
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
    page_citation_correctness = (
        len(cited_pages & rel_pages) / len(cited_pages)
        if cited_pages and rel_pages
        else 0.0
    )
    return {
        "section_hit_rate": 1.0 if rel_sections and (top_sections & rel_sections) else 0.0,
        "page_hit_rate": 1.0 if rel_pages and (top_pages & rel_pages) else 0.0,
        "page_level_citation_correctness": page_citation_correctness,
    }


def _rse_metrics(hits: list[Any]) -> dict:
    if not hits:
        return {
            "avg_segment_length": 0.0,
            "bridge_chunk_ratio": 0.0,
            "segment_count": 0.0,
        }
    segment_keys = {
        (h.document_id, h.rse_segment_start, h.rse_segment_end)
        for h in hits[:K]
        if h.rse_segment_start is not None and h.rse_segment_end is not None
    }
    segment_lengths = [
        int(end) - int(start) + 1
        for _, start, end in segment_keys
        if start is not None and end is not None
    ]
    bridge_count = sum(1 for h in hits[:K] if getattr(h, "rse_seed", True) is False)
    return {
        "avg_segment_length": average_score([float(x) for x in segment_lengths]),
        "bridge_chunk_ratio": bridge_count / max(len(hits[:K]), 1),
        "segment_count": float(len(segment_keys)),
    }


def _contains_formula(text: str) -> bool:
    return bool(re.search(r"[=∑√∏∫≤≥≠±×÷]|\\frac|\\sum|\\sqrt|\$[^$]+\$", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--benchmark", default="../data/benchmark/context_retrieval_benchmark.json")
    ap.add_argument("--storage", default="storage_eval_rse")
    ap.add_argument("--out-dir", default="../data/eval_relevant_segment_extraction")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    storage = Path(args.storage).resolve()
    if args.reset and storage.exists():
        shutil.rmtree(storage)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.test_dir).resolve()
    benchmark_path = Path(args.benchmark).resolve()

    baseline = KnowledgeBase(_settings(storage, rse=False))
    if not baseline.settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")
    dataset, pdf_paths = _load_inputs(test_dir, benchmark_path)
    _ensure_ingested(baseline, pdf_paths)
    rse = KnowledgeBase(_settings(storage, rse=True))
    llm = LLM()

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
    methods = {"Baseline": baseline, "RSE": rse}
    for qa in dataset:
        ranked: dict[str, list[int]] = {}
        hit_objs: dict[str, list[Any]] = {}
        retrieval_latency: dict[str, float] = {}
        latency_breakdown: dict[str, dict] = {}
        for name, kb in methods.items():
            ids, hits, elapsed, lb = _rank(kb, qa.question, max(20, kb.settings.rse_max_context_chunks))
            ranked[name] = ids
            hit_objs[name] = hits
            retrieval_latency[name] = elapsed
            latency_breakdown[name] = lb

        pool = list(dict.fromkeys(ranked["Baseline"][:POOL_PER_METHOD] + ranked["RSE"][:POOL_PER_METHOD]))
        relevant = judge_relevant(rse, qa, pool, relevance_cache, llm)
        relevance_cache_path.write_text(json.dumps(relevance_cache, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "qid": qa.qid,
            "difficulty": qa.difficulty,
            "answerability": qa.answerability,
            "doc_id": qa.doc_id,
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
            meta = _meta_metrics(kb, ids, relevant, citations)
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
                **meta,
                **_rse_metrics(hit_objs[name]),
            }
        rows.append(row)
        print(
            f"{qa.qid}: "
            f"base_hit={row['methods']['Baseline'][f'hit@{K}']:.0f} "
            f"rse_hit={row['methods']['RSE'][f'hit@{K}']:.0f}"
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
        "avg_segment_length",
        "bridge_chunk_ratio",
        "segment_count",
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
        "technique": "Relevant Segment Extraction",
        "k": K,
        "n_questions": len(rows),
        "settings": {
            "rse_irrelevant_chunk_penalty": rse.settings.rse_irrelevant_chunk_penalty,
            "rse_max_segment_chunks": rse.settings.rse_max_segment_chunks,
            "rse_max_context_chunks": rse.settings.rse_max_context_chunks,
        },
        "avg_latency_breakdown_ms": avg_lb,
        "llm_cost_by_node": {node: dict(cost) for node, cost in llm.node_usage.items()},
        "overall": overall,
        "per_question": rows,
    }
    (out_dir / "relevant_segment_extraction_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(out_dir / "relevant_segment_extraction_report.md", results, metric_names)
    print(f"\nĐã ghi báo cáo: {out_dir / 'relevant_segment_extraction_report.md'}")


def _write_report(path: Path, results: dict, metric_names: list[str]) -> None:
    md = [
        f"# Technique 8 Evaluation: Relevant Segment Extraction (k={results['k']})",
        "",
        f"- Số câu hỏi: **{results['n_questions']}**",
        "- Baseline: top-k chunk rời rạc sau hybrid retrieval/rerank.",
        "- RSE: chọn các segment liên tiếp bằng max-subarray, có thể thêm bridge chunk nằm giữa các seed chunk liên quan.",
        f"- RSE config: penalty={results['settings']['rse_irrelevant_chunk_penalty']}, "
        f"max_segment_chunks={results['settings']['rse_max_segment_chunks']}, "
        f"max_context_chunks={results['settings']['rse_max_context_chunks']}.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Baseline | RSE | Delta |",
        "|---|---:|---:|---:|",
    ]
    base = results["overall"]["Baseline"]
    rse = results["overall"]["RSE"]
    for metric in metric_names:
        delta = rse[metric] - base[metric]
        md.append(f"| {metric} | {base[metric]:.3f} | {rse[metric]:.3f} | {delta:+.3f} |")

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
        "- RSE-specific: average segment length, bridge chunk ratio, segment count.",
        "",
        "## Per-question Deltas",
        "",
        "| QID | Difficulty | Recall Δ | MRR Δ | nDCG Δ | Faithfulness Δ | Citation Δ | Segment len | Bridge ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    k = results["k"]
    for row in results["per_question"]:
        b = row["methods"]["Baseline"]
        r = row["methods"]["RSE"]
        md.append(
            f"| {row['qid']} | {row['difficulty']} | "
            f"{r[f'recall@{k}'] - b[f'recall@{k}']:+.3f} | "
            f"{r[f'mrr@{k}'] - b[f'mrr@{k}']:+.3f} | "
            f"{r[f'ndcg@{k}'] - b[f'ndcg@{k}']:+.3f} | "
            f"{r['faithfulness'] - b['faithfulness']:+.3f} | "
            f"{r['citation_accuracy'] - b['citation_accuracy']:+.3f} | "
            f"{r['avg_segment_length']:.2f} | "
            f"{r['bridge_chunk_ratio']:.2f} |"
        )
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
