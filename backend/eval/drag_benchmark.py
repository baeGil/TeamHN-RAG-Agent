"""Compare baseline RAG with DRAG conflict-aware RAG on DRAG.pdf.

Usage from backend/:
    python -m eval.drag_benchmark --reset

Outputs:
    ../data/vietanh-data/drag_eval/drag_benchmark_results.json
    ../data/vietanh-data/drag_eval/drag_benchmark_report.md
    ../data/vietanh-data/drag_eval/drag_answer_comparison.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.llm import LLM
from app.agent.prompts import ANSWER_SYSTEM, build_context
from app.config import Settings
from app.indexing.store import KnowledgeBase, RetrievedChunk
from app.retrieval.drag import DRAGRetriever
from eval.metrics import hit_at_k, mrr_at_k, recall_at_k


K = 5
POOL_K = 10
RAGAS_METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

RELEVANCE_JUDGE_SYSTEM = """You judge retrieval relevance.

Given a question, expected answer, and candidate chunks with ids, return JSON:
{"relevant_ids": [<ids>]}.

Select chunks that contain information needed to answer the question or verify
the expected answer. Ignore chunks that are only topically related."""

RAGAS_JUDGE_SYSTEM = """You are an evaluator for grounded RAG answers.

Score the answer from 0 to 1 for these RAGAS-style metrics:
- faithfulness: claims are supported by the supplied contexts.
- answer_relevancy: answer addresses the question.
- context_precision: supplied contexts used by the answer are relevant.
- context_recall: contexts plus answer cover the expected answer.
- answer_correctness: answer is correct against the expected answer.

Also score this DRAG-specific metric separately:
- behavior_alignment: answer follows the expected conflict behavior.

Return JSON:
{
  "faithfulness": 0-1,
  "answer_relevancy": 0-1,
  "context_precision": 0-1,
  "context_recall": 0-1,
  "answer_correctness": 0-1,
  "behavior_alignment": 0-1,
  "reason": "<short explanation>"
}"""


@dataclass
class DragQA:
    qid: str
    question: str
    expected: str
    conflict_type: str
    difficulty: str
    expected_behavior: str


def load_jsonl_dataset(path: Path) -> list[DragQA]:
    rows: list[DragQA] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        rows.append(
            DragQA(
                qid=str(data.get("qid") or f"q{line_no:03d}"),
                question=str(data["question"]),
                expected=str(data["expected"]),
                conflict_type=str(data.get("conflict_type", "no_conflict")),
                difficulty=str(data.get("difficulty", "medium")),
                expected_behavior=str(data.get("expected_behavior", "")),
            )
        )
    return rows


def ensure_kb(pdf_path: Path, storage: Path, reset: bool) -> KnowledgeBase:
    if reset and storage.exists():
        shutil.rmtree(storage)
    settings = Settings()
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY is not configured in backend/.env")
    settings.storage_dir = storage.resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    kb = KnowledgeBase(settings)
    if not kb.repo.list_documents():
        print(f"Ingesting {pdf_path} ...")
        kb.ingest_pdf(pdf_path.read_bytes(), pdf_path.name)
    return kb


def _chunk_dicts(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    out = []
    for i, chunk in enumerate(chunks, start=1):
        out.append(
            {
                "label": i,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "doc_title": chunk.doc_title,
                "doc_source": chunk.doc_source,
                "page": chunk.page,
                "section": chunk.section,
                "score": chunk.score,
            }
        )
    return out


def run_baseline(kb: KnowledgeBase, qa: DragQA) -> dict[str, Any]:
    llm = LLM()
    started = time.perf_counter()
    chunks = kb.retrieve(qa.question, K)
    contexts = _chunk_dicts(chunks)
    user = (
        f"NGỮ CẢNH:\n{build_context(contexts, label_key='label')}\n\n"
        f"CÂU HỎI:\n{qa.question}\n\n"
        "Trả lời dựa trên ngữ cảnh và trích dẫn bằng [số]."
    )
    answer = llm.chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        fast=False,
        node="baseline_answer",
    ).strip()
    elapsed = time.perf_counter() - started
    return {
        "answer": answer,
        "chunks": contexts,
        "ranked_ids": [c.chunk_id for c in chunks],
        "elapsed": elapsed,
        "usage": dict(llm.usage),
    }


def run_drag(kb: KnowledgeBase, qa: DragQA) -> dict[str, Any]:
    llm = LLM()
    drag = DRAGRetriever(kb, llm=llm, evidence_k=8)
    started = time.perf_counter()
    chunks, assessment = drag.retrieve(qa.question, K)
    contexts = _chunk_dicts(chunks)
    messages = drag.build_answer_messages(qa.question, contexts, assessment)
    answer = llm.chat(messages, fast=False, node="drag_answer").strip()
    elapsed = time.perf_counter() - started
    return {
        "answer": answer,
        "chunks": contexts,
        "ranked_ids": [c.chunk_id for c in chunks],
        "elapsed": elapsed,
        "usage": dict(llm.usage),
        "conflict_assessment": {
            "conflict_type": assessment.conflict_type,
            "confidence": assessment.confidence,
            "rationale": assessment.rationale,
            "answer_policy": assessment.answer_policy,
        },
    }


def judge_relevance(
    kb: KnowledgeBase,
    qa: DragQA,
    candidate_ids: list[int],
    cache: dict[str, Any],
    cache_path: Path,
) -> set[int]:
    key = f"{qa.qid}:{_qa_cache_key(qa)}:{','.join(str(i) for i in candidate_ids)}"
    if key in cache:
        return {int(x) for x in cache[key]}

    meta = kb.repo.get_chunks(candidate_ids)
    candidates = []
    for cid in candidate_ids:
        row = meta.get(cid)
        if row:
            candidates.append(f"[id={cid}] {row['text'][:900].replace(chr(10), ' ')}")
    llm = LLM()
    user = (
        f"QUESTION:\n{qa.question}\n\n"
        f"EXPECTED ANSWER:\n{qa.expected}\n\n"
        f"CANDIDATE CHUNKS:\n" + "\n\n".join(candidates)
    )
    data = llm.chat_json(
        [{"role": "system", "content": RELEVANCE_JUDGE_SYSTEM}, {"role": "user", "content": user}],
        fast=True,
        node="drag_relevance_judge",
    )
    relevant = [
        int(x)
        for x in data.get("relevant_ids", [])
        if str(x).isdigit() and int(x) in set(candidate_ids)
    ]
    cache[key] = relevant
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return set(relevant)


def judge_ragas_metrics(
    qa: DragQA,
    method_name: str,
    result: dict[str, Any],
    cache: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    key = f"{qa.qid}:{_qa_cache_key(qa)}:{method_name}:{_stable_answer_key(result['answer'])}"
    if key in cache:
        return _normalize_eval_metrics(cache[key])

    contexts = "\n\n".join(
        f"[{c['label']}] {c['text'][:1000].replace(chr(10), ' ')}"
        for c in result.get("chunks", [])
    )
    user = (
        f"QUESTION:\n{qa.question}\n\n"
        f"EXPECTED ANSWER:\n{qa.expected}\n\n"
        f"GOLD CONFLICT TYPE:\n{qa.conflict_type}\n\n"
        f"EXPECTED CONFLICT BEHAVIOR:\n{qa.expected_behavior}\n\n"
        f"ANSWER:\n{result['answer']}\n\n"
        f"CONTEXTS:\n{contexts}"
    )
    llm = LLM()
    data = llm.chat_json(
        [{"role": "system", "content": RAGAS_JUDGE_SYSTEM}, {"role": "user", "content": user}],
        fast=True,
        node="drag_ragas_judge",
    )
    metrics = {}
    for name in (*RAGAS_METRIC_KEYS, "behavior_alignment"):
        try:
            metrics[name] = max(0.0, min(1.0, float(data.get(name, 0.0))))
        except (TypeError, ValueError):
            metrics[name] = 0.0
    metrics["ragas_avg"] = sum(metrics[k] for k in RAGAS_METRIC_KEYS) / len(RAGAS_METRIC_KEYS)
    metrics["reason"] = str(data.get("reason", ""))
    cache[key] = metrics
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _normalize_eval_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Keep old cached judgments usable after metric naming changes."""
    out = dict(metrics)
    for key in (*RAGAS_METRIC_KEYS, "behavior_alignment"):
        try:
            out[key] = max(0.0, min(1.0, float(out.get(key, 0.0))))
        except (TypeError, ValueError):
            out[key] = 0.0
    out["ragas_avg"] = sum(out[k] for k in RAGAS_METRIC_KEYS) / len(RAGAS_METRIC_KEYS)
    out["reason"] = str(out.get("reason", ""))
    return out


def _stable_answer_key(answer: str) -> str:
    import hashlib

    return hashlib.sha1(answer.encode("utf-8")).hexdigest()[:16]


def _qa_cache_key(qa: DragQA) -> str:
    import hashlib

    text = "\n".join([qa.question, qa.expected, qa.conflict_type, qa.expected_behavior])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _usage_total(usage: dict[str, Any]) -> int:
    return int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ("baseline", "drag")
    out: dict[str, Any] = {}
    for method in methods:
        n = max(len(rows), 1)
        retrieval = {
            "recall@5": sum(r[method]["retrieval"]["recall@5"] for r in rows) / n,
            "mrr@5": sum(r[method]["retrieval"]["mrr@5"] for r in rows) / n,
            "hit@5": sum(r[method]["retrieval"]["hit@5"] for r in rows) / n,
        }
        ragas_keys = [
            *RAGAS_METRIC_KEYS,
            "behavior_alignment",
            "ragas_avg",
        ]
        ragas = {k: sum(r[method]["ragas"][k] for r in rows) / n for k in ragas_keys}
        total_latency = sum(r[method]["elapsed"] for r in rows)
        total_tokens = sum(_usage_total(r[method]["usage"]) for r in rows)
        out[method] = {
            "retrieval": retrieval,
            "ragas": ragas,
            "avg_latency_s": total_latency / n,
            "total_latency_s": total_latency,
            "avg_tokens": total_tokens / n,
            "total_tokens": total_tokens,
            "total_llm_calls": sum(int(r[method]["usage"].get("calls", 0) or 0) for r in rows),
        }
    return out


def by_group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups = sorted({r[key] for r in rows})
    return {group: aggregate([r for r in rows if r[key] == group]) for group in groups}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="../data/vietanh-data/DRAG.pdf")
    ap.add_argument("--dataset", default="../data/vietanh-data/drag_eval/drag_mock_qa.jsonl")
    ap.add_argument("--storage", default="storage_eval_drag")
    ap.add_argument("--out-dir", default="../data/vietanh-data/drag_eval")
    ap.add_argument("--reset", action="store_true", help="Rebuild DRAG eval storage")
    ap.add_argument("--qid", help="Run only one question by qid, for example D001")
    ap.add_argument("--limit", type=int, help="Run only the first N questions after filtering")
    ap.add_argument(
        "--write-comparison-only",
        action="store_true",
        help="Create drag_answer_comparison.md from existing drag_benchmark_results.json",
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    dataset_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = Path(args.storage).resolve()

    if args.write_comparison_only:
        results_path = out_dir / "drag_benchmark_results.json"
        if not results_path.exists():
            raise SystemExit(f"Missing results file: {results_path}")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        write_answer_comparison(out_dir / "drag_answer_comparison.md", results)
        print(f"Answer comparison: {out_dir / 'drag_answer_comparison.md'}")
        return

    kb = ensure_kb(pdf_path, storage, args.reset)
    dataset = load_jsonl_dataset(dataset_path)
    if args.qid:
        dataset = [qa for qa in dataset if qa.qid == args.qid]
        if not dataset:
            raise SystemExit(f"Question id not found: {args.qid}")
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than 0")
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} DRAG benchmark questions.")

    relevance_cache_path = out_dir / "drag_relevance_judgments.json"
    ragas_cache_path = out_dir / "drag_ragas_judgments.json"
    relevance_cache = json.loads(relevance_cache_path.read_text(encoding="utf-8")) if relevance_cache_path.exists() else {}
    ragas_cache = json.loads(ragas_cache_path.read_text(encoding="utf-8")) if ragas_cache_path.exists() else {}

    rows = []
    for qa in dataset:
        print(f"\n[{qa.qid}] {qa.conflict_type}: {qa.question[:90]}")
        baseline = run_baseline(kb, qa)
        drag = run_drag(kb, qa)

        candidate_ids = list(dict.fromkeys(
            baseline["ranked_ids"][:POOL_K] + drag["ranked_ids"][:POOL_K]
        ))
        relevant = judge_relevance(kb, qa, candidate_ids, relevance_cache, relevance_cache_path)

        for name, result in (("baseline", baseline), ("drag", drag)):
            ids = result["ranked_ids"]
            result["retrieval"] = {
                "recall@5": recall_at_k(ids, relevant, K),
                "mrr@5": mrr_at_k(ids, relevant, K),
                "hit@5": hit_at_k(ids, relevant, K),
                "top5": ids[:K],
                "n_relevant_in_pool": len(relevant),
            }
            result["ragas"] = judge_ragas_metrics(qa, name, result, ragas_cache, ragas_cache_path)

        row = {
            "qid": qa.qid,
            "question": qa.question,
            "expected": qa.expected,
            "conflict_type": qa.conflict_type,
            "difficulty": qa.difficulty,
            "expected_behavior": qa.expected_behavior,
            "baseline": baseline,
            "drag": drag,
        }
        rows.append(row)
        print(
            "  baseline "
            f"R@5={baseline['retrieval']['recall@5']:.2f} "
            f"MRR@5={baseline['retrieval']['mrr@5']:.2f} "
            f"ragas={baseline['ragas']['ragas_avg']:.2f} "
            f"lat={baseline['elapsed']:.1f}s tok={_usage_total(baseline['usage'])}"
        )
        print(
            "  drag     "
            f"R@5={drag['retrieval']['recall@5']:.2f} "
            f"MRR@5={drag['retrieval']['mrr@5']:.2f} "
            f"ragas={drag['ragas']['ragas_avg']:.2f} "
            f"lat={drag['elapsed']:.1f}s tok={_usage_total(drag['usage'])} "
            f"type={drag['conflict_assessment']['conflict_type']}"
        )

    results = {
        "n_questions": len(rows),
        "k": K,
        "pdf": str(pdf_path),
        "dataset": str(dataset_path),
        "overall": aggregate(rows),
        "by_conflict_type": by_group(rows, "conflict_type"),
        "by_difficulty": by_group(rows, "difficulty"),
        "per_question": rows,
    }
    (out_dir / "drag_benchmark_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir / "drag_benchmark_report.md", results)
    write_answer_comparison(out_dir / "drag_answer_comparison.md", results)
    print(f"\nReport: {out_dir / 'drag_benchmark_report.md'}")
    print(f"Answer comparison: {out_dir / 'drag_answer_comparison.md'}")
    print(f"Results: {out_dir / 'drag_benchmark_results.json'}")


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# DRAG vs Baseline Benchmark",
        "",
        f"- PDF: `{results['pdf']}`",
        f"- Dataset: `{results['dataset']}`",
        f"- Questions: **{results['n_questions']}**",
        f"- Retrieval k: **{results['k']}**",
        "",
        "## Overall",
        "",
        _summary_table(results["overall"]),
        "",
        "## Delta (DRAG - Baseline)",
        "",
        _delta_table(results["overall"]),
        "",
        "## By Conflict Type",
        "",
    ]
    for group, summary in results["by_conflict_type"].items():
        lines.extend([f"### {group}", "", _summary_table(summary), ""])
    lines.extend(["## Per Question", ""])
    lines.append("| QID | Type | Baseline R@5 | DRAG R@5 | Baseline RAGAS avg | DRAG RAGAS avg | DRAG behavior | DRAG predicted |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in results["per_question"]:
        lines.append(
            f"| {row['qid']} | {row['conflict_type']} | "
            f"{row['baseline']['retrieval']['recall@5']:.3f} | "
            f"{row['drag']['retrieval']['recall@5']:.3f} | "
            f"{row['baseline']['ragas']['ragas_avg']:.3f} | "
            f"{row['drag']['ragas']['ragas_avg']:.3f} | "
            f"{row['drag']['ragas']['behavior_alignment']:.3f} | "
            f"{row['drag']['conflict_assessment']['conflict_type']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_answer_comparison(path: Path, results: dict[str, Any]) -> None:
    """Write a readable per-question comparison of baseline vs DRAG answers."""
    lines = [
        "# So sánh câu trả lời Baseline vs DRAG",
        "",
        f"- PDF: `{results['pdf']}`",
        f"- Dataset: `{results['dataset']}`",
        f"- Số câu hỏi: **{results['n_questions']}**",
        "",
        "File này dùng để đọc thủ công: mỗi câu có nhãn conflict đúng, nhãn DRAG dự đoán, "
        "câu trả lời baseline và câu trả lời DRAG.",
        "",
    ]

    for row in results["per_question"]:
        baseline = row["baseline"]
        drag = row["drag"]
        drag_assess = drag.get("conflict_assessment", {})
        lines.extend(
            [
                f"## {row['qid']} - {row['conflict_type']}",
                "",
                f"**Câu hỏi:** {row['question']}",
                "",
                f"**Expected:** {row['expected']}",
                "",
                f"**Expected behavior:** {row['expected_behavior']}",
                "",
                "| Trường | Baseline | DRAG |",
                "|---|---:|---:|",
                (
                    f"| Recall@5 | {baseline['retrieval']['recall@5']:.3f} | "
                    f"{drag['retrieval']['recall@5']:.3f} |"
                ),
                (
                    f"| MRR@5 | {baseline['retrieval']['mrr@5']:.3f} | "
                    f"{drag['retrieval']['mrr@5']:.3f} |"
                ),
                (
                    f"| RAGAS avg | {baseline['ragas']['ragas_avg']:.3f} | "
                    f"{drag['ragas']['ragas_avg']:.3f} |"
                ),
                (
                    f"| Behavior alignment | {baseline['ragas']['behavior_alignment']:.3f} | "
                    f"{drag['ragas']['behavior_alignment']:.3f} |"
                ),
                "",
                f"**DRAG phân loại:** `{drag_assess.get('conflict_type', '')}` "
                f"(confidence={float(drag_assess.get('confidence', 0.0)):.2f})",
                "",
                f"**Lý do DRAG:** {drag_assess.get('rationale', '')}",
                "",
                f"**Policy DRAG:** {drag_assess.get('answer_policy', '')}",
                "",
                "**Baseline answer:**",
                "",
                "```markdown",
                baseline.get("answer", "").strip(),
                "```",
                "",
                "**DRAG answer:**",
                "",
                "```markdown",
                drag.get("answer", "").strip(),
                "```",
                "",
                "**Top chunks:**",
                "",
                f"- Baseline: `{baseline['retrieval'].get('top5', [])}`",
                f"- DRAG: `{drag['retrieval'].get('top5', [])}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | Recall@5 | MRR@5 | Hit@5 | RAGAS avg | Faithfulness | Answer rel | Ctx precision | Ctx recall | Correctness | Behavior | Avg latency | Total latency | Avg tokens | Total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("baseline", "drag"):
        s = summary[method]
        lines.append(
            f"| {method} | "
            f"{s['retrieval']['recall@5']:.3f} | "
            f"{s['retrieval']['mrr@5']:.3f} | "
            f"{s['retrieval']['hit@5']:.3f} | "
            f"{s['ragas']['ragas_avg']:.3f} | "
            f"{s['ragas']['faithfulness']:.3f} | "
            f"{s['ragas']['answer_relevancy']:.3f} | "
            f"{s['ragas']['context_precision']:.3f} | "
            f"{s['ragas']['context_recall']:.3f} | "
            f"{s['ragas']['answer_correctness']:.3f} | "
            f"{s['ragas']['behavior_alignment']:.3f} | "
            f"{s['avg_latency_s']:.2f}s | "
            f"{s['total_latency_s']:.2f}s | "
            f"{s['avg_tokens']:.1f} | "
            f"{s['total_tokens']} |"
        )
    return "\n".join(lines)


def _delta_table(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    d = summary["drag"]
    rows = [
        ("Recall@5", d["retrieval"]["recall@5"] - b["retrieval"]["recall@5"]),
        ("MRR@5", d["retrieval"]["mrr@5"] - b["retrieval"]["mrr@5"]),
        ("Hit@5", d["retrieval"]["hit@5"] - b["retrieval"]["hit@5"]),
        ("RAGAS avg", d["ragas"]["ragas_avg"] - b["ragas"]["ragas_avg"]),
        ("Faithfulness", d["ragas"]["faithfulness"] - b["ragas"]["faithfulness"]),
        ("Answer relevancy", d["ragas"]["answer_relevancy"] - b["ragas"]["answer_relevancy"]),
        ("Context precision", d["ragas"]["context_precision"] - b["ragas"]["context_precision"]),
        ("Context recall", d["ragas"]["context_recall"] - b["ragas"]["context_recall"]),
        ("Answer correctness", d["ragas"]["answer_correctness"] - b["ragas"]["answer_correctness"]),
        ("Behavior alignment", d["ragas"]["behavior_alignment"] - b["ragas"]["behavior_alignment"]),
        ("Avg latency", d["avg_latency_s"] - b["avg_latency_s"]),
        ("Avg tokens", d["avg_tokens"] - b["avg_tokens"]),
    ]
    lines = ["| Metric | Delta |", "|---|---:|"]
    for name, value in rows:
        suffix = "s" if name == "Avg latency" else ""
        lines.append(f"| {name} | {value:+.3f}{suffix} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
