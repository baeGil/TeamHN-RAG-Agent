"""Compare baseline RAG with DRAG on data/data_conflict.

Usage from backend/:
    python -m eval.conflict_benchmark --limit 10

Outputs:
    ../data/data_conflict/conflict_benchmark_results.json
    ../data/data_conflict/conflict_benchmark_report.md
    ../data/data_conflict/conflict_answer_comparison.md
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.indexing.store import KnowledgeBase
from eval.drag_benchmark import (
    POOL_K,
    DragQA,
    aggregate,
    by_group,
    judge_ragas_metrics,
    judge_relevance,
    run_baseline,
    run_drag,
    _usage_total,
)


DEFAULT_QIDS = [
    "DC007",   # Simple, World Bank
    "DC014",  # Simple w. condition, temporal VAT
    "DC027",  # Set
    "DC039",  # Comparison
    "DC051",  # Aggregation
    "DC063",  # Multi-hop
    "DC076",  # Post-processing
    "DC089",  # False premise
    "DC032",  # Set, World Bank
    "DC070",  # Multi-hop, freshness
]


def load_conflict_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_drag_qa(row: dict[str, Any]) -> DragQA:
    return DragQA(
        qid=str(row.get("qid") or ""),
        question=str(row["question"]),
        expected=str(row["expected_answer"]),
        conflict_type=str(row.get("conflict_type", "no_conflict")),
        difficulty=str(row.get("difficulty", "medium")),
        expected_behavior=str(row.get("expected_behavior", "")),
    )


def select_rows(rows: list[dict[str, Any]], qids: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    if qids:
        wanted = set(qids)
        selected = [r for r in rows if r.get("qid") in wanted]
        missing = [qid for qid in qids if qid not in {r.get("qid") for r in selected}]
        if missing:
            raise SystemExit(f"Question id not found: {', '.join(missing)}")
        return selected
    if limit and limit <= len(DEFAULT_QIDS):
        wanted = set(DEFAULT_QIDS[:limit])
        return [r for r in rows if r.get("qid") in wanted]
    if limit:
        return rows[:limit]
    return [r for r in rows if r.get("qid") in set(DEFAULT_QIDS)]


def ensure_conflict_kb(pdf_root: Path, storage: Path, reset: bool, vlm_parse: str) -> KnowledgeBase:
    if reset:
        resolved = storage.resolve()
        cwd = Path.cwd().resolve()
        if cwd not in resolved.parents and resolved != cwd:
            raise SystemExit(f"Refusing to reset storage outside cwd: {resolved}")
        shutil.rmtree(resolved, ignore_errors=True)

    # Keep loader settings and KnowledgeBase settings pointed at the same eval storage.
    os.environ["STORAGE_DIR"] = str(storage.resolve())
    os.environ["VLM_PARSE"] = vlm_parse
    get_settings.cache_clear()

    settings = Settings()
    if not settings.has_openai:
        raise SystemExit("OPENAI_API_KEY is not configured in backend/.env")
    settings.storage_dir = storage.resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    kb = KnowledgeBase(settings)
    if kb.repo.list_documents():
        return kb

    pdfs = sorted(pdf_root.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {pdf_root}")

    for pdf in pdfs:
        print(f"Ingesting {pdf.relative_to(pdf_root.parent)} ...")
        try:
            kb.ingest_pdf(pdf.read_bytes(), str(pdf.relative_to(pdf_root.parent)).replace("\\", "/"))
        except Exception as exc:
            print(f"  skipped: {exc}")

    if not kb.repo.list_documents():
        raise SystemExit("No documents were ingested.")
    return kb


def write_conflict_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Data Conflict: DRAG vs Baseline",
        "",
        f"- Dataset: `{results['dataset']}`",
        f"- PDF root: `{results['pdf_root']}`",
        f"- Questions: **{results['n_questions']}**",
        "",
        "## Overall",
        "",
        summary_table(results["overall"]),
        "",
        "## Per Question",
        "",
        "| QID | Q type | Conflict | Baseline RAGAS | DRAG RAGAS | Baseline behavior | DRAG behavior | DRAG predicted |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results["per_question"]:
        lines.append(
            f"| {row['qid']} | {row['question_type']} | {row['conflict_type']} | "
            f"{row['baseline']['ragas']['ragas_avg']:.3f} | "
            f"{row['drag']['ragas']['ragas_avg']:.3f} | "
            f"{row['baseline']['ragas']['behavior_alignment']:.3f} | "
            f"{row['drag']['ragas']['behavior_alignment']:.3f} | "
            f"{row['drag']['conflict_assessment']['conflict_type']} |"
        )
    lines.extend(["", "## Delta", "", delta_table(results["overall"])])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_conflict_comparison(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# So sánh câu trả lời Baseline vs DRAG",
        "",
        f"- Dataset: `{results['dataset']}`",
        f"- Số câu hỏi: **{results['n_questions']}**",
        "",
    ]
    for row in results["per_question"]:
        lines.extend(
            [
                f"## {row['qid']} - {row['question_type']} - {row['conflict_type']}",
                "",
                f"**Câu hỏi:** {row['question']}",
                "",
                f"**Expected:** {row['expected']}",
                "",
                f"**Evidence:** {row.get('evidence_pages', '')}",
                "",
                (
                    f"**Baseline:** RAGAS={row['baseline']['ragas']['ragas_avg']:.3f}, "
                    f"Behavior={row['baseline']['ragas']['behavior_alignment']:.3f}, "
                    f"Latency={row['baseline']['elapsed']:.1f}s"
                ),
                "",
                row["baseline"]["answer"].strip(),
                "",
                (
                    f"**DRAG:** RAGAS={row['drag']['ragas']['ragas_avg']:.3f}, "
                    f"Behavior={row['drag']['ragas']['behavior_alignment']:.3f}, "
                    f"Latency={row['drag']['elapsed']:.1f}s, "
                    f"Predicted={row['drag']['conflict_assessment']['conflict_type']}"
                ),
                "",
                row["drag"]["answer"].strip(),
                "",
                f"**Judge baseline:** {row['baseline']['ragas'].get('reason', '')}",
                "",
                f"**Judge DRAG:** {row['drag']['ragas'].get('reason', '')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | Recall@5 | MRR@5 | Hit@5 | RAGAS avg | Behavior | Avg latency | Total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("baseline", "drag"):
        row = summary[method]
        lines.append(
            f"| {method} | {row['retrieval']['recall@5']:.3f} | "
            f"{row['retrieval']['mrr@5']:.3f} | "
            f"{row['retrieval']['hit@5']:.3f} | "
            f"{row['ragas']['ragas_avg']:.3f} | "
            f"{row['ragas']['behavior_alignment']:.3f} | "
            f"{row['avg_latency_s']:.1f}s | "
            f"{row['total_tokens']} |"
        )
    return "\n".join(lines)


def delta_table(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    d = summary["drag"]
    vals = [
        ("Recall@5", d["retrieval"]["recall@5"] - b["retrieval"]["recall@5"]),
        ("MRR@5", d["retrieval"]["mrr@5"] - b["retrieval"]["mrr@5"]),
        ("RAGAS avg", d["ragas"]["ragas_avg"] - b["ragas"]["ragas_avg"]),
        ("Behavior", d["ragas"]["behavior_alignment"] - b["ragas"]["behavior_alignment"]),
        ("Avg latency", d["avg_latency_s"] - b["avg_latency_s"]),
        ("Avg tokens", d["avg_tokens"] - b["avg_tokens"]),
    ]
    lines = ["| Metric | Delta |", "|---|---:|"]
    for name, value in vals:
        if "latency" in name.lower():
            lines.append(f"| {name} | {value:.1f}s |")
        else:
            lines.append(f"| {name} | {value:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../data/data_conflict/conflict_test_qa.csv")
    ap.add_argument("--pdf-root", default="../data/data_conflict")
    ap.add_argument("--storage", default="storage_eval_conflict")
    ap.add_argument("--out-dir", default="../data/data_conflict")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--qid", action="append", help="Run one or more qids. Can be repeated.")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument(
        "--vlm-parse",
        default="off",
        choices=["off", "auto", "on"],
        help="Use auto/on if you want scanned PDF pages transcribed during ingestion.",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset).resolve()
    pdf_root = Path(args.pdf_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = Path(args.storage).resolve()

    kb = ensure_conflict_kb(pdf_root, storage, args.reset, args.vlm_parse)
    all_rows = load_conflict_dataset(dataset_path)
    selected_rows = select_rows(all_rows, args.qid, args.limit)
    print(f"Loaded {len(selected_rows)} conflict questions.")

    relevance_cache_path = out_dir / "conflict_relevance_judgments.json"
    ragas_cache_path = out_dir / "conflict_ragas_judgments.json"
    relevance_cache = json.loads(relevance_cache_path.read_text(encoding="utf-8")) if relevance_cache_path.exists() else {}
    ragas_cache = json.loads(ragas_cache_path.read_text(encoding="utf-8")) if ragas_cache_path.exists() else {}

    results_rows = []
    for raw in selected_rows:
        qa = to_drag_qa(raw)
        print(f"\n[{qa.qid}] {raw.get('question_type')} / {qa.conflict_type}: {qa.question[:90]}")
        baseline = run_baseline(kb, qa)
        drag = run_drag(kb, qa)

        candidate_ids = list(dict.fromkeys(baseline["ranked_ids"][:POOL_K] + drag["ranked_ids"][:POOL_K]))
        relevant = judge_relevance(kb, qa, candidate_ids, relevance_cache, relevance_cache_path)
        for name, result in (("baseline", baseline), ("drag", drag)):
            ids = result["ranked_ids"]
            from eval.metrics import hit_at_k, mrr_at_k, recall_at_k

            result["retrieval"] = {
                "recall@5": recall_at_k(ids, relevant, 5),
                "mrr@5": mrr_at_k(ids, relevant, 5),
                "hit@5": hit_at_k(ids, relevant, 5),
                "top5": ids[:5],
                "n_relevant_in_pool": len(relevant),
            }
            result["ragas"] = judge_ragas_metrics(qa, name, result, ragas_cache, ragas_cache_path)

        row = {
            "qid": qa.qid,
            "question_type": raw.get("question_type", ""),
            "question": qa.question,
            "expected": qa.expected,
            "conflict_type": qa.conflict_type,
            "difficulty": qa.difficulty,
            "expected_behavior": qa.expected_behavior,
            "evidence_pages": raw.get("evidence_pages", ""),
            "source_documents": raw.get("source_documents", ""),
            "baseline": baseline,
            "drag": drag,
        }
        results_rows.append(row)
        print(
            f"  baseline ragas={baseline['ragas']['ragas_avg']:.2f} "
            f"behavior={baseline['ragas']['behavior_alignment']:.2f} "
            f"lat={baseline['elapsed']:.1f}s tok={_usage_total(baseline['usage'])}"
        )
        print(
            f"  drag     ragas={drag['ragas']['ragas_avg']:.2f} "
            f"behavior={drag['ragas']['behavior_alignment']:.2f} "
            f"lat={drag['elapsed']:.1f}s tok={_usage_total(drag['usage'])} "
            f"type={drag['conflict_assessment']['conflict_type']}"
        )

    results = {
        "n_questions": len(results_rows),
        "dataset": str(dataset_path),
        "pdf_root": str(pdf_root),
        "overall": aggregate(results_rows),
        "by_conflict_type": by_group(results_rows, "conflict_type"),
        "by_difficulty": by_group(results_rows, "difficulty"),
        "per_question": results_rows,
    }
    (out_dir / "conflict_benchmark_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_conflict_report(out_dir / "conflict_benchmark_report.md", results)
    write_conflict_comparison(out_dir / "conflict_answer_comparison.md", results)
    print(f"\nReport: {out_dir / 'conflict_benchmark_report.md'}")
    print(f"Answer comparison: {out_dir / 'conflict_answer_comparison.md'}")
    print(f"Results: {out_dir / 'conflict_benchmark_results.json'}")


if __name__ == "__main__":
    main()
