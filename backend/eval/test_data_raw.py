"""Evaluate Baseline RAG vs DRAG on a subset of data_raw.json.

This script:
1. Ingests contexts from data_raw.json to build a retrieval database.
2. Evaluates the first 20 questions using both Baseline RAG and DRAG RAG.
3. Computes: Recall@5, MRR@5, Hit@5, RAGAS scores (via LLM-as-judge), latency, and tokens.
4. Outputs a comparison report.

Usage from backend/:
    python -m eval.test_data_raw
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.indexing.store import KnowledgeBase
from eval.metrics import hit_at_k, mrr_at_k, recall_at_k
from eval.drag_benchmark import (
    DragQA,
    run_baseline,
    run_drag,
    judge_ragas_metrics,
    _usage_total,
)


def load_raw_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_data_raw_kb(items: list[dict[str, Any]], storage: Path, reset: bool, ingest_limit: int) -> KnowledgeBase:
    if reset and storage.exists():
        print(f"Resetting storage directory: {storage}")
        shutil.rmtree(storage, ignore_errors=True)

    # Point STORAGE_DIR to the eval target
    os.environ["STORAGE_DIR"] = str(storage.resolve())
    get_settings.cache_clear()

    settings = Settings()
    settings.storage_dir = storage.resolve()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    kb = KnowledgeBase(settings)
    if not kb.repo.list_documents():
        print(f"Ingesting top {ingest_limit} contexts from data_raw.json...")
        seen_contexts = set()
        count = 0
        for idx, item in enumerate(items):
            ctx = item["context"].strip()
            if not ctx:
                continue
            if ctx not in seen_contexts:
                seen_contexts.add(ctx)
                title = f"raw_doc_{idx}"
                kb.ingest_text(ctx, title=title)
                count += 1
                if count >= ingest_limit:
                    break
        print(f"Ingestion complete: {count} unique contexts ingested.")
    else:
        print("Using existing database storage.")
    return kb


def write_raw_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Kết quả Đánh giá: Baseline RAG vs DRAG trên data_raw.json",
        "",
        f"- Dataset: `{results['dataset']}`",
        f"- Số câu hỏi đánh giá: **{results['n_questions']}**",
        f"- Số tài liệu ngữ cảnh trong DB: **{results['n_contexts_ingested']}**",
        "",
        "## 1. Kết quả Tổng hợp",
        "",
        _summary_table(results["overall"]),
        "",
        "## 2. So sánh Chênh lệch (Delta: DRAG - Baseline)",
        "",
        _delta_table(results["overall"]),
        "",
        "## 3. Kết quả Chi tiết từng Câu hỏi",
        "",
        "| QID | Phương pháp | Faithfulness | Relevancy | Ctx Precision | Ctx Recall | Correctness | Behavior | Latency | Nhãn dự đoán |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results["per_question"]:
        b = row["baseline"]
        d = row["drag"]
        lines.append(
            f"| **{row['qid']}** | **{row['question'][:70]}...** | | | | | | | | |"
        )
        lines.append(
            f"| | baseline | "
            f"{b['ragas'].get('faithfulness', 0.0):.3f} | "
            f"{b['ragas'].get('answer_relevancy', 0.0):.3f} | "
            f"{b['ragas'].get('context_precision', 0.0):.3f} | "
            f"{b['ragas'].get('context_recall', 0.0):.3f} | "
            f"{b['ragas'].get('answer_correctness', 0.0):.3f} | "
            f"{b['ragas'].get('behavior_alignment', 0.0):.3f} | "
            f"{b['elapsed']:.2f}s | - |"
        )
        lines.append(
            f"| | drag | "
            f"{d['ragas'].get('faithfulness', 0.0):.3f} | "
            f"{d['ragas'].get('answer_relevancy', 0.0):.3f} | "
            f"{d['ragas'].get('context_precision', 0.0):.3f} | "
            f"{d['ragas'].get('context_recall', 0.0):.3f} | "
            f"{d['ragas'].get('answer_correctness', 0.0):.3f} | "
            f"{d['ragas'].get('behavior_alignment', 0.0):.3f} | "
            f"{d['elapsed']:.2f}s | `{d['conflict_assessment']['conflict_type']}` |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Phương pháp | Recall@5 | MRR@5 | Hit@5 | RAGAS Avg | Faithfulness | Relevancy | Correctness | Avg Latency | Total Latency | Avg Tokens | Total Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{s['ragas']['answer_correctness']:.3f} | "
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
        ("RAGAS Avg", d["ragas"]["ragas_avg"] - b["ragas"]["ragas_avg"]),
        ("Avg Latency", d["avg_latency_s"] - b["avg_latency_s"]),
        ("Avg Tokens", d["avg_tokens"] - b["avg_tokens"]),
    ]
    lines = ["| Chỉ số | Chênh lệch (Delta) |", "|---|---:|"]
    for name, value in rows:
        suffix = "s" if "Latency" in name else ""
        lines.append(f"| {name} | {value:+.3f}{suffix} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../data/vietanh-data/data_raw.json")
    parser.add_argument("--storage", default="storage_eval_data_raw")
    parser.add_argument("--out-dir", default="../data/vietanh-data/drag_eval")
    parser.add_argument("--ingest-limit", type=int, default=150, help="Số lượng ngữ cảnh nạp vào để tạo pool tìm kiếm")
    parser.add_argument("--test-limit", type=int, default=20, help="Số lượng câu hỏi đánh giá")
    parser.add_argument("--reset", action="store_true", help="Xoá cơ sở dữ liệu cũ để nạp lại")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = Path(args.storage).resolve()

    items = load_raw_dataset(dataset_path)
    print(f"Loaded {len(items)} items from raw dataset.")

    # 1. Khởi tạo KnowledgeBase và nạp ngữ cảnh
    kb = ensure_data_raw_kb(items, storage, args.reset, args.ingest_limit)

    # Tạo từ điển map từ nội dung text sang chunk_id trong DB
    all_chunks = kb.repo.all_chunks()
    text_to_cid = {c["text"].strip(): int(c["id"]) for c in all_chunks}
    print(f"Mapped {len(text_to_cid)} chunks in SQLite database.")

    # 2. Chuẩn bị tập câu hỏi đánh giá
    test_items = items[:args.test_limit]
    print(f"Evaluating the first {len(test_items)} questions...")

    # Cấu hình caches để giảm tải gọi API trùng lặp nếu chạy lại
    relevance_cache_path = out_dir / "raw_relevance_judgments.json"
    ragas_cache_path = out_dir / "raw_ragas_judgments.json"
    relevance_cache = json.loads(relevance_cache_path.read_text(encoding="utf-8")) if relevance_cache_path.exists() else {}
    ragas_cache = json.loads(ragas_cache_path.read_text(encoding="utf-8")) if ragas_cache_path.exists() else {}

    results_rows = []
    for idx, item in enumerate(test_items, start=1):
        qid = item.get("id") or f"R{idx:03d}"
        q_text = item["question"]
        expected = item["answer"][0] if item["answer"] else "Không rõ"
        context_text = item["context"].strip()

        # Tìm chunk ID chính xác làm ground-truth
        gt_cid = text_to_cid.get(context_text)
        if gt_cid is None:
            print(f"Warning: Context text for QID {qid} not found in database. Ingesting single context...")
            res = kb.ingest_text(context_text, title=f"extra_doc_{qid}")
            # Refresh chunk map
            all_chunks = kb.repo.all_chunks()
            text_to_cid = {c["text"].strip(): int(c["id"]) for c in all_chunks}
            gt_cid = text_to_cid.get(context_text)

        relevant = {gt_cid} if gt_cid is not None else set()

        qa = DragQA(
            qid=qid,
            question=q_text,
            expected=expected,
            conflict_type="no_conflict",
            difficulty="medium",
            expected_behavior="Trả lời trực tiếp dựa trên ngữ cảnh được trích dẫn.",
        )

        print(f"\n[{qid}] Đang chạy: {q_text[:80]}...")
        baseline = run_baseline(kb, qa)
        drag = run_drag(kb, qa)

        # Tính toán các chỉ số
        for name, res_dict in (("baseline", baseline), ("drag", drag)):
            ids = res_dict["ranked_ids"]
            res_dict["retrieval"] = {
                "recall@5": recall_at_k(ids, relevant, 5),
                "mrr@5": mrr_at_k(ids, relevant, 5),
                "hit@5": hit_at_k(ids, relevant, 5),
                "top5": ids[:5],
                "n_relevant_in_pool": len(relevant),
            }
            res_dict["ragas"] = judge_ragas_metrics(qa, name, res_dict, ragas_cache, ragas_cache_path)

        row = {
            "qid": qid,
            "question": q_text,
            "expected": expected,
            "baseline": baseline,
            "drag": drag,
        }
        results_rows.append(row)

        print(
            f"  baseline: recall@5={baseline['retrieval']['recall@5']:.1f} "
            f"ragas={baseline['ragas']['ragas_avg']:.2f} "
            f"latency={baseline['elapsed']:.1f}s"
        )
        print(
            f"  drag:     recall@5={drag['retrieval']['recall@5']:.1f} "
            f"ragas={drag['ragas']['ragas_avg']:.2f} "
            f"latency={drag['elapsed']:.1f}s "
            f"type={drag['conflict_assessment']['conflict_type']}"
        )

    # 3. Tổng hợp kết quả
    methods = ("baseline", "drag")
    overall_summary = {}
    for method in methods:
        n = len(results_rows)
        retrieval = {
            "recall@5": sum(r[method]["retrieval"]["recall@5"] for r in results_rows) / n,
            "mrr@5": sum(r[method]["retrieval"]["mrr@5"] for r in results_rows) / n,
            "hit@5": sum(r[method]["retrieval"]["hit@5"] for r in results_rows) / n,
        }
        ragas_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness", "behavior_alignment", "ragas_avg"]
        ragas = {k: sum(r[method]["ragas"].get(k, 0.0) for r in results_rows) / n for k in ragas_keys}
        total_latency = sum(r[method]["elapsed"] for r in results_rows)
        total_tokens = sum(_usage_total(r[method]["usage"]) for r in results_rows)

        overall_summary[method] = {
            "retrieval": retrieval,
            "ragas": ragas,
            "avg_latency_s": total_latency / n,
            "total_latency_s": total_latency,
            "avg_tokens": total_tokens / n,
            "total_tokens": total_tokens,
        }

    results = {
        "n_questions": len(results_rows),
        "n_contexts_ingested": len(seen_contexts) if "seen_contexts" in locals() else args.ingest_limit,
        "dataset": str(dataset_path),
        "overall": overall_summary,
        "per_question": results_rows,
    }

    # Lưu kết quả
    (out_dir / "raw_benchmark_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_raw_report(out_dir / "raw_benchmark_report.md", results)

    print(f"\nĐã hoàn thành đánh giá!")
    print(f"File kết quả JSON: {out_dir / 'raw_benchmark_results.json'}")
    print(f"Báo cáo Markdown: {out_dir / 'raw_benchmark_report.md'}")


if __name__ == "__main__":
    main()
