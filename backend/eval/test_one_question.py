"""Dry-run: ingest PDF then evaluate one question with all 5 configs.

Usage (from backend/):
    # From JSON benchmark
    python -m eval.test_one_question --item-id tt29_easy_01

    # From markdown golden dataset
    python -m eval.test_one_question --md ../data/golden-dataset/test_paper.md --item-id 1 \
        --pdf ../data/pdf/NIL_ds_CIKM.pdf

    python -m eval.test_one_question --reset     # wipe test storages
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from app.agent.llm import LLM
from app.agent.query_transform import QueryTransformer
from eval.run_techniques_ab import (
    CONFIGS,
    BenchmarkItem,
    _compute_metrics,
    _judge_relevant,
    _make_settings,
    _snap,
    _delta,
    load_benchmark,
    retrieve_baseline,
    retrieve_t1,
    retrieve_t4,
    retrieve_t8,
    retrieve_t10,
)
from app.indexing.store import KnowledgeBase

BENCHMARK  = Path("../data/benchmark/context_retrieval_benchmark.json").resolve()
STORAGE_NH = Path("storage_test_noheader").resolve()
STORAGE_H  = Path("storage_test_header").resolve()
POOL_K     = 15
EVAL_K     = 5


def load_md_benchmark(path: Path, pdf_name: str = "") -> list[BenchmarkItem]:
    """Load BenchmarkItem list từ golden dataset markdown (5 cột)."""
    items: list[BenchmarkItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        stt = cells[0].strip()
        if not re.fullmatch(r"\d+", stt):
            continue
        difficulty = re.sub(r"\*+", "", cells[1]).strip()
        question   = cells[2].strip()
        expected   = cells[3].strip()
        page_text  = cells[4].strip() if len(cells) > 4 else ""

        # Parse "Trang 1, 2" hoặc "Trang 1-3"
        page_nums: list[int] = []
        for m in re.finditer(r"Trang\s+([\d,\s\-]+)", page_text, re.IGNORECASE):
            for p in re.findall(r"\d+", m.group(1)):
                page_nums.append(int(p))
        gold_context = [{"page": p} for p in page_nums]

        items.append(BenchmarkItem(
            item_id=stt,
            doc_id=pdf_name,
            difficulty=difficulty,
            answerability="answerable",
            question=question,
            expected_answer=expected,
            gold_context=gold_context,
        ))
    return items


def ingest_pdfs(kb: KnowledgeBase, pdf_paths: list[Path]) -> None:
    existing = {d["source"] for d in kb.repo.list_documents()} | \
               {d["title"]  for d in kb.repo.list_documents()}
    for pdf in pdf_paths:
        if pdf.name in existing or pdf.stem in existing:
            print(f"  [skip] {pdf.name}")
            continue
        print(f"  ingesting {pdf.name} ({pdf.stat().st_size // 1024} KB)...")
        t0 = time.perf_counter()
        kb.ingest_pdf(pdf.read_bytes(), pdf.name)
        print(f"    done in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item-id", default="1",
                    help="ID câu hỏi (số thứ tự trong MD, hoặc item_id trong JSON)")
    ap.add_argument("--md", default=None,
                    help="Đường dẫn file markdown golden dataset (5 cột)")
    ap.add_argument("--pdf", default=None,
                    help="Đường dẫn PDF cần ingest")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        import shutil
        for p in [STORAGE_NH, STORAGE_H]:
            if p.exists():
                shutil.rmtree(p)
                print(f"Removed {p}")

    # ── Setup KBs ─────────────────────────────────────────────────────────────
    print("\n=== Setting up KnowledgeBases ===")
    settings_nh = _make_settings(STORAGE_NH, headers_on=False)
    settings_h  = _make_settings(STORAGE_H,  headers_on=True)
    kb_nh = KnowledgeBase(settings_nh)
    kb_h  = KnowledgeBase(settings_h)

    # ── Ingest PDF ─────────────────────────────────────────────────────────────
    pdf_paths: list[Path] = []
    if args.pdf:
        p = Path(args.pdf)
        if not p.is_absolute():
            p = p.resolve()
        if not p.exists():
            raise SystemExit(f"PDF not found: {p}")
        pdf_paths = [p]

    if pdf_paths:
        print(f"\n=== Ingesting {len(pdf_paths)} PDF(s) ===")
        ingest_pdfs(kb_nh, pdf_paths)
        ingest_pdfs(kb_h,  pdf_paths)

    # ── Pre-build T10 parent index ────────────────────────────────────────────
    print("\n=== Building hierarchical parent index (T10) ===")
    kb_nh._rebuild_hierarchical_indexes(kb_nh.repo.all_chunks_with_embeddings())
    print(f"  parent nodes: {len(kb_nh.parent_nodes)}")

    # ── Load question ─────────────────────────────────────────────────────────
    if args.md:
        md_path = Path(args.md)
        if not md_path.is_absolute():
            md_path = md_path.resolve()
        pdf_name = pdf_paths[0].name if pdf_paths else ""
        all_items = load_md_benchmark(md_path, pdf_name)
    else:
        all_items = load_benchmark(BENCHMARK)

    matches = [it for it in all_items if it.item_id == args.item_id]
    if not matches:
        ids_available = [it.item_id for it in all_items]
        raise SystemExit(f"Item '{args.item_id}' not found. Available: {ids_available}")
    item: BenchmarkItem = matches[0]

    print(f"\n{'='*70}")
    print(f"Question [{item.item_id}] ({item.difficulty}) — {item.doc_id}")
    print(f"  {item.question}")
    print(f"Expected: {item.expected_answer[:150]}...")
    print(f"Gold pages: {[g['page'] for g in item.gold_context]}")
    print(f"{'='*70}")

    # ── Retrieve ──────────────────────────────────────────────────────────────
    llm         = LLM()
    transformer = QueryTransformer(llm, max_variants=settings_nh.query_transform_max_variants)
    retrieved:      dict[str, list[int]]   = {}
    retrieved_info: dict[str, list[dict]]  = {}
    latency_s:      dict[str, float]       = {}

    def _timed(name, fn, *a):
        t = time.perf_counter()
        ids, info = fn(*a)
        latency_s[name] = round(time.perf_counter() - t, 3)
        retrieved_info[name] = info
        return ids

    print("\n--- Retrieving ---")
    retrieved["baseline"]           = _timed("baseline",           retrieve_baseline, kb_nh, item.question, POOL_K)
    retrieved["T4_ctxt_headers"]    = _timed("T4_ctxt_headers",    retrieve_t4,       kb_h,  item.question, POOL_K)
    retrieved["T8_rse"]             = _timed("T8_rse",             retrieve_t8,       kb_nh, item.question, POOL_K)
    retrieved["T10_hierarchical"]   = _timed("T10_hierarchical",   retrieve_t10,      kb_nh, item.question, POOL_K)

    snap_t1 = _snap(llm)
    retrieved["T1_query_transform"] = _timed("T1_query_transform", retrieve_t1, kb_nh, item.question, POOL_K, transformer)
    t1_tokens = _delta(snap_t1, _snap(llm))

    for cfg, ids in retrieved.items():
        print(f"  {CONFIGS[cfg]['label']:38s} {len(ids)} candidates  ({latency_s[cfg]:.2f}s)")

    # ── Judge ─────────────────────────────────────────────────────────────────
    print("\n--- Judging relevance ---")
    cache: dict = {}

    pool_nh = list(dict.fromkeys(
        retrieved["baseline"] + retrieved["T1_query_transform"]
        + retrieved["T8_rse"] + retrieved["T10_hierarchical"]
    ))
    meta_nh = kb_nh.repo.get_chunks(pool_nh)
    rel_nh, usage_nh = _judge_relevant(llm, item, pool_nh, meta_nh, cache, "nh")
    print(f"  noheader pool: {len(pool_nh)} candidates → {len(rel_nh)} relevant  "
          f"tokens={usage_nh['total_tokens']}")

    pool_h  = retrieved["T4_ctxt_headers"]
    meta_h  = kb_h.repo.get_chunks(pool_h)
    rel_h, usage_h = _judge_relevant(llm, item, pool_h, meta_h, cache, "h")
    print(f"  header pool:   {len(pool_h)} candidates → {len(rel_h)} relevant  "
          f"tokens={usage_h['total_tokens']}")

    print(f"\n  T1 transform tokens: {t1_tokens['total_tokens']} "
          f"(prompt={t1_tokens['prompt_tokens']}, completion={t1_tokens['completion_tokens']})")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Config':<38} {'R@5':>5} {'R@10':>6} {'P@5':>5} {'P@10':>6} "
          f"{'MRR@5':>6} {'nDCG@5':>7} {'CtxP':>5} {'CtxR':>5} {'Redn':>5} {'lat':>6}")
    print(f"{'-'*70}")

    config_map = {
        "baseline":           (retrieved["baseline"],           rel_nh, meta_nh),
        "T1_query_transform": (retrieved["T1_query_transform"], rel_nh, meta_nh),
        "T4_ctxt_headers":    (retrieved["T4_ctxt_headers"],    rel_h,  meta_h),
        "T8_rse":             (retrieved["T8_rse"],             rel_nh, meta_nh),
        "T10_hierarchical":   (retrieved["T10_hierarchical"],   rel_nh, meta_nh),
    }

    for cfg, (ids, rel, meta) in config_map.items():
        m = _compute_metrics(ids, rel, meta, EVAL_K, latency_s[cfg], item.gold_context,
                             chunk_info=retrieved_info.get(cfg))
        print(
            f"  {CONFIGS[cfg]['label']:<36} "
            f"{m['recall@5']:>5.2f} {m['recall@10']:>6.2f} "
            f"{m['precision@5']:>5.2f} {m['precision@10']:>6.2f} "
            f"{m['mrr@5']:>6.2f} {m['ndcg@5']:>7.2f} "
            f"{m['context_precision']:>5.2f} {m['context_recall']:>5.2f} "
            f"{m['redundancy_rate']:>5.2f} {latency_s[cfg]:>5.2f}s"
        )

    print(f"{'='*70}")
    print(f"\nDocuments indexed: {len(kb_nh.repo.list_documents())}")
    for d in kb_nh.repo.list_documents():
        print(f"  [{d['id']}] {d['title']}")


if __name__ == "__main__":
    main()
