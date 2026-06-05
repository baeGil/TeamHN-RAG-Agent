"""Patch partial_results.jsonl: re-run T1_query_transform with the new implementation
(LLM selects best variant → 1 full retrieve with reranker) for already-processed questions.

- Reads existing partial_results.jsonl (keeps all other configs untouched).
- Re-runs retrieve_t1 (new: LLM picks best variant → 1 full retrieve) for each row.
- Reuses relevance judgments from judge_cache.json (no new LLM judge calls).
- Writes updated rows back to the same JSONL file (atomic: temp file + rename).

Usage (from backend/):
    python -m eval.patch_t1_results \\
      --benchmark ../data/benchmark/golden_dataset_5files_imported8_benchmark.json \\
      --storage-noheader storage_imported_8pdf_noheader \\
      --storage-header   storage_imported_8pdf_header \\
      --out-dir          ../data/ab_golden5_imported8_resumable

NOTE: --use-reranker is omitted so it reads from settings (reranker enabled by default).
Pass --use-reranker false only if the rest of your eval data was also collected without reranker.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.agent.llm import LLM
from app.agent.query_transform import QueryTransformer
from app.indexing.store import KnowledgeBase
from eval.run_techniques_ab import (
    CONFIGS,
    _build_header_kb_from_noheader,
    _compute_metrics,
    _delta,
    _delta_token_usage,
    _header_storage_ready,
    _make_settings,
    _retrieval_token_cost,
    _snap,
    _snap_embedder,
    _zero_usage,
    load_benchmark,
    retrieve_t1,
)


def _bool_arg(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    obj = json.loads(s)
                    if obj.get("status") == "complete":
                        rows.append(obj)
                except json.JSONDecodeError:
                    pass
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch T1 results in partial_results.jsonl")
    ap.add_argument("--benchmark",        required=True)
    ap.add_argument("--storage-noheader", required=True)
    ap.add_argument("--storage-header",   required=True)
    ap.add_argument("--out-dir",          required=True)
    ap.add_argument("--use-reranker",     default=None, help="true/false")
    ap.add_argument("--start-after",      default="",
                    help="Skip rows until after this item_id (resume from a specific question)")
    args = ap.parse_args()

    out_dir      = Path(args.out_dir).resolve()
    rows_path    = out_dir / "partial_results.jsonl"
    cache_path   = out_dir / "judge_cache.json"

    if not rows_path.exists():
        raise SystemExit(f"partial_results.jsonl not found: {rows_path}")
    if not cache_path.exists():
        raise SystemExit(f"judge_cache.json not found: {cache_path}")

    # ── load existing data ────────────────────────────────────────────────────
    existing_rows = _load_jsonl(rows_path)
    judge_cache   = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(existing_rows)} rows from {rows_path.name}")
    print(f"Judge cache has {len(judge_cache)} entries\n")

    # ── build index: item_id → benchmark item ─────────────────────────────────
    benchmark_items = {
        it.item_id: it
        for it in load_benchmark(Path(args.benchmark).resolve())
    }

    # ── set up KnowledgeBases ─────────────────────────────────────────────────
    settings_nh = _make_settings(Path(args.storage_noheader).resolve(), headers_on=False)
    settings_h  = _make_settings(Path(args.storage_header).resolve(),   headers_on=True)
    if args.use_reranker is not None:
        use_rr = _bool_arg(args.use_reranker)
        settings_nh.use_reranker = use_rr
        settings_h.use_reranker  = use_rr

    kb_nh = KnowledgeBase(settings_nh)
    if _header_storage_ready(settings_h, kb_nh):
        kb_h = KnowledgeBase(settings_h)
    else:
        kb_h = _build_header_kb_from_noheader(kb_nh, settings_h, settings_h.storage_dir)

    print("=== Pre-build hierarchical parent index ===")
    kb_nh._rebuild_hierarchical_indexes(kb_nh.repo.all_chunks_with_embeddings())

    llm         = LLM()
    transformer = QueryTransformer(llm, max_variants=settings_nh.query_transform_max_variants)

    # T1 always uses the noheader KB
    kb = kb_nh

    # ── re-run T1 for each existing row ──────────────────────────────────────
    start_seen = not bool(args.start_after)
    updated_rows: list[dict] = []
    for idx, row in enumerate(existing_rows, start=1):
        item_id = row["item_id"]

        # skip rows until we pass --start-after item_id
        if not start_seen:
            updated_rows.append(row)   # keep original, do not re-run
            if item_id == args.start_after:
                start_seen = True
                print(f"  [{idx}] {item_id} — start-after marker reached, patching from next row")
            else:
                print(f"  [{idx}] {item_id} — skipped (before start-after)")
            continue

        item    = benchmark_items.get(item_id)
        if item is None:
            print(f"  [{idx}] {item_id} — not found in benchmark, skipping")
            updated_rows.append(row)
            continue

        # retrieve_t1 with new implementation
        llm_before = _snap(llm)
        emb_before = _snap_embedder(kb)
        t0 = time.perf_counter()
        ids, chunk_info = retrieve_t1(kb, item.question, 15, transformer)
        latency_s = round(time.perf_counter() - t0, 3)
        lb = dict(kb.last_retrieve_latency)
        token_cost = _retrieval_token_cost(
            _delta(llm_before, _snap(llm)),
            _delta_token_usage(emb_before, _snap_embedder(kb)),
        )

        # relevance from cache (no new judge call)
        cache_key = f"{item_id}_nh"
        cached = judge_cache.get(cache_key)
        if cached is None:
            print(f"  [{idx}] {item_id} — relevance not in cache, skipping T1 patch")
            updated_rows.append(row)
            continue
        relevant = set(cached["relevant_ids"] if isinstance(cached, dict) else cached)

        # re-compute metrics
        meta    = kb.repo.get_chunks(ids)
        metrics = _compute_metrics(
            ids, relevant, meta,
            k=5,
            latency_s=latency_s,
            gold_context=item.gold_context,
            latency_breakdown=lb,
            chunk_info=chunk_info,
        )

        old_hit = row["configs"].get("T1_query_transform", {}).get("hit@5", "N/A")
        new_hit = metrics["hit@5"]
        print(f"  [{idx:>3}] {item_id}  hit@5: {old_hit} → {new_hit}  "
              f"latency: {row['configs'].get('T1_query_transform',{}).get('latency_s','?')}s "
              f"→ {latency_s}s")

        # patch only T1 fields
        row["configs"]["T1_query_transform"] = metrics
        row["retrieval_token_cost"]["T1_query_transform"] = token_cost

        # recalculate retrieval_token_cost_total
        total = {k: 0 for k in next(iter(row["retrieval_token_cost"].values()))}
        for cfg_cost in row["retrieval_token_cost"].values():
            for k in total:
                total[k] += cfg_cost.get(k, 0)
        row["retrieval_token_cost_total"] = total

        updated_rows.append(row)

    # ── write back atomically ─────────────────────────────────────────────────
    _write_jsonl_atomic(rows_path, updated_rows)
    print(f"\nPatched {len(updated_rows)} rows → {rows_path}")


if __name__ == "__main__":
    main()
