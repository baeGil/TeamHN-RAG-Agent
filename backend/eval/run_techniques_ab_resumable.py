"""Resumable retrieval-only A/B evaluation.

Writes one JSONL row per question immediately, so interrupted runs can resume.

Usage:
    python -m eval.run_techniques_ab_resumable \
      --benchmark ../data/benchmark/golden_dataset_5files_imported8_benchmark.json \
      --storage-noheader storage_imported_8pdf_noheader \
      --storage-header storage_imported_8pdf_header \
      --out-dir ../data/ab_golden5_imported8_resumable \
      --use-reranker false
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
    POOL_PER_CONFIG,
    _add_retrieval_token_cost,
    _add_usage,
    _build_header_kb_from_noheader,
    _compute_metrics,
    _delta,
    _delta_token_usage,
    _header_storage_ready,
    _ingest_benchmark_pdfs,
    _make_settings,
    _retrieval_token_cost,
    _snap,
    _snap_embedder,
    _zero_retrieval_token_cost,
    _zero_usage,
    load_benchmark,
    retrieve_baseline,
    retrieve_t1,
    retrieve_t10,
    retrieve_t4,
    retrieve_t8,
    _judge_relevant,
)


CONFIG_TO_RETRIEVER = {
    "baseline": retrieve_baseline,
    "T1_query_transform": retrieve_t1,
    "T4_ctxt_headers": retrieve_t4,
    "T8_rse": retrieve_t8,
    "T10_hierarchical": retrieve_t10,
}


def _bool_arg(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "complete" and row.get("item_id"):
                done.add(row["item_id"])
    return done


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _retrieve_one(
    cfg: str,
    kb_nh: KnowledgeBase,
    kb_h: KnowledgeBase,
    question: str,
    pool_k: int,
    llm: LLM,
    transformer: QueryTransformer,
) -> tuple[list[int], list[dict], float, dict, dict]:
    kb = kb_h if CONFIGS[cfg]["kb_group"] == "header" else kb_nh
    fn = CONFIG_TO_RETRIEVER[cfg]
    llm_before = _snap(llm)
    emb_before = _snap_embedder(kb)
    started = time.perf_counter()
    if cfg == "T1_query_transform":
        ids, info = fn(kb, question, pool_k, transformer)
    else:
        ids, info = fn(kb, question, pool_k)
    latency_s = round(time.perf_counter() - started, 3)
    token_cost = _retrieval_token_cost(
        _delta(llm_before, _snap(llm)),
        _delta_token_usage(emb_before, _snap_embedder(kb)),
    )
    return ids, info, latency_s, dict(kb.last_retrieve_latency), token_cost


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumable retrieval-only A/B eval")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--storage-noheader", required=True)
    ap.add_argument("--storage-header", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--pool-per-config", type=int, default=POOL_PER_CONFIG)
    ap.add_argument("--configs", default="baseline,T1_query_transform,T4_ctxt_headers,T8_rse,T10_hierarchical")
    ap.add_argument("--use-reranker", default=None, help="true/false; overrides settings")
    ap.add_argument("--limit", type=int, default=0, help="Max new questions to process this run")
    ap.add_argument("--start-after", default="", help="Skip until after this item id")
    ap.add_argument("--reset", action="store_true", help="Delete partial output and judge cache")
    ap.add_argument("--no-ingest", action="store_true", help="Assume storage DBs already exist; do not ingest documents from benchmark")
    args = ap.parse_args()

    benchmark_path = Path(args.benchmark).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "partial_results.jsonl"
    progress_path = out_dir / "progress.json"
    cache_path = out_dir / "judge_cache.json"

    if args.reset:
        rows_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}")

    settings_nh = _make_settings(Path(args.storage_noheader).resolve(), headers_on=False)
    settings_h = _make_settings(Path(args.storage_header).resolve(), headers_on=True)
    if args.use_reranker is not None:
        use_reranker = _bool_arg(args.use_reranker)
        settings_nh.use_reranker = use_reranker
        settings_h.use_reranker = use_reranker

    kb_nh = KnowledgeBase(settings_nh)
    if not args.no_ingest:
        print("=== Ingest noheader, if benchmark has documents ===")
        _ingest_benchmark_pdfs(kb_nh, benchmark_path)

    if _header_storage_ready(settings_h, kb_nh):
        kb_h = KnowledgeBase(settings_h)
    else:
        kb_h = _build_header_kb_from_noheader(kb_nh, settings_h, settings_h.storage_dir)
    if args.use_reranker is not None:
        kb_nh.settings.use_reranker = _bool_arg(args.use_reranker)
        kb_h.settings.use_reranker = _bool_arg(args.use_reranker)

    print("=== Pre-build hierarchical parent index ===")
    kb_nh._rebuild_hierarchical_indexes(kb_nh.repo.all_chunks_with_embeddings())

    llm = LLM()
    transformer = QueryTransformer(llm, max_variants=settings_nh.query_transform_max_variants)
    judge_cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    items = [it for it in load_benchmark(benchmark_path) if it.answerability == "answerable"]
    done = _read_done(rows_path)
    start_seen = not bool(args.start_after)
    processed = 0

    print(f"Dataset={len(items)} done={len(done)} configs={','.join(selected)}")
    with rows_path.open("a", encoding="utf-8") as out:
        for idx, item in enumerate(items, start=1):
            if not start_seen:
                start_seen = item.item_id == args.start_after
                continue
            if item.item_id in done:
                continue
            if args.limit and processed >= args.limit:
                break

            row_started = time.perf_counter()
            print(f"[{idx}/{len(items)}] {item.item_id} {item.question[:80]}...")
            retrieved: dict[str, list[int]] = {}
            info: dict[str, list[dict]] = {}
            latency: dict[str, float] = {}
            latency_breakdown: dict[str, dict] = {}
            retrieval_tokens: dict[str, dict] = {}
            total_retrieval_tokens = _zero_retrieval_token_cost()

            for cfg in selected:
                ids, cfg_info, lat_s, lb, tc = _retrieve_one(
                    cfg, kb_nh, kb_h, item.question, args.pool_per_config, llm, transformer
                )
                retrieved[cfg] = ids
                info[cfg] = cfg_info
                latency[cfg] = lat_s
                latency_breakdown[cfg] = lb
                retrieval_tokens[cfg] = tc
                total_retrieval_tokens = _add_retrieval_token_cost(total_retrieval_tokens, tc)

            noheader_cfgs = [c for c in selected if CONFIGS[c]["kb_group"] == "noheader"]
            header_cfgs = [c for c in selected if CONFIGS[c]["kb_group"] == "header"]

            rel_by_group: dict[str, set[int]] = {"noheader": set(), "header": set()}
            judge_usage: dict[str, dict] = {}
            if noheader_cfgs:
                pool = list(dict.fromkeys([cid for c in noheader_cfgs for cid in retrieved[c]]))
                meta = kb_nh.repo.get_chunks(pool)
                rel, usage = _judge_relevant(llm, item, pool, meta, judge_cache, f"{item.item_id}_nh")
                rel_by_group["noheader"] = rel
                judge_usage["noheader"] = usage
            if header_cfgs:
                pool = list(dict.fromkeys([cid for c in header_cfgs for cid in retrieved[c]]))
                meta = kb_h.repo.get_chunks(pool)
                rel, usage = _judge_relevant(llm, item, pool, meta, judge_cache, f"{item.item_id}_h")
                rel_by_group["header"] = rel
                judge_usage["header"] = usage
            cache_path.write_text(json.dumps(judge_cache, ensure_ascii=False, indent=2), encoding="utf-8")

            metrics: dict[str, dict] = {}
            for cfg in selected:
                kb = kb_h if CONFIGS[cfg]["kb_group"] == "header" else kb_nh
                rel = rel_by_group[CONFIGS[cfg]["kb_group"]]
                meta = kb.repo.get_chunks(retrieved[cfg])
                metrics[cfg] = _compute_metrics(
                    retrieved[cfg],
                    rel,
                    meta,
                    args.top_k,
                    latency[cfg],
                    item.gold_context,
                    latency_breakdown[cfg],
                    info[cfg],
                )

            row = {
                "status": "complete",
                "item_id": item.item_id,
                "doc_id": item.doc_id,
                "difficulty": item.difficulty,
                "question": item.question,
                "configs": metrics,
                "retrieval_token_cost": retrieval_tokens,
                "retrieval_token_cost_total": total_retrieval_tokens,
                "judge_usage": judge_usage,
                "elapsed_s": round(time.perf_counter() - row_started, 3),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            processed += 1
            done.add(item.item_id)
            _write_json(progress_path, {
                "last_item_id": item.item_id,
                "processed_this_run": processed,
                "completed_total": len(done),
                "dataset_total": len(items),
                "rows_file": str(rows_path),
            })
            print(f"  wrote {item.item_id}; done={len(done)}/{len(items)}")

    print(f"Done this run: {processed}. Rows: {rows_path}")


if __name__ == "__main__":
    main()
