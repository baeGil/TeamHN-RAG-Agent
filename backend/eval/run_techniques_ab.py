"""A/B testing pipeline for RAG techniques T1, T4, T8, T10.

Compares 5 configurations against a shared benchmark dataset:
  baseline  – all 4 techniques OFF  (reference)
  T1        – Query Transformation ON
  T4        – Contextual Chunk Headers ON
  T8        – Relevant Segment Extraction ON
  T10       – Hierarchical Indices ON

Metrics (retrieval-only):
  Recall@k, Precision@k, MRR@k, Hit@k, nDCG@k, Redundancy ratio
  Latency per config, Token cost (prompt + completion tokens)

Retrieval token cost breakdown per config:
  llm_*        – LLM calls inside retrieval-time techniques (T1 query transform)
  embedding_*  – OpenAI embedding calls for dense query retrieval

Dataset: data/benchmark/context_retrieval_benchmark.json

Storage layout:
  storage_ab_noheader/  – shared by baseline / T1 / T8 / T10  (headers=OFF)
  storage_ab_header/    – used by T4                           (headers=ON)

Usage (from backend/):
    python -m eval.run_techniques_ab
    python -m eval.run_techniques_ab --reset        # wipe storages and re-ingest
    python -m eval.run_techniques_ab --top-k 10     # change k
    python -m eval.run_techniques_ab --out-dir ../data/ab_results
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.agent.llm import LLM
from app.agent.query_transform import QueryTransformer
from app.config import Settings
from app.indexing.store import KnowledgeBase
from eval.metrics import (
    context_precision_at_k, hit_at_k, mrr_at_k, ndcg_at_k,
    precision_at_k, recall_at_k, redundancy_ratio,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K = 5
POOL_PER_CONFIG = 15   # candidates to pool per config for LLM judge

JUDGE_SYSTEM = (
    "Bạn là giám khảo đánh giá hệ thống truy hồi tài liệu.\n\n"
    "Từ danh sách ĐOẠN ỨNG VIÊN, xác định những đoạn CHỨA thông tin "
    "cần thiết để suy ra đáp án đúng. Chỉ chọn đoạn thực sự chứa bằng chứng.\n\n"
    'Trả về JSON: {"relevant_ids": [<id số nguyên>]}'
)

CONFIGS = {
    "baseline":           {"label": "Baseline",                  "kb_group": "noheader"},
    "T1_query_transform": {"label": "T1 – Query Transformation", "kb_group": "noheader"},
    "T4_ctxt_headers":    {"label": "T4 – Contextual Headers",   "kb_group": "header"},
    "T8_rse":             {"label": "T8 – RSE",                  "kb_group": "noheader"},
    "T10_hierarchical":   {"label": "T10 – Hierarchical Indices","kb_group": "noheader"},
}


# ---------------------------------------------------------------------------
# Benchmark dataset loader
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkItem:
    item_id: str
    doc_id: str
    difficulty: str
    answerability: str
    question: str
    expected_answer: str
    gold_context: list[dict]


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        BenchmarkItem(
            item_id=it["id"],
            doc_id=it["doc_id"],
            difficulty=it["difficulty"],
            answerability=it.get("answerability", "answerable"),
            question=it["question"],
            expected_answer=it.get("expected_answer", ""),
            gold_context=it.get("gold_context", []),
        )
        for it in data.get("items", [])
    ]


# ---------------------------------------------------------------------------
# Token usage helpers
# ---------------------------------------------------------------------------

def _snap(llm: LLM) -> dict:
    """Snapshot current cumulative token usage."""
    return dict(llm.usage)


def _delta(before: dict, after: dict) -> dict:
    """Compute token usage delta between two snapshots."""
    return {
        "prompt_tokens":     after["prompt_tokens"]     - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens":      (after["prompt_tokens"] + after["completion_tokens"])
                             - (before["prompt_tokens"] + before["completion_tokens"]),
        "calls":             after["calls"] - before["calls"],
    }


def _delta_token_usage(before: dict, after: dict) -> dict:
    prompt_before = before.get("prompt_tokens", 0)
    prompt_after = after.get("prompt_tokens", 0)
    completion_before = before.get("completion_tokens", 0)
    completion_after = after.get("completion_tokens", 0)
    total_before = before.get("total_tokens", prompt_before + completion_before)
    total_after = after.get("total_tokens", prompt_after + completion_after)
    return {
        "prompt_tokens": prompt_after - prompt_before,
        "completion_tokens": completion_after - completion_before,
        "total_tokens": total_after - total_before,
        "calls": after.get("calls", 0) - before.get("calls", 0),
    }


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def _snap_nodes(llm) -> dict:
    return {node: dict(usage) for node, usage in llm.node_usage.items()}


def _snap_embedder(kb: KnowledgeBase) -> dict:
    return dict(getattr(kb.embedder, "usage", _zero_usage()))


def _retrieval_token_cost(llm_usage: dict, embedding_usage: dict) -> dict:
    total = _add_usage(llm_usage, embedding_usage)
    return {
        "llm_prompt_tokens": llm_usage["prompt_tokens"],
        "llm_completion_tokens": llm_usage["completion_tokens"],
        "llm_total_tokens": llm_usage["total_tokens"],
        "llm_calls": llm_usage["calls"],
        "embedding_tokens": embedding_usage["total_tokens"],
        "embedding_calls": embedding_usage["calls"],
        "prompt_tokens": total["prompt_tokens"],
        "completion_tokens": total["completion_tokens"],
        "total_tokens": total["total_tokens"],
        "calls": total["calls"],
    }


def _zero_retrieval_token_cost() -> dict:
    return _retrieval_token_cost(_zero_usage(), _zero_usage())


def _add_retrieval_token_cost(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in _zero_retrieval_token_cost()}


def _delta_nodes(before: dict, after: dict) -> dict:
    result = {}
    for node in set(before) | set(after):
        b = before.get(node, {"model": "", "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "total_ms": 0.0})
        a = after.get(node, b)
        delta = {
            "model": a.get("model", ""),
            "prompt_tokens": a["prompt_tokens"] - b.get("prompt_tokens", 0),
            "completion_tokens": a["completion_tokens"] - b.get("completion_tokens", 0),
            "calls": a["calls"] - b.get("calls", 0),
            "total_ms": round(a.get("total_ms", 0.0) - b.get("total_ms", 0.0), 1),
        }
        if delta["calls"] > 0:
            result[node] = delta
    return result


# ---------------------------------------------------------------------------
# KnowledgeBase factory
# ---------------------------------------------------------------------------

def _make_settings(storage_dir: Path, headers_on: bool) -> Settings:
    s = Settings()
    s.storage_dir = storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    s.enable_contextual_chunk_headers = headers_on
    s.enable_query_transformation = False
    s.enable_relevant_segment_extraction = False
    s.enable_hierarchical_indices = False
    return s


def _ingest_benchmark_pdfs(kb: KnowledgeBase, benchmark_path: Path) -> None:
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_root = benchmark_path.parent.parent.parent  # project root
    existing_titles = {d["title"] for d in kb.repo.list_documents()}
    for doc_meta in data.get("documents", []):
        pdf_path = benchmark_root / doc_meta["local_pdf"]
        title = doc_meta["title"]
        if title in existing_titles:
            print(f"  [skip] already indexed: {title}")
            continue
        if not pdf_path.exists():
            print(f"  [WARN] PDF not found: {pdf_path}")
            continue
        print(f"  ingesting: {pdf_path.name} ...")
        t0 = time.perf_counter()
        kb.ingest_pdf(pdf_path.read_bytes(), pdf_path.name)
        print(f"    done in {time.perf_counter()-t0:.1f}s")


def _build_header_kb_from_noheader(
    kb_noheader: KnowledgeBase,
    settings_header: Settings,
    storage_header: Path,
) -> KnowledgeBase:
    """Reuse parsed chunks from noheader storage, then rebuild indexes with headers ON.

    This avoids calling the PDF parser/Reducto twice. Header mode changes index text,
    not the extracted chunk text itself.
    """
    storage_header.mkdir(parents=True, exist_ok=True)
    kb_noheader.db.conn.commit()
    try:
        kb_noheader.db.conn.execute("PRAGMA wal_checkpoint(FULL)")
    except Exception:
        pass

    src_db = kb_noheader.settings.db_path
    dst_db = settings_header.db_path
    shutil.copy2(src_db, dst_db)

    src_pdfs = kb_noheader.settings.storage_dir / "pdfs"
    dst_pdfs = storage_header / "pdfs"
    if dst_pdfs.exists():
        shutil.rmtree(dst_pdfs)
    if src_pdfs.exists():
        shutil.copytree(src_pdfs, dst_pdfs)

    for stale in (settings_header.vector_path, settings_header.bm25_path, settings_header.meta_path):
        stale.unlink(missing_ok=True)

    kb_header = KnowledgeBase(settings_header)
    kb_header.rebuild_indexes(force_reembed=True)
    return kb_header


def _header_storage_ready(settings_header: Settings, kb_noheader: KnowledgeBase) -> bool:
    if not settings_header.db_path.exists():
        return False
    try:
        kb_header = KnowledgeBase(settings_header)
        return len(kb_header.repo.all_chunks()) == len(kb_noheader.repo.all_chunks())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Retrieval helpers (one function per config)
# ---------------------------------------------------------------------------

def _chunk_ids(chunks) -> list[int]:
    return [c.chunk_id for c in chunks]


def _chunk_info(chunks) -> list[dict]:
    """Extract rank + score info for each chunk (post-rerank order is the input order)."""
    pre_rerank = sorted(range(len(chunks)), key=lambda i: chunks[i].rrf_score, reverse=True)
    pre_rank = {chunks[i].chunk_id: rank + 1 for rank, i in enumerate(pre_rerank)}
    return [
        {
            "chunk_id":         h.chunk_id,
            "rank_pre_rerank":  pre_rank[h.chunk_id],
            "rank_post_rerank": rank + 1,
            "rrf_score":        round(h.rrf_score, 6),
            "bm25_score":       round(h.bm25_score, 6) if h.bm25_score is not None else None,
            "dense_score":      round(h.dense_score, 6) if h.dense_score is not None else None,
            "rerank_score":     round(h.rerank_score, 6) if h.rerank_score is not None else None,
        }
        for rank, h in enumerate(chunks)
    ]


def retrieve_baseline(kb: KnowledgeBase, query: str, k: int) -> tuple[list[int], list[dict]]:
    kb.settings.enable_query_transformation = False
    kb.settings.enable_relevant_segment_extraction = False
    kb.settings.enable_hierarchical_indices = False
    chunks = kb.retrieve(query, k)
    return _chunk_ids(chunks), _chunk_info(chunks)


def retrieve_t1(kb: KnowledgeBase, query: str, k: int,
                transformer: QueryTransformer) -> tuple[list[int], list[dict]]:
    """Query Transformation: LLM generates variants AND selects the best one, retrieve once.

    1. One LLM call generates up to 3 variants (original / rewrite / step_back) and
       returns a "best" field indicating which variant to use for retrieval.
    2. QueryTransformer.transform() reorders variants so variants[0] is LLM-selected.
    3. Full pipeline (BM25 + Dense + RRF + optional Rerank) runs once for variants[0].
    Result: 1 LLM call + 1 rerank call — same cost as baseline retrieval plus variant generation.
    """
    kb.settings.enable_relevant_segment_extraction = False
    kb.settings.enable_hierarchical_indices = False
    variants = transformer.transform(query, "simple")
    # variants[0] is the LLM-selected best variant (see QueryTransformer.transform)
    chunks = kb.retrieve(variants[0].query, k)
    return _chunk_ids(chunks), _chunk_info(chunks)


def retrieve_t4(kb_header: KnowledgeBase, query: str, k: int) -> tuple[list[int], list[dict]]:
    kb_header.settings.enable_relevant_segment_extraction = False
    kb_header.settings.enable_hierarchical_indices = False
    chunks = kb_header.retrieve(query, k)
    return _chunk_ids(chunks), _chunk_info(chunks)


def retrieve_t8(kb: KnowledgeBase, query: str, k: int) -> tuple[list[int], list[dict]]:
    kb.settings.enable_relevant_segment_extraction = True
    kb.settings.enable_hierarchical_indices = False
    chunks = kb.retrieve(query, k)
    kb.settings.enable_relevant_segment_extraction = False
    return _chunk_ids(chunks), _chunk_info(chunks)


def retrieve_t10(kb: KnowledgeBase, query: str, k: int) -> tuple[list[int], list[dict]]:
    kb.settings.enable_relevant_segment_extraction = False
    kb.settings.enable_hierarchical_indices = True
    chunks = kb.retrieve(query, k)
    kb.settings.enable_hierarchical_indices = False
    return _chunk_ids(chunks), _chunk_info(chunks)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _judge_relevant(
    llm: LLM,
    item: BenchmarkItem,
    candidate_ids: list[int],
    meta: dict,
    cache: dict,
    cache_key: str,
) -> tuple[set[int], dict]:
    """Return (relevant_ids, token_usage). Reads from cache if available (usage = zero)."""
    if cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, list):
            return set(cached), _zero_usage()
        return set(cached["relevant_ids"]), _zero_usage()

    passages = []
    for cid in candidate_ids:
        if cid in meta:
            txt = meta[cid].get("text", "")[:450].replace("\n", " ")
            page = meta[cid].get("page", "?")
            section = (meta[cid].get("section") or "")[:40]
            passages.append(f"[id={cid}, trang={page}, mục={section}] {txt}")

    if not passages:
        cache[cache_key] = {"relevant_ids": []}
        return set(), _zero_usage()

    user_msg = (
        f"CÂU HỎI:\n{item.question}\n\n"
        f"ĐÁP ÁN MONG ĐỢI:\n{item.expected_answer}\n\n"
        f"CÁC ĐOẠN ỨNG VIÊN:\n" + "\n\n".join(passages)
    )

    before = _snap(llm)
    try:
        res = llm.chat_json(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user_msg}],
            fast=True,
        )
        rel = [int(x) for x in res.get("relevant_ids", []) if x in set(candidate_ids)]
    except Exception as e:
        print(f"    [judge error] {e}")
        rel = []
    usage = _delta(before, _snap(llm))

    cache[cache_key] = {"relevant_ids": rel}
    return set(rel), usage


# ---------------------------------------------------------------------------
# Metrics per config
# ---------------------------------------------------------------------------

def _context_recall_from_gold(ids: list[int], meta: dict, gold_context: list[dict]) -> float:
    """Tỷ lệ gold context items có ít nhất 1 chunk retrieved cùng page."""
    if not gold_context:
        return 0.0
    retrieved_pages = {meta[cid].get("page") for cid in ids if cid in meta}
    covered = sum(1 for g in gold_context if g.get("page") in retrieved_pages)
    return round(covered / len(gold_context), 4)


def _compute_metrics(ids: list[int], relevant: set[int], meta: dict, k: int,
                     latency_s: float, gold_context: list[dict],
                     latency_breakdown: Optional[dict] = None,
                     chunk_info: Optional[list[dict]] = None) -> dict:
    top = ids[:k]
    texts = [meta[cid]["text"] for cid in top if cid in meta]
    return {
        # IR metrics (chunk-level)
        "recall@5":              round(recall_at_k(ids[:5],  relevant, 5),  4),
        "recall@10":             round(recall_at_k(ids[:10], relevant, 10), 4),
        "precision@5":           round(precision_at_k(ids[:5],  relevant, 5),  4),
        "precision@10":          round(precision_at_k(ids[:10], relevant, 10), 4),
        "mrr@5":                 round(mrr_at_k(ids[:5],  relevant, 5),  4),
        "mrr@10":                round(mrr_at_k(ids[:10], relevant, 10), 4),
        "hit@5":                 round(hit_at_k(ids[:5],  relevant, 5),  4),
        "hit@10":                round(hit_at_k(ids[:10], relevant, 10), 4),
        "ndcg@5":                round(ndcg_at_k(ids[:5],  relevant, 5),  4),
        "ndcg@10":               round(ndcg_at_k(ids[:10], relevant, 10), 4),
        # RAGAS-style metrics
        "context_precision":     round(context_precision_at_k(top, relevant, k), 4),
        "context_recall":        _context_recall_from_gold(ids, meta, gold_context),
        # Other
        "redundancy_rate":        round(redundancy_ratio(texts), 4),
        "n_relevant":             len(relevant),
        "latency_s":              latency_s,
        "latency_breakdown_ms":   latency_breakdown or {},
        "top_ids":                top,
        "chunk_info":             chunk_info or [],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="A/B test: RAG techniques T1/T4/T8/T10 vs baseline")
    ap.add_argument("--benchmark",        default="../data/benchmark/context_retrieval_benchmark.json")
    ap.add_argument("--storage-noheader", default="storage_ab_noheader")
    ap.add_argument("--storage-header",   default="storage_ab_header")
    ap.add_argument("--out-dir",          default="../data/ab_techniques")
    ap.add_argument("--top-k",  type=int, default=K,   help="k for retrieval and metrics")
    ap.add_argument("--reset",  action="store_true",   help="Wipe storages and re-ingest")
    args = ap.parse_args()

    k = args.top_k
    benchmark_path = Path(args.benchmark).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_nh = Path(args.storage_noheader).resolve()
    storage_h  = Path(args.storage_header).resolve()

    if args.reset:
        import shutil
        for p in [storage_nh, storage_h]:
            if p.exists():
                shutil.rmtree(p)
                print(f"Removed {p}")
        cache_file = out_dir / "judge_cache.json"
        if cache_file.exists():
            cache_file.unlink()

    # ---- KnowledgeBases ----
    print("\n=== Setting up KnowledgeBases ===")
    settings_nh = _make_settings(storage_nh, headers_on=False)
    settings_h  = _make_settings(storage_h,  headers_on=True)
    if not settings_nh.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")

    kb_nh = KnowledgeBase(settings_nh)

    # ---- Ingest ----
    print("\n=== Ingesting PDFs (noheader) ===")
    _ingest_benchmark_pdfs(kb_nh, benchmark_path)
    if _header_storage_ready(settings_h, kb_nh):
        print("\n=== Using existing header index (no PDF re-parse) ===")
        kb_h = KnowledgeBase(settings_h)
        print(f"  header chunks: {len(kb_h.repo.all_chunks())}")
    else:
        print("\n=== Building header index from noheader chunks (no PDF re-parse) ===")
        t_header = time.perf_counter()
        kb_h = _build_header_kb_from_noheader(kb_nh, settings_h, storage_h)
        print(f"  header chunks: {len(kb_h.repo.all_chunks())}  ({time.perf_counter()-t_header:.1f}s)")

    # ---- Pre-build T10 parent index ----
    print("\n=== Pre-building Hierarchical parent index (T10) ===")
    t0 = time.perf_counter()
    kb_nh._rebuild_hierarchical_indexes(kb_nh.repo.all_chunks_with_embeddings())
    print(f"  parent nodes: {len(kb_nh.parent_nodes)}  ({time.perf_counter()-t0:.1f}s)")

    # ---- Dataset ----
    items = load_benchmark(benchmark_path)
    answerable   = [it for it in items if it.answerability == "answerable"]
    unanswerable = [it for it in items if it.answerability == "unanswerable"]
    print(f"\nDataset: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

    llm         = LLM()
    transformer = QueryTransformer(llm, max_variants=settings_nh.query_transform_max_variants)

    cache_file  = out_dir / "judge_cache.json"
    judge_cache: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    # ---- Cumulative token counters ----
    # Key naming convention:
    #   prod__*  → chi phí thật trong production (kỹ thuật gọi LLM lúc serve)
    #   eval__*  → chi phí chỉ trong eval (LLM-as-judge, không chạy khi serve)
    total_tokens: dict[str, dict] = {
        # production: T1 dùng LLM để sinh rewrite + step-back cho mỗi query
        "prod__T1_query_rewrite":    _zero_usage(),
        # eval-only: LLM judge xác định chunk nào liên quan cho baseline/T1/T8/T10
        "eval__judge_noheader_pool": _zero_usage(),
        # eval-only: LLM judge xác định chunk nào liên quan cho T4
        "eval__judge_header_pool":   _zero_usage(),
    }
    total_retrieval_tokens: dict[str, dict] = {
        cfg: _zero_retrieval_token_cost() for cfg in CONFIGS
    }

    # ---- Evaluate answerable items ----
    print("\n=== Evaluating answerable items ===")
    per_question: list[dict] = []

    for item in answerable:
        print(f"\n[{item.item_id}] ({item.difficulty}) {item.question[:70]}...")

        retrieved:          dict[str, list[int]]   = {}
        retrieved_info:     dict[str, list[dict]]  = {}
        latency_s:          dict[str, float]       = {}
        latency_breakdown:  dict[str, dict]        = {}
        retrieval_tokens:   dict[str, dict]        = {}
        q_tokens:           dict[str, dict]        = {}
        snap_nodes_q = _snap_nodes(llm)

        # -- Retrieve --
        def _timed(name: str, kb: KnowledgeBase, fn, *args):
            llm_before = _snap(llm)
            emb_before = _snap_embedder(kb)
            t = time.perf_counter()
            ids, info = fn(*args)
            latency_s[name] = round(time.perf_counter() - t, 3)
            retrieved_info[name] = info
            llm_usage = _delta(llm_before, _snap(llm))
            embedding_usage = _delta_token_usage(emb_before, _snap_embedder(kb))
            retrieval_tokens[name] = _retrieval_token_cost(llm_usage, embedding_usage)
            if name in total_retrieval_tokens:
                total_retrieval_tokens[name] = _add_retrieval_token_cost(total_retrieval_tokens[name], retrieval_tokens[name])
            return ids

        retrieved["baseline"]           = _timed("baseline",           kb_nh, retrieve_baseline, kb_nh, item.question, POOL_PER_CONFIG)
        latency_breakdown["baseline"]   = dict(kb_nh.last_retrieve_latency)
        retrieved["T4_ctxt_headers"]    = _timed("T4_ctxt_headers",    kb_h,  retrieve_t4,       kb_h,  item.question, POOL_PER_CONFIG)
        latency_breakdown["T4_ctxt_headers"] = dict(kb_h.last_retrieve_latency)
        retrieved["T8_rse"]             = _timed("T8_rse",             kb_nh, retrieve_t8,       kb_nh, item.question, POOL_PER_CONFIG)
        latency_breakdown["T8_rse"]     = dict(kb_nh.last_retrieve_latency)
        retrieved["T10_hierarchical"]   = _timed("T10_hierarchical",   kb_nh, retrieve_t10,      kb_nh, item.question, POOL_PER_CONFIG)
        latency_breakdown["T10_hierarchical"] = dict(kb_nh.last_retrieve_latency)

        # T1 — track token usage from the QueryTransformer LLM call
        snap_before_t1 = _snap(llm)
        retrieved["T1_query_transform"] = _timed("T1_query_transform", kb_nh, retrieve_t1, kb_nh, item.question, POOL_PER_CONFIG, transformer)
        latency_breakdown["T1_query_transform"] = dict(kb_nh.last_retrieve_latency)
        q_tokens["prod__T1_query_rewrite"] = _delta(snap_before_t1, _snap(llm))
        total_tokens["prod__T1_query_rewrite"] = _add_usage(
            total_tokens["prod__T1_query_rewrite"], q_tokens["prod__T1_query_rewrite"]
        )

        print(f"  latency(s): " + "  ".join(f"{c}={latency_s[c]:.2f}" for c in latency_s))
        print(f"  retrieval tokens: " + "  ".join(
            f"{c}={retrieval_tokens[c]['total_tokens']}" for c in retrieval_tokens
        ))

        # -- Judge: noheader pool (baseline / T1 / T8 / T10) --
        pool_nh = list(dict.fromkeys(
            retrieved["baseline"] + retrieved["T1_query_transform"]
            + retrieved["T8_rse"] + retrieved["T10_hierarchical"]
        ))
        meta_nh = kb_nh.repo.get_chunks(pool_nh)
        rel_nh, usage_nh = _judge_relevant(llm, item, pool_nh, meta_nh,
                                            judge_cache, f"{item.item_id}_nh")
        q_tokens["eval__judge_noheader_pool"] = usage_nh
        total_tokens["eval__judge_noheader_pool"] = _add_usage(
            total_tokens["eval__judge_noheader_pool"], usage_nh
        )
        cache_file.write_text(json.dumps(judge_cache, ensure_ascii=False, indent=2))

        # -- Judge: header pool (T4) --
        pool_h = retrieved["T4_ctxt_headers"]
        meta_h = kb_h.repo.get_chunks(pool_h)
        rel_h, usage_h = _judge_relevant(llm, item, pool_h, meta_h,
                                          judge_cache, f"{item.item_id}_h")
        q_tokens["eval__judge_header_pool"] = usage_h
        total_tokens["eval__judge_header_pool"] = _add_usage(
            total_tokens["eval__judge_header_pool"], usage_h
        )
        cache_file.write_text(json.dumps(judge_cache, ensure_ascii=False, indent=2))

        t1_tok  = q_tokens["prod__T1_query_rewrite"]["total_tokens"]
        jnh_tok = q_tokens["eval__judge_noheader_pool"]["total_tokens"]
        jh_tok  = q_tokens["eval__judge_header_pool"]["total_tokens"]
        print(f"  relevant: nh={len(rel_nh)}, h={len(rel_h)}  "
              f"tokens → T1={t1_tok}  judge_nh={jnh_tok}  judge_h={jh_tok}")

        # -- Compute metrics --
        # Fetch texts for redundancy calculation (top-k only, not full pool)
        all_top_ids = set()
        for cfg in CONFIGS:
            all_top_ids.update(retrieved.get(cfg, [])[:k])
        meta_all = {
            **kb_nh.repo.get_chunks([i for i in all_top_ids if i in set(retrieved.get("baseline", []))
                                     or i in set(retrieved.get("T1_query_transform", []))
                                     or i in set(retrieved.get("T8_rse", []))
                                     or i in set(retrieved.get("T10_hierarchical", []))]),
            **kb_h.repo.get_chunks([i for i in all_top_ids if i in set(retrieved.get("T4_ctxt_headers", []))]),
        }

        row: dict = {
            "item_id":        item.item_id,
            "doc_id":         item.doc_id,
            "difficulty":     item.difficulty,
            "question":       item.question,
            "retrieval_token_cost": {k_: dict(v_) for k_, v_ in retrieval_tokens.items()},
            "token_cost":     {k_: dict(v_) for k_, v_ in q_tokens.items()},
            "llm_cost_by_node": _delta_nodes(snap_nodes_q, _snap_nodes(llm)),
            "configs": {
                "baseline":           _compute_metrics(retrieved["baseline"],           rel_nh, meta_nh, k, latency_s["baseline"],           item.gold_context, latency_breakdown["baseline"],           retrieved_info.get("baseline")),
                "T1_query_transform": _compute_metrics(retrieved["T1_query_transform"], rel_nh, meta_nh, k, latency_s["T1_query_transform"], item.gold_context, latency_breakdown["T1_query_transform"], retrieved_info.get("T1_query_transform")),
                "T4_ctxt_headers":    _compute_metrics(retrieved["T4_ctxt_headers"],    rel_h,  meta_h,  k, latency_s["T4_ctxt_headers"],    item.gold_context, latency_breakdown["T4_ctxt_headers"],    retrieved_info.get("T4_ctxt_headers")),
                "T8_rse":             _compute_metrics(retrieved["T8_rse"],             rel_nh, meta_nh, k, latency_s["T8_rse"],             item.gold_context, latency_breakdown["T8_rse"],             retrieved_info.get("T8_rse")),
                "T10_hierarchical":   _compute_metrics(retrieved["T10_hierarchical"],   rel_nh, meta_nh, k, latency_s["T10_hierarchical"],   item.gold_context, latency_breakdown["T10_hierarchical"],   retrieved_info.get("T10_hierarchical")),
            },
        }
        per_question.append(row)

        for cfg, m in row["configs"].items():
            label = CONFIGS[cfg]["label"]
            print(
                f"    {label:38s} "
                f"R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}  P@5={m['precision@5']:.3f}  "
                f"MRR@5={m['mrr@5']:.3f}  nDCG@5={m['ndcg@5']:.3f}  "
                f"Redun={m['redundancy_rate']:.3f}  lat={m['latency_s']:.2f}s"
            )

    # ---- Unanswerable ----
    print("\n=== Evaluating unanswerable items ===")
    unanswerable_results: list[dict] = []
    for item in unanswerable:
        print(f"\n[{item.item_id}] {item.question[:70]}...")
        retrieval_un_tokens: dict[str, dict] = {}

        def _timed_unanswerable(name: str, kb: KnowledgeBase, fn, *args):
            llm_before = _snap(llm)
            emb_before = _snap_embedder(kb)
            ids, _ = fn(*args)
            llm_usage = _delta(llm_before, _snap(llm))
            embedding_usage = _delta_token_usage(emb_before, _snap_embedder(kb))
            retrieval_un_tokens[name] = _retrieval_token_cost(llm_usage, embedding_usage)
            if name in total_retrieval_tokens:
                total_retrieval_tokens[name] = _add_retrieval_token_cost(total_retrieval_tokens[name], retrieval_un_tokens[name])
            if name == "T1_query_transform":
                total_tokens["prod__T1_query_rewrite"] = _add_usage(
                    total_tokens["prod__T1_query_rewrite"], llm_usage
                )
            return ids

        retrieved_un = {
            "baseline":           _timed_unanswerable("baseline",           kb_nh, retrieve_baseline, kb_nh, item.question, k),
            "T1_query_transform": _timed_unanswerable("T1_query_transform", kb_nh, retrieve_t1,       kb_nh, item.question, k, transformer),
            "T4_ctxt_headers":    _timed_unanswerable("T4_ctxt_headers",    kb_h,  retrieve_t4,       kb_h,  item.question, k),
            "T8_rse":             _timed_unanswerable("T8_rse",             kb_nh, retrieve_t8,       kb_nh, item.question, k),
            "T10_hierarchical":   _timed_unanswerable("T10_hierarchical",   kb_nh, retrieve_t10,      kb_nh, item.question, k),
        }
        unanswerable_results.append({
            "item_id":  item.item_id,
            "question": item.question,
            "retrieved": {cfg: ids for cfg, ids in retrieved_un.items()},
            "retrieval_token_cost": {cfg: dict(usage) for cfg, usage in retrieval_un_tokens.items()},
        })
        for cfg, ids in retrieved_un.items():
            print(f"  {CONFIGS[cfg]['label']:38s} {len(ids)} chunks")

    # ---- Aggregate ----
    print("\n=== Aggregating ===")
    metric_keys = [
        "recall@5", "recall@10", "precision@5", "precision@10",
        "mrr@5", "mrr@10", "hit@5", "hit@10", "ndcg@5", "ndcg@10",
        "context_precision", "context_recall",
        "redundancy_rate",
    ]

    def _agg(rows: list[dict], cfg: str) -> dict:
        n = max(len(rows), 1)
        result = {}
        for m in metric_keys:
            vals = [r["configs"][cfg][m] for r in rows]
            result[m] = round(sum(vals) / n, 4)
        lats = sorted(r["configs"][cfg]["latency_s"] for r in rows)
        result["avg_latency_s"] = round(sum(lats) / n, 4)
        result["p95_latency_s"] = round(lats[max(0, int(len(lats) * 0.95) - 1)] if lats else 0.0, 4)
        steps = ["bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"]
        result["avg_latency_breakdown_ms"] = {
            step: round(sum(r["configs"][cfg].get("latency_breakdown_ms", {}).get(step, 0.0) for r in rows) / n, 1)
            for step in steps
        }
        token_rows = [r.get("retrieval_token_cost", {}).get(cfg, _zero_retrieval_token_cost()) for r in rows]
        result["retrieval_token_cost_total"] = {
            key: round(sum(t.get(key, 0) for t in token_rows), 1)
            for key in _zero_retrieval_token_cost()
        }
        result["retrieval_token_cost_avg"] = {
            key: round(result["retrieval_token_cost_total"][key] / n, 2)
            for key in _zero_retrieval_token_cost()
        }
        result["avg_retrieval_tokens"] = result["retrieval_token_cost_avg"]["total_tokens"]
        result["avg_retrieval_llm_tokens"] = result["retrieval_token_cost_avg"]["llm_total_tokens"]
        result["avg_retrieval_embedding_tokens"] = result["retrieval_token_cost_avg"]["embedding_tokens"]
        return result

    configs_order = list(CONFIGS.keys())
    overall      = {cfg: _agg(per_question, cfg) for cfg in configs_order}
    by_difficulty = {
        d: {cfg: _agg([r for r in per_question if r["difficulty"] == d], cfg)
            for cfg in configs_order}
        for d in sorted({r["difficulty"] for r in per_question})
    }
    by_document = {
        d: {cfg: _agg([r for r in per_question if r["doc_id"] == d], cfg)
            for cfg in configs_order}
        for d in sorted({r["doc_id"] for r in per_question})
    }

    # Aggregate total token cost — grouped by category
    prod_total = _zero_usage()
    eval_total = _zero_usage()
    for source, usage in total_tokens.items():
        if source.startswith("prod__"):
            prod_total = _add_usage(prod_total, usage)
        else:
            eval_total = _add_usage(eval_total, usage)

    grand_total = _add_usage(prod_total, eval_total)
    n_q = max(len(per_question), 1)
    n_retrieval_q = max(len(per_question) + len(unanswerable_results), 1)

    token_summary = {
        "by_source": {source: dict(usage) for source, usage in total_tokens.items()},
        "by_category": {
            "production":  dict(prod_total),
            "eval_only":   dict(eval_total),
        },
        "grand_total":      dict(grand_total),
        "per_question_avg": {
            k_: round(grand_total[k_] / n_q, 1) for k_ in grand_total
        },
        "source_descriptions": {
            "prod__T1_query_rewrite":    "T1: LLM sinh rewrite + step-back variant trước khi retrieve (production)",
            "eval__judge_noheader_pool": "Eval: LLM-as-judge xác định chunk liên quan cho baseline/T1/T8/T10",
            "eval__judge_header_pool":   "Eval: LLM-as-judge xác định chunk liên quan cho T4",
        },
    }

    retrieval_token_summary = {
        "by_config": {cfg: dict(usage) for cfg, usage in total_retrieval_tokens.items()},
        "per_question_avg": {
            cfg: {k_: round(v_ / n_retrieval_q, 2) for k_, v_ in usage.items()}
            for cfg, usage in total_retrieval_tokens.items()
        },
        "n_retrieval_questions": n_retrieval_q,
        "note": "Chỉ tính token phát sinh trong retrieval trên cả answerable và unanswerable queries: query transformation LLM và dense query embedding. Không tính LLM-as-judge.",
    }

    # Aggregate llm_cost_by_node across all questions
    total_node_cost: dict[str, dict] = {}
    for row in per_question:
        for node, cost in row.get("llm_cost_by_node", {}).items():
            if node not in total_node_cost:
                total_node_cost[node] = {"model": cost.get("model", ""), "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "total_ms": 0.0}
            total_node_cost[node]["model"] = cost.get("model", total_node_cost[node]["model"])
            total_node_cost[node]["prompt_tokens"] += cost.get("prompt_tokens", 0)
            total_node_cost[node]["completion_tokens"] += cost.get("completion_tokens", 0)
            total_node_cost[node]["calls"] += cost.get("calls", 0)
            total_node_cost[node]["total_ms"] = round(total_node_cost[node]["total_ms"] + cost.get("total_ms", 0.0), 1)

    full_results = {
        "k":               k,
        "n_answerable":    len(per_question),
        "n_unanswerable":  len(unanswerable_results),
        "overall":         overall,
        "by_difficulty":   by_difficulty,
        "by_document":     by_document,
        "retrieval_token_cost": retrieval_token_summary,
        "token_cost":      token_summary,
        "llm_cost_by_node": total_node_cost,
        "per_question":    per_question,
        "unanswerable":    unanswerable_results,
    }
    (out_dir / "results.json").write_text(
        json.dumps(full_results, ensure_ascii=False, indent=2)
    )

    _write_report(out_dir / "report.md", full_results, k)
    _write_flat_table(out_dir / "flat_table.md", full_results, k)
    print(f"\nReport: {out_dir / 'report.md'}")
    print(f"Flat table: {out_dir / 'flat_table.md'}")

    # ---- Print summary ----
    display_metrics = [
        "recall@5", "recall@10", "precision@5", "precision@10",
        "context_precision", "context_recall",
        "mrr@5", "mrr@10", "ndcg@5", "ndcg@10", "redundancy_rate", "avg_latency_s",
        "avg_retrieval_tokens",
    ]
    col_w = 10
    header = f"{'Configuration':<40}" + "".join(f"{m:>{col_w}}" for m in display_metrics)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for cfg in configs_order:
        label = CONFIGS[cfg]["label"]
        row_s = f"{label:<40}" + "".join(f"{overall[cfg][m]:>{col_w}.3f}" for m in display_metrics)
        print(row_s)
    print(f"{'='*len(header)}")

    print("\n--- Retrieval token cost by config (cumulative, excludes judge) ---")
    for cfg in configs_order:
        label = CONFIGS[cfg]["label"]
        u = retrieval_token_summary["by_config"][cfg]
        avg = retrieval_token_summary["per_question_avg"][cfg]
        print(
            f"  {label:38s} total={u['total_tokens']:>6} avg/q={avg['total_tokens']:>6.2f} "
            f"llm={u['llm_total_tokens']:>6} embedding={u['embedding_tokens']:>6} calls={u['calls']}"
        )

    descs = token_summary["source_descriptions"]
    print("\n--- Token cost (cumulative) ---")
    print(f"\n  [PRODUCTION — chi phí thật khi serve]")
    for src, u in token_summary["by_source"].items():
        if not src.startswith("prod__"):
            continue
        print(f"    {descs[src]}")
        print(f"      prompt={u['prompt_tokens']:>6}  completion={u['completion_tokens']:>6}  "
              f"total={u['total_tokens']:>7}  calls={u['calls']}")
    prod = token_summary["by_category"]["production"]
    print(f"    → production subtotal: total={prod['total_tokens']:>7}")

    print(f"\n  [EVAL-ONLY — không tốn khi serve, cached sau lần đầu]")
    for src, u in token_summary["by_source"].items():
        if not src.startswith("eval__"):
            continue
        print(f"    {descs[src]}")
        print(f"      prompt={u['prompt_tokens']:>6}  completion={u['completion_tokens']:>6}  "
              f"total={u['total_tokens']:>7}  calls={u['calls']}")
    ev = token_summary["by_category"]["eval_only"]
    print(f"    → eval subtotal: total={ev['total_tokens']:>7}")

    g = token_summary["grand_total"]
    avg = token_summary["per_question_avg"]
    print(f"\n  Grand total:   prompt={g['prompt_tokens']}  completion={g['completion_tokens']}  "
          f"total={g['total_tokens']}  calls={g['calls']}")
    print(f"  Avg/question:  total={avg['total_tokens']:.1f}  calls={avg['calls']:.1f}  "
          f"(production only: {prod['total_tokens']/n_q:.1f} tok/q)")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _delta_str(base: float, val: float) -> str:
    d = val - base
    if abs(d) < 0.0005:
        return "—"
    return f"{'+'if d>0 else ''}{d:.3f}"


def _write_report(path: Path, results: dict, k: int) -> None:
    metric_keys = [
        "recall@5", "recall@10", "precision@5", "precision@10",
        "mrr@5", "mrr@10", "hit@5", "hit@10", "ndcg@5", "ndcg@10",
        "context_precision", "context_recall",
        "redundancy_rate",
    ]
    configs_order = list(CONFIGS.keys())

    md: list[str] = [
        "# A/B Test: RAG Techniques (T1 / T4 / T8 / T10)",
        "",
        f"- Answerable: **{results['n_answerable']}**  |  Unanswerable: **{results['n_unanswerable']}**  |  k = {results['k']}",
        "",
    ]

    def _table(section: dict, baseline: dict) -> list[str]:
        delta_cols = [
            ("Δ recall@5",   "recall@5"),
            ("Δ recall@10",  "recall@10"),
            ("Δ ctx_recall",  "context_recall"),
            ("Δ ctx_prec",    "context_precision"),
        ]
        cols = metric_keys + ["avg_lat(s)", "p95_lat(s)", "avg_ret_tok", "avg_ret_llm_tok", "avg_ret_emb_tok"] + [d[0] for d in delta_cols]
        header = "| Config | " + " | ".join(cols) + " |"
        sep    = "|" + "---|" * (len(cols) + 1)
        rows   = [header, sep]
        for cfg in configs_order:
            r = section[cfg]
            b = baseline
            cells = [f"`{CONFIGS[cfg]['label']}`"]
            cells += [f"{r[m]:.3f}" for m in metric_keys]
            cells += [f"{r['avg_latency_s']:.3f}", f"{r['p95_latency_s']:.3f}"]
            cells += [
                f"{r.get('avg_retrieval_tokens', 0):.2f}",
                f"{r.get('avg_retrieval_llm_tokens', 0):.2f}",
                f"{r.get('avg_retrieval_embedding_tokens', 0):.2f}",
            ]
            cells += [_delta_str(b[mk], r[mk]) for _, mk in delta_cols]
            rows.append("| " + " | ".join(cells) + " |")
        return rows

    md += ["## Overall", ""]
    md += _table(results["overall"], results["overall"]["baseline"])
    md.append("")

    for diff in sorted(results.get("by_difficulty", {})):
        md += [f"## Difficulty: {diff}", ""]
        md += _table(results["by_difficulty"][diff], results["by_difficulty"][diff]["baseline"])
        md.append("")

    for doc_id in sorted(results.get("by_document", {})):
        md += [f"## Document: `{doc_id}`", ""]
        md += _table(results["by_document"][doc_id], results["by_document"][doc_id]["baseline"])
        md.append("")

    # Retrieval latency breakdown
    md += [
        "",
        "## Retrieval Latency Breakdown (avg ms per question)",
        "",
        "| Config | BM25 | Dense+Embed | Fusion | Rerank | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cfg in configs_order:
        lb = results["overall"][cfg].get("avg_latency_breakdown_ms", {})
        md.append(
            f"| `{CONFIGS[cfg]['label']}` | "
            f"{lb.get('bm25_ms', 0):.1f} | {lb.get('dense_ms', 0):.1f} | "
            f"{lb.get('fusion_ms', 0):.1f} | {lb.get('rerank_ms', 0):.1f} | "
            f"{lb.get('total_ms', 0):.1f} |"
        )
    md.append("> T1 tổng hợp latency qua tất cả query variants.")

    rtc = results.get("retrieval_token_cost", {})
    if rtc:
        md += [
            "",
            "## Retrieval Token Cost (cumulative, excludes judge)",
            "",
            rtc.get("note", ""),
            "",
            "| Config | Total | Avg/question | LLM total | Embedding total | Calls |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cfg in configs_order:
            u = rtc.get("by_config", {}).get(cfg, {})
            avg = rtc.get("per_question_avg", {}).get(cfg, {})
            md.append(
                f"| `{CONFIGS[cfg]['label']}` | {u.get('total_tokens', 0)} | "
                f"{avg.get('total_tokens', 0):.2f} | {u.get('llm_total_tokens', 0)} | "
                f"{u.get('embedding_tokens', 0)} | {u.get('calls', 0)} |"
            )

    # LLM cost by node
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
            md.append(
                f"| `{node}` | {cost['model']} | "
                f"{cost['prompt_tokens']} | {cost['completion_tokens']} | "
                f"{cost['calls']} | {cost['total_ms']:.1f} |"
            )

    # Token cost
    tc = results.get("token_cost", {})
    if tc:
        descs = tc.get("source_descriptions", {})
        md += ["## Token cost", "",
               "> `prod__*` = chi phí thật khi serve. "
               "`eval__*` = chỉ trong eval, cached sau lần đầu, không tốn khi serve.",
               "",
               "| Nguồn | Mục đích | Prompt | Completion | Total | Calls |",
               "|---|---|---|---|---|---|"]
        for src, u in tc.get("by_source", {}).items():
            desc = descs.get(src, src)
            md.append(f"| `{src}` | {desc} | {u['prompt_tokens']} | {u['completion_tokens']} | {u['total_tokens']} | {u['calls']} |")

        md += ["", "| Category | Total tokens | Calls |", "|---|---|---|"]
        for cat, u in tc.get("by_category", {}).items():
            md.append(f"| **{cat}** | {u['total_tokens']} | {u['calls']} |")
        gt = tc.get("grand_total", {})
        md.append(f"| **grand_total** | {gt.get('total_tokens',0)} | {gt.get('calls',0)} |")

        avg = tc.get("per_question_avg", {})
        prod_avg = round(tc.get("by_category", {}).get("production", {}).get("total_tokens", 0)
                         / max(results["n_answerable"], 1), 1)
        md.append(f"\n*Avg/question (all): {avg.get('total_tokens',0):.1f} tok — "
                  f"production only: {prod_avg:.1f} tok/q*")
        md.append("")

    # Per-question detail
    md += ["## Per-question detail", ""]
    short = {c: CONFIGS[c]["label"].split("–")[-1].strip()[:10] for c in configs_order}
    q_header = "| Item | Diff | " + " | ".join(f"{short[c]} R@{k}" for c in configs_order) + " | T1_tok | judge_tok |"
    q_sep    = "|" + "---|" * (len(configs_order) + 4)
    md += [q_header, q_sep]
    for row in results["per_question"]:
        cells = [row["item_id"], row["difficulty"][:1]]
        for cfg in configs_order:
            cells.append(f"{row['configs'][cfg]['recall@5']:.2f}")
        tc_row = row.get("token_cost", {})
        cells.append(str(tc_row.get("prod__T1_query_rewrite", {}).get("total_tokens", 0)))
        cells.append(str(tc_row.get("eval__judge_noheader_pool", {}).get("total_tokens", 0)
                     + tc_row.get("eval__judge_header_pool", {}).get("total_tokens", 0)))
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    if results.get("unanswerable"):
        md += ["## Unanswerable items", "",
               "| Item | " + " | ".join(configs_order) + " |",
               "|" + "---|" * (len(configs_order) + 1)]
        for row in results["unanswerable"]:
            cells = [row["item_id"]] + [str(len(row["retrieved"].get(c, []))) for c in configs_order]
            md.append("| " + " | ".join(cells) + " |")
        md.append("")

    path.write_text("\n".join(md), encoding="utf-8")


def _write_flat_table(path: Path, results: dict, k: int) -> None:
    """Bảng phẳng: mỗi dòng là 1 cặp (question × config), mỗi cột là 1 thông số."""
    metric_cols = [
        "recall@5", "recall@10",
        "precision@5", "precision@10",
        "mrr@5", "mrr@10",
        "hit@5", "hit@10",
        "ndcg@5", "ndcg@10",
        "context_precision", "context_recall",
        "redundancy_rate",
        "latency_s",
        "bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms",
        "retrieval_total_tokens", "retrieval_llm_tokens", "retrieval_embedding_tokens",
        "retrieval_calls", "retrieval_llm_calls", "retrieval_embedding_calls",
    ]

    header_cols = ["item_id", "doc_id", "difficulty", "config"] + metric_cols
    header = "| " + " | ".join(header_cols) + " |"
    sep    = "|" + "---|" * len(header_cols)

    rows = [header, sep]
    for row in results.get("per_question", []):
        for cfg, label in [(c, CONFIGS[c]["label"]) for c in CONFIGS]:
            m = row["configs"].get(cfg, {})
            lb = m.get("latency_breakdown_ms", {})
            rtc = row.get("retrieval_token_cost", {}).get(cfg, {})
            cells = [
                row["item_id"],
                row.get("doc_id", ""),
                row["difficulty"],
                label,
            ]
            for col in metric_cols:
                if col in ("bm25_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"):
                    cells.append(f"{lb.get(col, 0.0):.1f}")
                elif col == "retrieval_total_tokens":
                    cells.append(str(rtc.get("total_tokens", 0)))
                elif col == "retrieval_llm_tokens":
                    cells.append(str(rtc.get("llm_total_tokens", 0)))
                elif col == "retrieval_embedding_tokens":
                    cells.append(str(rtc.get("embedding_tokens", 0)))
                elif col == "retrieval_calls":
                    cells.append(str(rtc.get("calls", 0)))
                elif col == "retrieval_llm_calls":
                    cells.append(str(rtc.get("llm_calls", 0)))
                elif col == "retrieval_embedding_calls":
                    cells.append(str(rtc.get("embedding_calls", 0)))
                elif col in m:
                    val = m[col]
                    cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
                else:
                    cells.append("")
            rows.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
