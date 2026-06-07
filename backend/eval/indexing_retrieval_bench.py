"""Full pipeline benchmark: indexing + retrieval (no LLM generation).

Measures every latency phase separately:

  INDEXING:
    parsing → chunking → doc-summary (CCH) → embedding → index build

  RETRIEVAL (per query):
    BM25 → Dense embed → Dense search → RRF → Reranker → RSE

Also collects context quality signals (context size, section coverage,
keyword overlap with expected answers) WITHOUT calling the generation LLM.

Usage (from backend/):
    python -m eval.indexing_retrieval_bench --test-dir ../test
    python -m eval.indexing_retrieval_bench --test-dir ../test --parser mineru
    python -m eval.indexing_retrieval_bench --test-dir ../test --no-rse --no-summary
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ─── project path ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings
from app.db.database import Database
from app.db.repo import Repo
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import Embedder
from app.indexing.store import KnowledgeBase, RetrievedChunk, _index_text
from app.indexing.vector_index import VectorIndex
from app.ingestion import loaders
from app.ingestion.chunker import Chunk, chunk_blocks
from app.retrieval.hybrid import reciprocal_rank_fusion
from eval.dataset import load_dataset

# ─── token / cost constants ──────────────────────────────────────────────────
_CHARS_PER_TOKEN = 3.8          # rough approximation for Vietnamese text
_EMBED_PRICE_PER_1M = 0.02      # text-embedding-3-small USD/1M tokens
_GPT4O_MINI_IN = 0.15           # gpt-4o-mini input  USD/1M tokens
_GPT4O_MINI_OUT = 0.60          # gpt-4o-mini output USD/1M tokens


# ─── result dataclasses ──────────────────────────────────────────────────────

@dataclass
class IndexingResult:
    parser: str
    use_rse: bool
    enable_doc_summary: bool
    # counts
    n_pages: int = 0
    n_blocks: int = 0
    n_chunks: int = 0
    avg_chunk_chars: float = 0.0
    min_chunk_chars: int = 0
    max_chunk_chars: int = 0
    n_sections: int = 0
    # latencies (ms)
    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    summary_ms: float = 0.0
    embed_ms: float = 0.0
    index_build_ms: float = 0.0
    total_ingest_ms: float = 0.0
    # tokens / cost
    embed_tokens_est: int = 0
    embed_cost_usd: float = 0.0
    summary_input_tokens_est: int = 0
    summary_output_tokens_est: int = 0
    summary_cost_usd: float = 0.0
    section_summary_ms: float = 0.0
    section_summary_input_tokens_est: int = 0
    section_summary_output_tokens_est: int = 0
    section_summary_cost_usd: float = 0.0
    n_section_summaries: int = 0
    total_cost_usd: float = 0.0
    # output
    doc_summary: Optional[str] = None
    doc_title: str = ""


@dataclass
class QueryResult:
    qid: str
    question: str
    difficulty: str
    # retrieval latencies (ms)
    bm25_ms: float = 0.0
    embed_query_ms: float = 0.0
    dense_search_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    rse_ms: float = 0.0
    meta_fetch_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    # context quality
    n_returned: int = 0            # final chunks / segments
    total_context_chars: int = 0
    avg_segment_chars: float = 0.0
    n_sections_covered: int = 0
    n_docs_covered: int = 0
    n_rse_segments: int = 0        # how many are is_segment=True
    keyword_hit: bool = False      # context contains key terms from expected
    keyword_overlap_pct: float = 0.0
    # embed cost
    embed_query_tokens_est: int = 0
    embed_query_cost_usd: float = 0.0
    # top hits for inspection
    top_sections: list[str] = field(default_factory=list)
    top_snippets: list[str] = field(default_factory=list)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _keyword_overlap(context: str, expected: str) -> tuple[bool, float]:
    import re
    ctx_lower = context.lower()
    # extract meaningful words (≥3 chars) from expected answer
    words = [w for w in re.findall(r"\w+", expected.lower()) if len(w) >= 3]
    if not words:
        return False, 0.0
    hits = sum(1 for w in words if w in ctx_lower)
    pct = hits / len(words)
    return hits > 0, pct


def _make_settings(
    storage_dir: Path,
    parser: str,
    use_rse: bool,
    enable_doc_summary: bool,
    mineru_cmd: str = "",
) -> Settings:
    s = Settings()
    s.storage_dir = storage_dir
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    # parser
    s.reducto_parse = "off"
    s.mineru_parse = "on" if parser == "mineru" else "off"
    s.mineru_cmd = mineru_cmd
    # features
    s.use_rse = use_rse
    s.enable_doc_summary = enable_doc_summary
    # keep other defaults from .env
    return s


def _generate_doc_summary_timed(
    full_text: str, settings: Settings
) -> tuple[Optional[str], float, int, int]:
    """Returns (summary, elapsed_ms, input_tokens_est, output_tokens_est).
    Uses full document text (up to doc_summary_chars) — mirrors store.py logic."""
    if not settings.enable_doc_summary or not settings.has_openai:
        return None, 0.0, 0, 0
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        sample = full_text[: settings.doc_summary_chars]
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=settings.doc_summary_model or settings.llm_model_fast,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Tài liệu này là gì và nội dung chính của nó là gì?\n"
                        "Trả lời bằng 1 câu duy nhất, không quá dài, dạng: "
                        "'Tài liệu này trình bày về: X.' "
                        "Chỉ trả lời câu đó, không có gì thêm.\n\n"
                        f"{sample}"
                    ),
                },
            ],
            max_tokens=150,
            temperature=0.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        in_tok = getattr(u, "prompt_tokens", _est_tokens(sample)) if u else _est_tokens(sample)
        out_tok = getattr(u, "completion_tokens", _est_tokens(summary)) if u else _est_tokens(summary)
        return summary or None, elapsed_ms, in_tok, out_tok
    except Exception as exc:
        print(f"  [WARN] Doc summary failed: {exc}")
        return None, 0.0, 0, 0


def _generate_section_summaries_timed(
    chunks: list, title: str, settings: Settings
) -> tuple[dict, float, int, int]:
    """Returns (section_summaries_dict, elapsed_ms, total_in_tokens, total_out_tokens).
    Mirrors store.py _generate_section_summaries() — parallel LLM calls."""
    if not settings.enable_section_summary or not settings.has_openai:
        return {}, 0.0, 0, 0

    import concurrent.futures
    from openai import OpenAI

    section_texts: dict[str, list[str]] = {}
    for ch in chunks:
        if not ch.section:
            continue
        section_texts.setdefault(ch.section, []).append(ch.text)

    total_in, total_out = 0, 0
    summaries: dict[str, str] = {}

    def _one(section: str, texts: list[str]) -> tuple[str, Optional[str], int, int]:
        combined = " ".join(texts)[: settings.section_summary_chars]
        if len(combined.strip()) < 50:
            return section, None, 0, 0
        try:
            client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
            resp = client.chat.completions.create(
                model=settings.doc_summary_model or settings.llm_model_fast,
                messages=[{"role": "user", "content": (
                    f"Đây là nội dung phần '{section}' trong tài liệu '{title}'.\n"
                    "Phần này trình bày về điều gì?\n"
                    "Trả lời bằng 1 câu duy nhất dạng: 'Phần này trình bày về: X.' "
                    "Chỉ trả lời câu đó, không có gì thêm.\n\n"
                    f"{combined}"
                )}],
                max_tokens=100,
                temperature=0.0,
            )
            result = (resp.choices[0].message.content or "").strip()
            u = getattr(resp, "usage", None)
            it = getattr(u, "prompt_tokens", _est_tokens(combined)) if u else _est_tokens(combined)
            ot = getattr(u, "completion_tokens", _est_tokens(result)) if u else _est_tokens(result)
            return section, result or None, it, ot
        except Exception:
            return section, None, 0, 0

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_one, sec, txts): sec for sec, txts in section_texts.items()}
        for fut in concurrent.futures.as_completed(futs):
            sec, s, it, ot = fut.result()
            total_in += it
            total_out += ot
            if s:
                summaries[sec] = s
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return summaries, elapsed_ms, total_in, total_out


# ─── INDEXING BENCHMARK ──────────────────────────────────────────────────────

def run_indexing_bench(
    pdf_data: bytes,
    filename: str,
    settings: Settings,
    parser: str,
) -> tuple[IndexingResult, KnowledgeBase]:
    print(f"\n{'─'*60}")
    print(f"  INDEXING  |  parser={parser.upper()}  rse={settings.use_rse}  "
          f"doc_summary={settings.enable_doc_summary}")
    print(f"{'─'*60}")

    # Fresh storage
    if settings.storage_dir.exists():
        shutil.rmtree(settings.storage_dir)
    settings.storage_dir.mkdir(parents=True)

    res = IndexingResult(
        parser=parser,
        use_rse=settings.use_rse,
        enable_doc_summary=settings.enable_doc_summary,
    )

    t_total = time.perf_counter()

    # ── Phase 1: Parse ───────────────────────────────────────────────────────
    print(f"  [1/6] Parsing PDF ({len(pdf_data)//1024} KB) with {parser.upper()}...")
    t0 = time.perf_counter()
    title, blocks = loaders.load_pdf(pdf_data, filename, cache_dir=settings.storage_dir)
    res.parse_ms = (time.perf_counter() - t0) * 1000
    res.doc_title = title
    res.n_blocks = len(blocks)
    res.n_sections = len({b.section for b in blocks if b.section})

    # Estimate pages (blocks come back with page numbers)
    pages = {b.page for b in blocks if b.page}
    res.n_pages = len(pages) if pages else 0
    print(f"     → {res.n_blocks} blocks, {res.n_pages} pages, "
          f"{res.n_sections} sections  [{res.parse_ms:.0f} ms]")

    # ── Phase 2: Chunk ───────────────────────────────────────────────────────
    print(f"  [2/6] Chunking blocks...")
    has_embed_text = any(getattr(b, "embed_text", None) for b in blocks)
    t0 = time.perf_counter()
    if has_embed_text:
        chunks = [
            Chunk(text=b.text, page=b.page, section=b.section, embed_text=b.embed_text)
            for b in blocks if b.text.strip()
        ]
    else:
        chunks = chunk_blocks(
            blocks,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap,
        )
    res.chunk_ms = (time.perf_counter() - t0) * 1000
    res.n_chunks = len(chunks)
    chunk_sizes = [len(ch.text) for ch in chunks]
    res.avg_chunk_chars = statistics.mean(chunk_sizes) if chunk_sizes else 0
    res.min_chunk_chars = min(chunk_sizes) if chunk_sizes else 0
    res.max_chunk_chars = max(chunk_sizes) if chunk_sizes else 0
    print(f"     → {res.n_chunks} chunks  avg={res.avg_chunk_chars:.0f} chars  "
          f"min={res.min_chunk_chars}  max={res.max_chunk_chars}  [{res.chunk_ms:.0f} ms]")

    # ── Phase 3: Doc Summary (CCH tier 2) ───────────────────────────────────
    print(f"  [3/6] Doc summary (CCH tier 2)...")
    full_text = " ".join(ch.text for ch in chunks)
    summary, res.summary_ms, in_tok, out_tok = _generate_doc_summary_timed(
        full_text, settings
    )
    res.summary_input_tokens_est = in_tok
    res.summary_output_tokens_est = out_tok
    res.summary_cost_usd = (in_tok * _GPT4O_MINI_IN + out_tok * _GPT4O_MINI_OUT) / 1_000_000
    res.doc_summary = summary
    if summary:
        print(f"     → Summary ({len(summary)} chars): \"{summary[:100]}...\"  "
              f"[{res.summary_ms:.0f} ms]  cost=${res.summary_cost_usd:.5f}")
    else:
        print(f"     → Skipped (disabled or no API key)  [{res.summary_ms:.0f} ms]")

    # ── Phase 3b: Section Summaries (CCH tier 4) ────────────────────────────
    print(f"  [4/6] Section summaries (CCH tier 4)...")
    section_summaries, res.section_summary_ms, ss_in, ss_out = _generate_section_summaries_timed(
        chunks, title, settings
    )
    res.section_summary_input_tokens_est = ss_in
    res.section_summary_output_tokens_est = ss_out
    res.section_summary_cost_usd = (ss_in * _GPT4O_MINI_IN + ss_out * _GPT4O_MINI_OUT) / 1_000_000
    res.n_section_summaries = len(section_summaries)
    if section_summaries:
        print(f"     → {len(section_summaries)} section summaries  "
              f"[{res.section_summary_ms:.0f} ms]  cost=${res.section_summary_cost_usd:.5f}")
    else:
        print(f"     → Skipped (disabled or no API key)  [{res.section_summary_ms:.0f} ms]")

    # ── Phase 4: Embed ───────────────────────────────────────────────────────
    print(f"  [5/6] Embedding {res.n_chunks} chunks...")
    embedder = Embedder(cache_path=settings.storage_dir / "emb_cache.db")
    dense_texts = [
        _index_text(
            title, ch.section, ch.embed_text or ch.text,
            summary=summary,
            section_summary=section_summaries.get(ch.section) if ch.section else None,
        )
        for ch in chunks
    ]
    res.embed_tokens_est = sum(_est_tokens(t) for t in dense_texts)
    res.embed_cost_usd = res.embed_tokens_est * _EMBED_PRICE_PER_1M / 1_000_000

    t0 = time.perf_counter()
    vectors = embedder.embed_documents(dense_texts)
    res.embed_ms = (time.perf_counter() - t0) * 1000
    print(f"     → ~{res.embed_tokens_est:,} tokens  cost=${res.embed_cost_usd:.4f}  "
          f"[{res.embed_ms:.0f} ms]")

    # ── Phase 6: DB insert + index build ────────────────────────────────────
    print(f"  [6/6] Building BM25 + vector index...")
    db = Database(settings.db_path)
    repo = Repo(db)
    bm25 = BM25Index()
    vector = VectorIndex(bit_width=settings.turbovec_bit_width)

    t0 = time.perf_counter()
    doc_id = repo.add_document(title, filename, "pdf")
    if summary:
        repo.set_document_summary(doc_id, summary)
    if section_summaries:
        repo.set_section_summaries(doc_id, section_summaries)

    chunk_ids: list[int] = []
    index_texts: list[str] = []
    for i, ch in enumerate(chunks):
        cid = repo.add_chunk(doc_id, i, ch.text, ch.page, ch.section,
                             embed_text=ch.embed_text)
        chunk_ids.append(cid)
        sec_sum = section_summaries.get(ch.section) if ch.section else None
        itext = _index_text(title, ch.section, ch.text, summary=summary,
                            section_summary=sec_sum)
        index_texts.append(itext)
        bm25.add(cid, itext)

    repo.set_document_chunk_count(doc_id, len(chunks))
    repo.set_embeddings(chunk_ids, vectors)
    vector.rebuild(vectors, chunk_ids)

    # Persist
    bm25.save(settings.bm25_path)
    vector.save(settings.vector_path)
    import json as _json
    settings.meta_path.write_text(
        _json.dumps({
            "embed_dim": embedder.dim,
            "chunk_ids": list(bm25._chunk_ids),
            "index_version": 6,
        })
    )
    res.index_build_ms = (time.perf_counter() - t0) * 1000
    print(f"     → BM25 + TurboVec index ready  [{res.index_build_ms:.0f} ms]")

    res.total_ingest_ms = (time.perf_counter() - t_total) * 1000
    res.total_cost_usd = (
        res.embed_cost_usd + res.summary_cost_usd + res.section_summary_cost_usd
    )

    print(f"\n  ✓ Total ingest: {res.total_ingest_ms:.0f} ms  "
          f"total cost: ${res.total_cost_usd:.4f}")

    # Build KnowledgeBase from existing storage (loads the indexes)
    kb = KnowledgeBase(settings)
    return res, kb


# ─── RETRIEVAL BENCHMARK ─────────────────────────────────────────────────────

def run_retrieval_bench(
    kb: KnowledgeBase,
    dataset: list,
    use_rse: bool,
) -> list[QueryResult]:
    print(f"\n{'─'*60}")
    print(f"  RETRIEVAL  |  {len(dataset)} queries  rse={use_rse}")
    print(f"{'─'*60}")

    s = kb.settings
    results: list[QueryResult] = []

    for qa in dataset:
        qr = QueryResult(
            qid=qa.qid,
            question=qa.question,
            difficulty=qa.difficulty,
        )
        t_total = time.perf_counter()

        # ── BM25 search ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        bm25_hits = kb.bm25.search(qa.question, s.bm25_top_k)
        qr.bm25_ms = (time.perf_counter() - t0) * 1000

        # ── Dense embed query ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        qv = kb.embedder.embed_query(qa.question)
        qr.embed_query_ms = (time.perf_counter() - t0) * 1000
        qr.embed_query_tokens_est = _est_tokens(qa.question)
        qr.embed_query_cost_usd = qr.embed_query_tokens_est * _EMBED_PRICE_PER_1M / 1_000_000

        # ── Dense search ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        dense_hits = kb.vector.search(qv, s.dense_top_k) if kb.vector.ready else []
        qr.dense_search_ms = (time.perf_counter() - t0) * 1000

        # ── RRF fusion ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits, k=s.rrf_k)
        qr.rrf_ms = (time.perf_counter() - t0) * 1000

        # ── Meta fetch ────────────────────────────────────────────────────────
        candidates = fused[: s.rerank_top_n]
        t0 = time.perf_counter()
        meta = kb.repo.get_chunks([h.chunk_id for h in candidates])
        qr.meta_fetch_ms = (time.perf_counter() - t0) * 1000

        # ── Reranker ──────────────────────────────────────────────────────────
        from app.retrieval.hybrid import FusedHit
        rerank_map: dict[int, float] = {}
        t0 = time.perf_counter()
        if s.use_reranker:
            pairs = [
                (h.chunk_id, meta[h.chunk_id]["text"])
                for h in candidates if h.chunk_id in meta
            ]
            reranked = kb.reranker.rerank(qa.question, pairs, top_k=len(pairs))
            if reranked:
                rerank_map = {cid: sc for cid, sc in reranked}
        qr.rerank_ms = (time.perf_counter() - t0) * 1000

        # ── Build pre-RSE chunks (ALL reranked, not sliced to final_top_k) ──────
        def _score(h: FusedHit) -> float:
            return rerank_map.get(h.chunk_id, h.rrf_score)

        all_scored = sorted(candidates, key=_score, reverse=True)
        pre_rse: list[RetrievedChunk] = []
        for h in (all_scored if use_rse else all_scored[: s.final_top_k]):
            m = meta.get(h.chunk_id)
            if not m:
                continue
            pre_rse.append(RetrievedChunk(
                chunk_id=h.chunk_id,
                text=m["text"],
                document_id=m["document_id"],
                doc_title=m["doc_title"],
                doc_source=m["doc_source"],
                page=m["page"],
                section=m["section"],
                rrf_score=h.rrf_score,
                bm25_score=h.bm25_score,
                dense_score=h.dense_score,
                rerank_score=rerank_map.get(h.chunk_id),
                score=_score(h),
                chunk_index=m.get("chunk_index", 0),
            ))

        # ── RSE ───────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if use_rse and pre_rse:
            final_chunks = kb._apply_rse(pre_rse)
        else:
            final_chunks = pre_rse
        qr.rse_ms = (time.perf_counter() - t0) * 1000

        qr.total_retrieval_ms = (time.perf_counter() - t_total) * 1000

        # ── Context quality ───────────────────────────────────────────────────
        qr.n_returned = len(final_chunks)
        context_text = "\n\n".join(c.text for c in final_chunks)
        qr.total_context_chars = len(context_text)
        qr.avg_segment_chars = (
            qr.total_context_chars / qr.n_returned if qr.n_returned else 0
        )
        qr.n_sections_covered = len({c.section for c in final_chunks if c.section})
        qr.n_docs_covered = len({c.document_id for c in final_chunks})
        qr.n_rse_segments = sum(1 for c in final_chunks if c.is_segment)
        qr.keyword_hit, qr.keyword_overlap_pct = _keyword_overlap(context_text, qa.expected)
        qr.top_sections = list({c.section for c in final_chunks if c.section})[:3]
        qr.top_snippets = [c.text[:120].replace("\n", " ") for c in final_chunks[:3]]

        hit_icon = "✓" if qr.keyword_hit else "✗"
        rse_tag = f" [RSE:{qr.n_rse_segments}segs]" if use_rse else ""
        print(
            f"  [{qa.qid}] {qa.difficulty:4s} | "
            f"bm25={qr.bm25_ms:.0f}ms  dense={qr.embed_query_ms+qr.dense_search_ms:.0f}ms  "
            f"rrf={qr.rrf_ms:.0f}ms  rerank={qr.rerank_ms:.0f}ms  rse={qr.rse_ms:.0f}ms  "
            f"total={qr.total_retrieval_ms:.0f}ms | "
            f"ctx={qr.total_context_chars}c  n={qr.n_returned}  "
            f"kw={qr.keyword_overlap_pct:.0%}{rse_tag}  {hit_icon}"
        )
        results.append(qr)

    return results


# ─── AGGREGATE & REPORT ──────────────────────────────────────────────────────

def _agg(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
    vals_s = sorted(vals)
    n = len(vals_s)
    return {
        "mean": statistics.mean(vals_s),
        "min": vals_s[0],
        "max": vals_s[-1],
        "p50": vals_s[n // 2],
        "p95": vals_s[min(int(n * 0.95), n - 1)],
    }


def _aggregate_retrieval(results: list[QueryResult]) -> dict[str, Any]:
    def _col(attr):
        return [getattr(r, attr) for r in results]

    keyword_hit_rate = sum(1 for r in results if r.keyword_hit) / max(len(results), 1)
    return {
        "n_queries": len(results),
        "keyword_hit_rate": keyword_hit_rate,
        "latency_ms": {
            "bm25":         _agg(_col("bm25_ms")),
            "embed_query":  _agg(_col("embed_query_ms")),
            "dense_search": _agg(_col("dense_search_ms")),
            "rrf":          _agg(_col("rrf_ms")),
            "rerank":       _agg(_col("rerank_ms")),
            "rse":          _agg(_col("rse_ms")),
            "total":        _agg(_col("total_retrieval_ms")),
        },
        "context": {
            "avg_context_chars": statistics.mean(_col("total_context_chars")),
            "avg_n_returned":    statistics.mean(_col("n_returned")),
            "avg_sections":      statistics.mean(_col("n_sections_covered")),
            "avg_segment_chars": statistics.mean(_col("avg_segment_chars")),
            "avg_keyword_overlap_pct": statistics.mean(_col("keyword_overlap_pct")),
        },
        "embed_query_cost_usd_total": sum(_col("embed_query_cost_usd")),
    }


def _print_retrieval_summary(agg: dict, label: str = "") -> None:
    la = agg["latency_ms"]
    ctx = agg["context"]
    print(f"\n  ── Retrieval summary {label} ──")
    print(f"  Keyword hit rate:   {agg['keyword_hit_rate']:.0%} "
          f"  avg overlap: {ctx['avg_keyword_overlap_pct']:.0%}")
    print(f"  Context (per query): avg={ctx['avg_context_chars']:.0f} chars  "
          f"n={ctx['avg_n_returned']:.1f} chunks  "
          f"sections={ctx['avg_sections']:.1f}  "
          f"avg_seg={ctx['avg_segment_chars']:.0f} chars")
    print(f"  Latency (mean / p95 ms):")
    for phase, stats in la.items():
        print(f"    {phase:15s}  mean={stats['mean']:.1f}  p50={stats['p50']:.1f}  "
              f"p95={stats['p95']:.1f}  max={stats['max']:.1f}")
    print(f"  Embed query cost: ${agg['embed_query_cost_usd_total']:.5f} total")


def _write_report(
    out_path: Path,
    indexing: IndexingResult,
    retrieval_rse: list[QueryResult],
    retrieval_no_rse: list[QueryResult],
) -> None:
    agg_rse    = _aggregate_retrieval(retrieval_rse)
    agg_no_rse = _aggregate_retrieval(retrieval_no_rse)

    def _lat_table(la: dict) -> str:
        rows = ["| Phase | Mean ms | P50 ms | P95 ms | Max ms |",
                "|---|---|---|---|---|"]
        for phase, s in la.items():
            rows.append(f"| {phase} | {s['mean']:.1f} | {s['p50']:.1f} | "
                        f"{s['p95']:.1f} | {s['max']:.1f} |")
        return "\n".join(rows)

    def _ctx_table(ctx: dict) -> str:
        rows = ["| Metric | Value |", "|---|---|"]
        for k, v in ctx.items():
            rows.append(f"| {k} | {v:.2f} |")
        return "\n".join(rows)

    md: list[str] = [
        f"# Pipeline Benchmark Report",
        f"",
        f"**Parser:** {indexing.parser.upper()}  "
        f"**Doc summary (CCH):** {indexing.enable_doc_summary}  "
        f"**RSE:** {indexing.use_rse}",
        f"",
        f"## 1. Indexing",
        f"",
        f"| Phase | Time (ms) |",
        f"|---|---|",
        f"| PDF Parsing ({indexing.parser}) | {indexing.parse_ms:.0f} |",
        f"| Chunking | {indexing.chunk_ms:.0f} |",
        f"| Doc Summary LLM (CCH) | {indexing.summary_ms:.0f} |",
        f"| Embedding ({indexing.n_chunks} chunks) | {indexing.embed_ms:.0f} |",
        f"| Index build (BM25 + TurboVec) | {indexing.index_build_ms:.0f} |",
        f"| **Total ingest** | **{indexing.total_ingest_ms:.0f}** |",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Pages | {indexing.n_pages} |",
        f"| Blocks (parser output) | {indexing.n_blocks} |",
        f"| Chunks (after splitting) | {indexing.n_chunks} |",
        f"| Sections detected | {indexing.n_sections} |",
        f"| Avg chunk size (chars) | {indexing.avg_chunk_chars:.0f} |",
        f"| Min / Max chunk size | {indexing.min_chunk_chars} / {indexing.max_chunk_chars} |",
        f"",
        f"**Cost:**",
        f"- Embedding: ~{indexing.embed_tokens_est:,} tokens → **${indexing.embed_cost_usd:.4f}**",
        f"- Doc summary: {indexing.summary_input_tokens_est} in + "
        f"{indexing.summary_output_tokens_est} out → **${indexing.summary_cost_usd:.5f}**",
        f"- **Total indexing cost: ${indexing.total_cost_usd:.4f}**",
        f"",
    ]

    if indexing.doc_summary:
        md += [
            f"**Generated doc summary (CCH header):**",
            f"> {indexing.doc_summary}",
            f"",
        ]

    for label, agg, qresults in [
        ("RSE ON", agg_rse, retrieval_rse),
        ("RSE OFF (top-k baseline)", agg_no_rse, retrieval_no_rse),
    ]:
        md += [
            f"## 2. Retrieval – {label}",
            f"",
            f"**Keyword hit rate:** {agg['keyword_hit_rate']:.0%}  "
            f"avg overlap: {agg['context']['avg_keyword_overlap_pct']:.0%}",
            f"",
            f"### Latency",
            f"",
            _lat_table(agg["latency_ms"]),
            f"",
            f"### Context quality",
            f"",
            _ctx_table(agg["context"]),
            f"",
            f"### Per-query breakdown",
            f"",
            f"| Q | Diff | BM25 | Dense | Rerank | RSE | Total | Ctx chars | kw% | Hit |",
            f"|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in qresults:
            md.append(
                f"| {r.qid} | {r.difficulty} | {r.bm25_ms:.0f} | "
                f"{r.embed_query_ms+r.dense_search_ms:.0f} | {r.rerank_ms:.0f} | "
                f"{r.rse_ms:.0f} | {r.total_retrieval_ms:.0f} | "
                f"{r.total_context_chars} | {r.keyword_overlap_pct:.0%} | "
                f"{'✓' if r.keyword_hit else '✗'} |"
            )
        md.append("")

    # RSE vs no-RSE delta
    if retrieval_rse and retrieval_no_rse:
        delta_kw = agg_rse["keyword_hit_rate"] - agg_no_rse["keyword_hit_rate"]
        delta_ctx = agg_rse["context"]["avg_context_chars"] - agg_no_rse["context"]["avg_context_chars"]
        delta_lat = (agg_rse["latency_ms"]["total"]["mean"]
                     - agg_no_rse["latency_ms"]["total"]["mean"])
        md += [
            f"## 3. RSE Impact (delta)",
            f"",
            f"| Metric | No RSE | With RSE | Delta |",
            f"|---|---|---|---|",
            f"| Keyword hit rate | {agg_no_rse['keyword_hit_rate']:.0%} | "
            f"{agg_rse['keyword_hit_rate']:.0%} | {delta_kw:+.0%} |",
            f"| Avg context chars | {agg_no_rse['context']['avg_context_chars']:.0f} | "
            f"{agg_rse['context']['avg_context_chars']:.0f} | {delta_ctx:+.0f} |",
            f"| Avg kw overlap | {agg_no_rse['context']['avg_keyword_overlap_pct']:.0%} | "
            f"{agg_rse['context']['avg_keyword_overlap_pct']:.0%} | "
            f"{agg_rse['context']['avg_keyword_overlap_pct'] - agg_no_rse['context']['avg_keyword_overlap_pct']:+.0%} |",
            f"| Avg total latency ms | {agg_no_rse['latency_ms']['total']['mean']:.1f} | "
            f"{agg_rse['latency_ms']['total']['mean']:.1f} | {delta_lat:+.1f} |",
            f"",
        ]

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n  Report written: {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="../test")
    ap.add_argument("--out-dir",  default="../data/bench")
    ap.add_argument("--parser",   default="pymupdf",
                    help="Parser to use: pymupdf | mineru")
    ap.add_argument("--mineru-cmd", default="",
                    help="Path to mineru binary (auto-detect if empty)")
    ap.add_argument("--no-rse",     action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load settings (reads .env) ────────────────────────────────────────────
    base_settings = Settings()
    if not base_settings.has_openai:
        raise SystemExit("OPENAI_API_KEY chưa cấu hình trong backend/.env")

    # ── Load PDF ──────────────────────────────────────────────────────────────
    pdf_path = next(test_dir.glob("*.pdf"), None)
    if pdf_path is None:
        raise SystemExit(f"Không tìm thấy PDF trong {test_dir}")
    pdf_data = pdf_path.read_bytes()
    print(f"\n  PDF: {pdf_path.name}  ({len(pdf_data)//1024} KB)")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset(test_dir)
    print(f"  Dataset: {len(dataset)} questions "
          f"({sum(1 for q in dataset if q.difficulty=='easy')} easy, "
          f"{sum(1 for q in dataset if q.difficulty=='hard')} hard)")

    use_rse = not args.no_rse
    enable_summary = not args.no_summary

    # ── Indexing ──────────────────────────────────────────────────────────────
    storage_dir = out_dir / f"storage_{args.parser}"
    settings = _make_settings(
        storage_dir=storage_dir,
        parser=args.parser,
        use_rse=use_rse,
        enable_doc_summary=enable_summary,
        mineru_cmd=args.mineru_cmd,
    )

    indexing_result, kb = run_indexing_bench(
        pdf_data, pdf_path.name, settings, args.parser
    )

    # ── Retrieval: RSE ON ─────────────────────────────────────────────────────
    kb.settings.use_rse = True
    retrieval_rse = run_retrieval_bench(kb, dataset, use_rse=True)

    # ── Retrieval: RSE OFF (baseline) ─────────────────────────────────────────
    kb.settings.use_rse = False
    retrieval_no_rse = run_retrieval_bench(kb, dataset, use_rse=False)

    # ── Aggregates ────────────────────────────────────────────────────────────
    agg_rse    = _aggregate_retrieval(retrieval_rse)
    agg_no_rse = _aggregate_retrieval(retrieval_no_rse)
    _print_retrieval_summary(agg_rse,    label="[RSE ON]")
    _print_retrieval_summary(agg_no_rse, label="[RSE OFF / top-k baseline]")

    # ── Write outputs ─────────────────────────────────────────────────────────
    full_results = {
        "indexing": asdict(indexing_result),
        "retrieval_rse":    [asdict(r) for r in retrieval_rse],
        "retrieval_no_rse": [asdict(r) for r in retrieval_no_rse],
        "aggregate_rse":    agg_rse,
        "aggregate_no_rse": agg_no_rse,
    }
    json_path = out_dir / f"bench_{args.parser}.json"
    json_path.write_text(json.dumps(full_results, ensure_ascii=False, indent=2))
    print(f"\n  JSON saved: {json_path}")

    report_path = out_dir / f"bench_{args.parser}.md"
    _write_report(report_path, indexing_result, retrieval_rse, retrieval_no_rse)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FINAL SUMMARY  [{args.parser.upper()}]")
    print(f"{'═'*60}")
    print(f"  Indexing  total={indexing_result.total_ingest_ms:.0f}ms"
          f"  parse={indexing_result.parse_ms:.0f}ms"
          f"  chunk={indexing_result.chunk_ms:.0f}ms"
          f"  doc_sum={indexing_result.summary_ms:.0f}ms"
          f"  sec_sum={indexing_result.section_summary_ms:.0f}ms"
          f"  embed={indexing_result.embed_ms:.0f}ms"
          f"  idx={indexing_result.index_build_ms:.0f}ms")
    print(f"  Chunks: {indexing_result.n_chunks}  avg={indexing_result.avg_chunk_chars:.0f}c"
          f"  sections={indexing_result.n_sections}"
          f"  sec_summaries={indexing_result.n_section_summaries}"
          f"  cost=${indexing_result.total_cost_usd:.4f}")
    if indexing_result.doc_summary:
        print(f"  CCH doc summary: \"{indexing_result.doc_summary[:80]}...\"")
    if indexing_result.n_section_summaries:
        print(f"  CCH section summaries: {indexing_result.n_section_summaries} sections")
    for label, agg in [("RSE ON", agg_rse), ("RSE OFF", agg_no_rse)]:
        print(f"  Retrieval [{label}]"
              f"  hit={agg['keyword_hit_rate']:.0%}"
              f"  kw={agg['context']['avg_keyword_overlap_pct']:.0%}"
              f"  ctx={agg['context']['avg_context_chars']:.0f}c"
              f"  lat={agg['latency_ms']['total']['mean']:.0f}ms")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
