"""KnowledgeBase: orchestrates ingestion, hybrid retrieval and persistence."""
import concurrent.futures
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import Settings, get_settings
from ..db.database import Database
from ..db.repo import Repo
from ..ingestion import loaders
from ..ingestion.chunker import Chunk, chunk_blocks
from ..retrieval.hybrid import FusedHit, reciprocal_rank_fusion
from ..retrieval.reranker import Reranker
from .bm25_index import BM25Index
from .embeddings import Embedder
from .vector_index import VectorIndex

logger = logging.getLogger("rag.flow")

# Bump when tokenization or index_text formatting changes, to force a rebuild of
# persisted indexes on next load (otherwise stale tokens silently degrade recall).
# Bumped to 6: 4-tier CCH (doc_summary + section_summary) changes embed text format.
_INDEX_VERSION = 6


def _index_text(
    title: Optional[str],
    section: Optional[str],
    text: str,
    summary: Optional[str] = None,
    section_summary: Optional[str] = None,
) -> str:
    """4-tier Contextual Chunk Header (CCH) — mirrors dsRAG auto_context.py.

    Embed text format (index-only, never shown to users in citations):

        Document context: the following excerpt is from a document titled '{title}'.
        {doc_summary}

        Section context: this excerpt is from the section titled '{section}'.
        {section_summary}

        {chunk_text}

    Why each tier matters (from dsRAG KITE benchmark, +28% accuracy vs no CCH):
      Tier 1 – title:           anchors every chunk to its document identity.
      Tier 2 – doc summary:     1 sentence describing the whole document; fixes
                                context-less chunks (pronouns, formulas without labels).
      Tier 3 – section title:   hierarchical path within the document.
      Tier 4 – section summary: 1 sentence describing the section; dramatically
                                improves recall for paraphrase queries that use
                                different words than the chunk itself.
    """
    parts: list[str] = []

    # Tier 1 + 2: document context
    if title:
        doc_ctx = f"Document context: the following excerpt is from a document titled '{title}'."
        if summary:
            doc_ctx += f" {summary.strip()}"
        parts.append(doc_ctx)

    # Tier 3 + 4: section context
    if section:
        sec_ctx = f"Section context: this excerpt is from the section titled '{section}'."
        if section_summary:
            sec_ctx += f" {section_summary.strip()}"
        parts.append(sec_ctx)

    # Tier 5: chunk text
    parts.append(text)
    return "\n\n".join(p for p in parts if p)


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    document_id: int
    doc_title: str
    doc_source: str
    page: Optional[int]
    section: Optional[str]
    rrf_score: float
    bm25_score: Optional[float]
    dense_score: Optional[float]
    rerank_score: Optional[float]
    score: float
    chunk_index: int = 0
    # RSE fields: populated when USE_RSE=true
    is_segment: bool = False
    segment_chunk_ids: list[int] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.db = Database(self.settings.db_path)
        self.repo = Repo(self.db)
        self.embedder = Embedder(cache_path=self.settings.storage_dir / "emb_cache.db")
        self.reranker = Reranker(self.settings.reranker_model)
        self._lock = threading.Lock()
        self.vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        self.bm25 = BM25Index()
        self._load_indexes()

    def pdf_path(self, doc_id: int) -> Path:
        return self.settings.storage_dir / "pdfs" / f"{doc_id}.pdf"

    # ---------------- persistence ----------------
    def _load_indexes(self) -> None:
        chunks = self.repo.all_chunks()
        db_ids = sorted(int(c["id"]) for c in chunks)
        if not db_ids:
            return

        meta = {}
        if self.settings.meta_path.exists():
            try:
                with open(self.settings.meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        if self.settings.vector_path.exists():
            try:
                self.vector = VectorIndex.load(
                    self.settings.vector_path, bit_width=self.settings.turbovec_bit_width
                )
            except Exception:
                self.vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        if self.settings.bm25_path.exists():
            try:
                self.bm25 = BM25Index.load(self.settings.bm25_path)
            except Exception:
                self.bm25 = BM25Index()

        # The persisted indexes are trusted ONLY if they cover exactly the DB chunk
        # set. Any drift (stale files, a previous partial ingest, a schema change)
        # silently drops chunks from retrieval, so we rebuild from the DB instead.
        indexed_ids = sorted(int(i) for i in meta.get("chunk_ids", []))
        in_sync = (
            indexed_ids == db_ids
            and self.bm25.count == len(db_ids)
            and self.vector._index is not None
            and meta.get("index_version") == _INDEX_VERSION
        )
        if in_sync:
            self.vector.set_count(len(db_ids))
        else:
            self.rebuild_indexes()

    def rebuild_indexes(self) -> None:
        """Rebuild BM25 + dense index from the DB so they always match stored chunks.
        Embeddings are read from the DB; any missing ones are (re)computed and saved.
        Uses embed_text for dense embedding when available (Reducto-optimized text)."""
        rows = self.repo.all_chunks_with_embeddings()
        if not rows:
            self.bm25 = BM25Index()
            self.vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
            self._persist()
            return

        ids = [int(r["id"]) for r in rows]
        index_texts = [
            _index_text(r.get("doc_title"), r.get("section"), r["text"],
                        summary=r.get("doc_summary"),
                        section_summary=r.get("section_summary"))
            for r in rows
        ]
        dense_texts = [
            _index_text(r.get("doc_title"), r.get("section"), r["embed_text"],
                        summary=r.get("doc_summary"),
                        section_summary=r.get("section_summary"))
            if r.get("embed_text")
            else _index_text(r.get("doc_title"), r.get("section"), r["text"],
                             summary=r.get("doc_summary"),
                             section_summary=r.get("section_summary"))
            for r in rows
        ]

        dim = self.embedder.dim
        vecs: list[Optional[np.ndarray]] = []
        missing: list[int] = []
        for i, r in enumerate(rows):
            blob = r.get("embedding")
            if blob:
                v = np.frombuffer(blob, dtype=np.float32)
                if dim and v.shape[0] != dim:
                    v = None
            else:
                v = None
            if v is None:
                missing.append(i)
            vecs.append(v)

        if missing:
            fresh = self.embedder.embed_documents([dense_texts[i] for i in missing])
            for j, i in enumerate(missing):
                vecs[i] = fresh[j]
            self.repo.set_embeddings([ids[i] for i in missing], fresh)

        matrix = np.vstack(vecs).astype(np.float32)

        self.bm25 = BM25Index()
        for cid, itext in zip(ids, index_texts):
            self.bm25.add(cid, itext)
        self.vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        self.vector.rebuild(matrix, ids)
        self._persist()

    def _persist(self) -> None:
        self.vector.save(self.settings.vector_path)
        self.bm25.save(self.settings.bm25_path)
        with open(self.settings.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embed_dim": self.embedder.dim,
                    "chunk_ids": list(self.bm25._chunk_ids),
                    "index_version": _INDEX_VERSION,
                },
                f,
            )

    # ---------------- ingestion ----------------
    def ingest_pdf(self, data: bytes, filename: str, doc_id: Optional[int] = None) -> dict[str, Any]:
        title, blocks = loaders.load_pdf(data, filename, cache_dir=self.settings.storage_dir)
        result = self._ingest(title, filename, "pdf", blocks, doc_id=doc_id)
        pdf_path = self.pdf_path(int(result["document_id"]))
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(data)
        return result

    def ingest_url(self, url: str) -> dict[str, Any]:
        title, blocks = loaders.load_url(url)
        return self._ingest(title, url, "url", blocks)

    def ingest_text(self, text: str, title: str = "Văn bản") -> dict[str, Any]:
        title, blocks = loaders.load_text(text, title)
        return self._ingest(title, title, "text", blocks)

    def _generate_doc_summary(self, chunks: list[Chunk]) -> Optional[str]:
        """Generate a 1-sentence document summary via LLM (Tier 2 of 4-tier CCH).

        Mirrors dsRAG get_document_summary():  takes up to doc_summary_chars of
        the FULL document (not just first N chunks) to capture documents where
        the main topic only appears mid-way through.  Returns None on failure.
        """
        if not self.settings.has_openai or not self.settings.enable_doc_summary:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            full_text = " ".join(ch.text for ch in chunks)
            sample = full_text[: self.settings.doc_summary_chars]
            resp = client.chat.completions.create(
                model=self.settings.doc_summary_model,
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
            summary = resp.choices[0].message.content or ""
            return summary.strip() or None
        except Exception:
            logger.exception("[CCH] Document summary generation failed, continuing without summary")
            return None

    def _generate_section_summaries(
        self, chunks: list[Chunk], title: str
    ) -> dict[str, str]:
        """Generate 1-sentence summaries for each unique section (Tier 4 of 4-tier CCH).

        Mirrors dsRAG get_section_summary(): 1 call per unique section, all calls
        run in parallel via ThreadPoolExecutor.  Returns {section: summary}.
        Sections with too little text (< 50 chars) get no summary.
        """
        if not self.settings.has_openai or not self.settings.enable_section_summary:
            return {}

        # Group chunk texts by section
        section_texts: dict[str, list[str]] = {}
        for ch in chunks:
            if not ch.section:
                continue
            section_texts.setdefault(ch.section, []).append(ch.text)

        def _summarize_one(section: str, texts: list[str]) -> tuple[str, Optional[str]]:
            combined = " ".join(texts)[: self.settings.section_summary_chars]
            if len(combined.strip()) < 50:
                return section, None
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=self.settings.openai_api_key,
                    base_url=self.settings.openai_base_url,
                )
                resp = client.chat.completions.create(
                    model=self.settings.doc_summary_model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Đây là nội dung phần '{section}' trong tài liệu '{title}'.\n"
                                "Phần này trình bày về điều gì?\n"
                                "Trả lời bằng 1 câu duy nhất dạng: 'Phần này trình bày về: X.' "
                                "Chỉ trả lời câu đó, không có gì thêm.\n\n"
                                f"{combined}"
                            ),
                        },
                    ],
                    max_tokens=100,
                    temperature=0.0,
                )
                result = (resp.choices[0].message.content or "").strip()
                return section, result or None
            except Exception:
                logger.warning("[CCH] Section summary failed for '%s'", section)
                return section, None

        summaries: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_summarize_one, sec, texts): sec
                for sec, texts in section_texts.items()
            }
            for fut in concurrent.futures.as_completed(futures):
                section, s = fut.result()
                if s:
                    summaries[section] = s

        return summaries

    def _ingest(self, title, source, source_type, blocks, doc_id: Optional[int] = None) -> dict[str, Any]:
        has_embed_text = any(getattr(b, "embed_text", None) for b in blocks)

        if has_embed_text:
            chunks = [
                Chunk(text=b.text, page=b.page, section=b.section, embed_text=b.embed_text)
                for b in blocks if b.text.strip()
            ]
        else:
            chunks = chunk_blocks(
                blocks,
                max_chars=self.settings.chunk_max_chars,
                overlap_chars=self.settings.chunk_overlap,
            )

        if not chunks:
            raise ValueError("Không tách được đoạn văn bản nào từ nguồn này.")

        # --- 4-tier CCH generation (runs outside the lock — pure API calls) ---
        # Tier 2: document summary (1 LLM call, full document text up to doc_summary_chars)
        doc_summary = self._generate_doc_summary(chunks)
        if doc_summary:
            logger.info("[CCH] Doc summary for '%s': %s", title, doc_summary[:80])

        # Tier 4: section summaries (1 LLM call per section, all parallel)
        section_summaries = self._generate_section_summaries(chunks, title)
        if section_summaries:
            logger.info("[CCH] Section summaries: %d sections", len(section_summaries))

        with self._lock:
            if doc_id is None:
                doc_id = self.repo.add_document(title, source, source_type)
            if doc_summary:
                self.repo.set_document_summary(doc_id, doc_summary)
            if section_summaries:
                self.repo.set_section_summaries(doc_id, section_summaries)

            chunk_ids: list[int] = []
            index_texts: list[str] = []
            embed_texts: list[Optional[str]] = []
            for i, ch in enumerate(chunks):
                cid = self.repo.add_chunk(
                    doc_id, i, ch.text, ch.page, ch.section,
                    embed_text=ch.embed_text,
                )
                chunk_ids.append(cid)
                sec_sum = section_summaries.get(ch.section) if ch.section else None
                itext = _index_text(title, ch.section, ch.text,
                                    summary=doc_summary, section_summary=sec_sum)
                index_texts.append(itext)
                embed_texts.append(ch.embed_text)
                self.bm25.add(cid, itext)
            self.repo.set_document_chunk_count(doc_id, len(chunks))

            dense_texts = [
                _index_text(title, ch.section,
                            et if et else ch.text,
                            summary=doc_summary,
                            section_summary=section_summaries.get(ch.section) if ch.section else None)
                for ch, et in zip(chunks, embed_texts)
            ]
            vectors = self.embedder.embed_documents(dense_texts)
            self.repo.set_embeddings(chunk_ids, vectors)
            self.vector.add(vectors, chunk_ids)
            self._persist()
        return {
            "document_id": doc_id,
            "title": title,
            "source": source,
            "source_type": source_type,
            "n_chunks": len(chunks),
            "doc_summary": doc_summary,
            "n_section_summaries": len(section_summaries),
        }

    def delete_document(self, doc_id: int) -> None:
        with self._lock:
            self.repo.delete_document(doc_id)
            self.pdf_path(doc_id).unlink(missing_ok=True)
            # Rebuild from the remaining DB chunks (embeddings are stored, so this
            # needs no API calls) to keep both indexes exactly in sync with the DB.
            self.rebuild_indexes()

    # ---------------- retrieval ----------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        started = time.perf_counter()
        s = self.settings
        top_k = top_k or s.final_top_k
        bm25_started = time.perf_counter()
        bm25_hits = self.bm25.search(query, s.bm25_top_k)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        dense_hits: list[tuple[int, float]] = []
        dense_ms = 0.0
        if self.vector.ready:
            dense_started = time.perf_counter()
            qv = self.embedder.embed_query(query)
            dense_hits = self.vector.search(qv, s.dense_top_k)
            dense_ms = (time.perf_counter() - dense_started) * 1000
        fusion_started = time.perf_counter()
        fused: list[FusedHit] = reciprocal_rank_fusion(bm25_hits, dense_hits, k=s.rrf_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        if not fused:
            logger.info(
                "RAG_FLOW retrieval node=retrieve duration_ms=%.1f bm25_ms=%.1f dense_ms=%.1f fusion_ms=%.1f "
                "query_chars=%s bm25_hits=%s dense_hits=%s fused_hits=0 rerank_enabled=%s returned=0",
                (time.perf_counter() - started) * 1000,
                bm25_ms,
                dense_ms,
                fusion_ms,
                len(query),
                len(bm25_hits),
                len(dense_hits),
                s.use_reranker,
            )
            return []

        candidates = fused[: s.rerank_top_n]
        meta_started = time.perf_counter()
        meta = self.repo.get_chunks([h.chunk_id for h in candidates])
        meta_ms = (time.perf_counter() - meta_started) * 1000

        rerank_map: dict[int, float] = {}
        rerank_ms = 0.0
        if s.use_reranker:
            pairs = [
                (h.chunk_id, meta[h.chunk_id]["text"])
                for h in candidates
                if h.chunk_id in meta
            ]
            rerank_started = time.perf_counter()
            reranked = self.reranker.rerank(query, pairs, top_k=len(pairs))
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            if reranked is not None:
                rerank_map = {cid: sc for cid, sc in reranked}

        def _final_score(h: FusedHit) -> float:
            return rerank_map.get(h.chunk_id, h.rrf_score)

        # Pass ALL reranked candidates to RSE (not pre-sliced to top_k).
        # RSE needs the full scored pool to correctly score "bridge" chunks relative
        # to retrieved ones; rse_overall_max_chunks controls the final context window.
        # Without RSE, fall back to a simple top_k slice.
        all_scored = sorted(candidates, key=_final_score, reverse=True)
        pre_rse: list[RetrievedChunk] = []
        for h in all_scored:
            m = meta.get(h.chunk_id)
            if not m:
                continue
            pre_rse.append(
                RetrievedChunk(
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
                    score=_final_score(h),
                    chunk_index=m.get("chunk_index", 0),
                )
            )

        # --- RSE: assemble contiguous segments from the full candidate pool ---
        rse_ms = 0.0
        if s.use_rse and pre_rse:
            rse_started = time.perf_counter()
            out = self._apply_rse(pre_rse)
            rse_ms = (time.perf_counter() - rse_started) * 1000
        else:
            out = pre_rse[:top_k]
        logger.info(
            "RAG_FLOW retrieval node=retrieve duration_ms=%.1f bm25_ms=%.1f dense_ms=%.1f fusion_ms=%.1f "
            "meta_ms=%.1f reranker_ms=%.1f rse_ms=%.1f query_chars=%s bm25_hits=%s dense_hits=%s "
            "fused_hits=%s candidates=%s reranked=%s rse=%s returned=%s",
            (time.perf_counter() - started) * 1000,
            bm25_ms,
            dense_ms,
            fusion_ms,
            meta_ms,
            rerank_ms,
            rse_ms,
            len(query),
            len(bm25_hits),
            len(dense_hits),
            len(fused),
            len(candidates),
            len(rerank_map),
            s.use_rse,
            len(out),
        )
        return out

    def _apply_rse(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Run RSE on reranked chunks and return segments as RetrievedChunk objects.

        Segments are contiguous windows of chunks from the same document.
        The returned list has the same interface as individual chunks so the
        rest of the pipeline (agent, citations) needs no changes.
        """
        from ..retrieval.rse import relevant_segment_extraction

        s = self.settings

        def _fetch_range(doc_id: int, start_idx: int, end_idx: int) -> list[dict]:
            return self.repo.get_chunks_by_doc_range(doc_id, start_idx, end_idx)

        segments = relevant_segment_extraction(
            scored_chunks=chunks,
            fetch_range_fn=_fetch_range,
            irrelevant_penalty=s.rse_irrelevant_penalty,
            max_segment_chunks=s.rse_max_segment_chunks,
            overall_max_chunks=s.rse_overall_max_chunks,
        )

        if not segments:
            return chunks  # Fall back to individual chunks if RSE finds nothing

        out: list[RetrievedChunk] = []
        for seg in segments:
            # Find the pre-RSE chunk for score metadata
            anchor_pre = next(
                (c for c in chunks if c.chunk_id == seg.anchor_chunk_id), None
            ) or chunks[0]
            out.append(
                RetrievedChunk(
                    chunk_id=seg.anchor_chunk_id,
                    text=seg.text,
                    document_id=seg.doc_id,
                    doc_title=seg.doc_title,
                    doc_source=seg.doc_source,
                    page=seg.page,
                    section=seg.section,
                    rrf_score=anchor_pre.rrf_score,
                    bm25_score=anchor_pre.bm25_score,
                    dense_score=anchor_pre.dense_score,
                    rerank_score=anchor_pre.rerank_score,
                    score=seg.value,
                    chunk_index=seg.start_idx,
                    is_segment=True,
                    segment_chunk_ids=seg.chunk_ids,
                )
            )
        return out

    def stats(self) -> dict[str, Any]:
        docs = self.repo.list_documents()
        return {
            "documents": len(docs),
            "chunks": sum(d["n_chunks"] for d in docs),
            "vector_ready": self.vector.ready,
            "reranker": self.settings.use_reranker,
            "rse": self.settings.use_rse,
        }
