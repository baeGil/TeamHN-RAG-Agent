"""KnowledgeBase: orchestrates ingestion, hybrid retrieval and persistence."""
import json
import logging
import threading
import time
from dataclasses import dataclass
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
from ..retrieval.segments import expand_segments, extract_relevant_segments
from .bm25_index import BM25Index
from .contextual_headers import apply_contextual_header, build_header, signature as cch_signature
from .embeddings import Embedder
from .vector_index import VectorIndex

logger = logging.getLogger("rag.flow")

# Bump when tokenization or index_text formatting changes, to force a rebuild of
# persisted indexes on next load (otherwise stale tokens silently degrade recall).
_INDEX_VERSION = 5


def _index_text(
    settings: Settings,
    title: Optional[str],
    section: Optional[str],
    page: Optional[int],
    text: str,
) -> str:
    return apply_contextual_header(
        text,
        title=title,
        section=section,
        page=page,
        enabled=settings.enable_contextual_chunk_headers,
        include_page=settings.contextual_headers_include_page,
    )


@dataclass
class RetrievedChunk:
    chunk_id: int
    chunk_index: int
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
    contextual_header: Optional[str] = None
    rse_segment_start: Optional[int] = None
    rse_segment_end: Optional[int] = None
    rse_segment_score: Optional[float] = None
    rse_seed: bool = True
    hierarchical_parent_id: Optional[int] = None
    hierarchical_parent_score: Optional[float] = None
    hierarchical_boosted: bool = False


@dataclass
class _ScoredHit:
    chunk_id: int
    rrf_score: float
    bm25_score: Optional[float]
    dense_score: Optional[float]
    rerank_score: Optional[float]
    score: float
    hierarchical_parent_id: Optional[int] = None
    hierarchical_parent_score: Optional[float] = None
    hierarchical_boosted: bool = False


@dataclass
class _ParentNode:
    parent_id: int
    document_id: int
    doc_title: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_ids: list[int]
    text: str


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
        self.parent_vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        self.parent_bm25 = BM25Index()
        self.parent_nodes: dict[int, _ParentNode] = {}
        self.chunk_to_parent: dict[int, int] = {}
        self.last_retrieve_latency: dict[str, float] = {}
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
        index_signature = cch_signature(
            self.settings.enable_contextual_chunk_headers,
            self.settings.contextual_headers_include_page,
        )
        signature_matches = meta.get("index_signature") == index_signature
        in_sync = (
            indexed_ids == db_ids
            and self.bm25.count == len(db_ids)
            and self.vector._index is not None
            and meta.get("index_version") == _INDEX_VERSION
            and signature_matches
        )
        if in_sync:
            self.vector.set_count(len(db_ids))
            if self.settings.enable_hierarchical_indices:
                self._rebuild_hierarchical_indexes(chunks)
        else:
            self.rebuild_indexes(force_reembed=not signature_matches)

    def rebuild_indexes(self, *, force_reembed: bool = False) -> None:
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
            _index_text(self.settings, r.get("doc_title"), r.get("section"), r.get("page"), r["text"])
            for r in rows
        ]
        dense_texts = [
            _index_text(self.settings, r.get("doc_title"), r.get("section"), r.get("page"), r["embed_text"])
            if r.get("embed_text")
            else _index_text(self.settings, r.get("doc_title"), r.get("section"), r.get("page"), r["text"])
            for r in rows
        ]

        dim = self.embedder.dim
        vecs: list[Optional[np.ndarray]] = []
        missing: list[int] = []
        for i, r in enumerate(rows):
            blob = r.get("embedding")
            if blob and not force_reembed:
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
        if self.settings.enable_hierarchical_indices:
            self._rebuild_hierarchical_indexes(rows)
        else:
            self._clear_hierarchical_indexes()
        self._persist()

    def _clear_hierarchical_indexes(self) -> None:
        self.parent_vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        self.parent_bm25 = BM25Index()
        self.parent_nodes = {}
        self.chunk_to_parent = {}

    def _parent_key(self, row: dict[str, Any]) -> tuple[int, str, Any]:
        section = (row.get("section") or "").strip()
        if section:
            return int(row["document_id"]), "section", section.casefold()
        page = row.get("page")
        if page is not None:
            return int(row["document_id"]), "page", int(page)
        return int(row["document_id"]), "document", "all"

    def _rebuild_hierarchical_indexes(self, rows: list[dict[str, Any]]) -> None:
        self._clear_hierarchical_indexes()
        if not rows:
            return

        grouped: dict[tuple[int, str, Any], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(self._parent_key(row), []).append(row)

        parent_texts: list[str] = []
        parent_ids: list[int] = []
        for parent_id, group in enumerate(grouped.values(), start=1):
            group = sorted(group, key=lambda r: (int(r["document_id"]), int(r["chunk_index"])))
            title = group[0].get("doc_title") or ""
            section = group[0].get("section")
            pages = [int(r["page"]) for r in group if r.get("page") is not None]
            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None
            body_parts = [
                (r.get("embed_text") or r.get("text") or "").strip()
                for r in group
                if (r.get("embed_text") or r.get("text") or "").strip()
            ]
            body = "\n".join(body_parts)
            if len(body) > self.settings.hierarchical_parent_max_chars:
                body = body[: self.settings.hierarchical_parent_max_chars]
            page_label = (
                f"Trang {page_start}-{page_end}" if page_start and page_end and page_start != page_end
                else f"Trang {page_start}" if page_start
                else ""
            )
            heading = " | ".join([x for x in [title, section, page_label] if x])
            text = f"{heading}\n{body}" if heading else body
            chunk_ids = [int(r["id"]) for r in group]
            node = _ParentNode(
                parent_id=parent_id,
                document_id=int(group[0]["document_id"]),
                doc_title=title,
                section=section,
                page_start=page_start,
                page_end=page_end,
                chunk_ids=chunk_ids,
                text=text,
            )
            self.parent_nodes[parent_id] = node
            for cid in chunk_ids:
                self.chunk_to_parent[cid] = parent_id
            self.parent_bm25.add(parent_id, text)
            parent_ids.append(parent_id)
            parent_texts.append(text)

        vectors = self.embedder.embed_documents(parent_texts)
        self.parent_vector.rebuild(vectors, parent_ids)

    def _hierarchical_parent_hits(
        self,
        query: str,
        qv: Optional[np.ndarray],
    ) -> dict[int, FusedHit]:
        if not self.settings.enable_hierarchical_indices or not self.parent_nodes:
            return {}
        bm25_hits = self.parent_bm25.search(query, self.settings.hierarchical_parent_top_k * 4)
        dense_hits: list[tuple[int, float]] = []
        if qv is not None and self.parent_vector.ready:
            dense_hits = self.parent_vector.search(qv, self.settings.hierarchical_parent_top_k * 4)
        parents = reciprocal_rank_fusion(bm25_hits, dense_hits, k=self.settings.rrf_k)
        return {h.chunk_id: h for h in parents[: self.settings.hierarchical_parent_top_k]}

    def _apply_hierarchical_boost(
        self,
        ordered: list[_ScoredHit],
        parent_hits: dict[int, FusedHit],
        meta: dict[int, dict[str, Any]],
    ) -> list[_ScoredHit]:
        if not parent_hits:
            return ordered
        by_id = {h.chunk_id: h for h in ordered}
        boosted: dict[int, _ScoredHit] = {}
        max_base = max((h.score for h in ordered), default=0.0)
        for parent_id, parent_hit in parent_hits.items():
            node = self.parent_nodes.get(parent_id)
            if not node:
                continue
            parent_score = float(parent_hit.rrf_score)
            child_ids = self._parent_child_candidates(node, {h.chunk_id for h in ordered})
            for rank, cid in enumerate(child_ids, start=1):
                base = by_id.get(cid)
                boost = self.settings.hierarchical_parent_boost * parent_score / rank
                if base is None:
                    boosted[cid] = _ScoredHit(
                        chunk_id=cid,
                        rrf_score=0.0,
                        bm25_score=None,
                        dense_score=None,
                        rerank_score=None,
                        score=max_base * 0.50 + boost,
                        hierarchical_parent_id=parent_id,
                        hierarchical_parent_score=parent_score,
                        hierarchical_boosted=True,
                    )
                else:
                    base.score += boost
                    base.hierarchical_parent_id = parent_id
                    base.hierarchical_parent_score = parent_score
                    base.hierarchical_boosted = True
        if boosted:
            meta.update(self.repo.get_chunks(list(boosted)))
        merged = list({h.chunk_id: h for h in [*ordered, *boosted.values()]}.values())
        return sorted(merged, key=lambda h: h.score, reverse=True)

    def _parent_child_candidates(self, node: _ParentNode, seed_ids: set[int]) -> list[int]:
        window = max(1, int(self.settings.hierarchical_parent_chunk_window))
        positions = [i for i, cid in enumerate(node.chunk_ids) if cid in seed_ids]
        out: list[int] = []
        if positions:
            radius = max(0, window // 2)
            for pos in positions:
                start = max(0, pos - radius)
                end = min(len(node.chunk_ids), start + window)
                for cid in node.chunk_ids[start:end]:
                    if cid not in out:
                        out.append(cid)
                    if len(out) >= window:
                        return out
        for cid in node.chunk_ids:
            if cid not in out:
                out.append(cid)
            if len(out) >= window:
                break
        return out

    def _expand_hierarchical_candidates(
        self,
        candidates: list[FusedHit],
        parent_hits: dict[int, FusedHit],
        meta: dict[int, dict[str, Any]],
    ) -> list[FusedHit]:
        if not parent_hits:
            return candidates
        by_id = {h.chunk_id: h for h in candidates}
        seed_ids = set(by_id)
        for parent_id, parent_hit in parent_hits.items():
            node = self.parent_nodes.get(parent_id)
            if not node:
                continue
            for rank, cid in enumerate(self._parent_child_candidates(node, seed_ids), start=1):
                if cid in by_id:
                    continue
                by_id[cid] = FusedHit(
                    chunk_id=cid,
                    rrf_score=float(parent_hit.rrf_score) * self.settings.hierarchical_parent_boost / rank,
                    sources=["hierarchical_parent"],
                )
        extra_ids = [cid for cid in by_id if cid not in meta]
        if extra_ids:
            meta.update(self.repo.get_chunks(extra_ids))
        return sorted(by_id.values(), key=lambda h: h.rrf_score, reverse=True)

    def _persist(self) -> None:
        self.vector.save(self.settings.vector_path)
        self.bm25.save(self.settings.bm25_path)
        with open(self.settings.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embed_dim": self.embedder.dim,
                    "chunk_ids": list(self.bm25._chunk_ids),
                    "index_version": _INDEX_VERSION,
                    "index_signature": cch_signature(
                        self.settings.enable_contextual_chunk_headers,
                        self.settings.contextual_headers_include_page,
                    ),
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

        with self._lock:
            if doc_id is None:
                doc_id = self.repo.add_document(title, source, source_type)
            chunk_ids: list[int] = []
            index_texts: list[str] = []
            embed_texts: list[Optional[str]] = []
            for i, ch in enumerate(chunks):
                cid = self.repo.add_chunk(
                    doc_id, i, ch.text, ch.page, ch.section,
                    embed_text=ch.embed_text,
                )
                chunk_ids.append(cid)
                itext = _index_text(self.settings, title, ch.section, ch.page, ch.text)
                index_texts.append(itext)
                embed_texts.append(ch.embed_text)
                self.bm25.add(cid, itext)
            self.repo.set_document_chunk_count(doc_id, len(chunks))

            dense_texts = [
                _index_text(self.settings, title, ch.section, ch.page, et) if et else index_texts[i]
                for i, (ch, et) in enumerate(zip(chunks, embed_texts))
            ]
            vectors = self.embedder.embed_documents(dense_texts)
            self.repo.set_embeddings(chunk_ids, vectors)
            self.vector.add(vectors, chunk_ids)
            if self.settings.enable_hierarchical_indices:
                self._rebuild_hierarchical_indexes(self.repo.all_chunks_with_embeddings())
            self._persist()
        return {
            "document_id": doc_id,
            "title": title,
            "source": source,
            "source_type": source_type,
            "n_chunks": len(chunks),
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
        qv: Optional[np.ndarray] = None
        if self.vector.ready:
            dense_started = time.perf_counter()
            qv = self.embedder.embed_query(query)
            dense_hits = self.vector.search(qv, s.dense_top_k)
            dense_ms = (time.perf_counter() - dense_started) * 1000
        parent_hits = self._hierarchical_parent_hits(query, qv)
        fusion_started = time.perf_counter()
        fused: list[FusedHit] = reciprocal_rank_fusion(bm25_hits, dense_hits, k=s.rrf_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        if not fused:
            self.last_retrieve_latency = {
                "bm25_ms": round(bm25_ms, 1),
                "dense_ms": round(dense_ms, 1),
                "fusion_ms": round(fusion_ms, 1),
                "rerank_ms": 0.0,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            }
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
        candidates = self._expand_hierarchical_candidates(candidates, parent_hits, meta)
        meta_ms = (time.perf_counter() - meta_started) * 1000

        rerank_map: dict[int, float] = {}
        rerank_ms = 0.0
        if s.use_reranker:
            pairs = [
                (
                    h.chunk_id,
                    _index_text(
                        self.settings,
                        meta[h.chunk_id].get("doc_title"),
                        meta[h.chunk_id].get("section"),
                        meta[h.chunk_id].get("page"),
                        meta[h.chunk_id].get("embed_text") or meta[h.chunk_id]["text"],
                    ),
                )
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

        ordered_candidates = sorted(candidates, key=_final_score, reverse=True)
        ordered = [
            _ScoredHit(
                chunk_id=h.chunk_id,
                rrf_score=h.rrf_score,
                bm25_score=h.bm25_score,
                dense_score=h.dense_score,
                rerank_score=rerank_map.get(h.chunk_id),
                score=_final_score(h),
                hierarchical_parent_id=self.chunk_to_parent.get(h.chunk_id)
                if self.chunk_to_parent.get(h.chunk_id) in parent_hits
                else None,
                hierarchical_parent_score=parent_hits[self.chunk_to_parent[h.chunk_id]].rrf_score
                if self.chunk_to_parent.get(h.chunk_id) in parent_hits
                else None,
                hierarchical_boosted=self.chunk_to_parent.get(h.chunk_id) in parent_hits,
            )
            for h in ordered_candidates
        ]
        ordered = self._apply_hierarchical_boost(ordered, parent_hits, meta)[:top_k]
        if self.settings.enable_relevant_segment_extraction:
            segments = extract_relevant_segments(
                ordered,
                meta,
                penalty=self.settings.rse_irrelevant_chunk_penalty,
                max_segment_chunks=self.settings.rse_max_segment_chunks,
            )
            expanded_rows = expand_segments(
                segments,
                self.repo.get_chunks_by_doc_index_range,
                max_context_chunks=max(top_k, self.settings.rse_max_context_chunks),
            )
            if expanded_rows:
                scored_by_id = {h.chunk_id: h for h in ordered}
                expanded: list[_ScoredHit] = []
                for rank, row in enumerate(expanded_rows, start=1):
                    cid = int(row["id"])
                    seed = scored_by_id.get(cid)
                    segment_score = float(row.get("rse_segment_score", 0.0) or 0.0)
                    bridge_score = segment_score / (rank + 1)
                    expanded.append(
                        _ScoredHit(
                            chunk_id=cid,
                            rrf_score=seed.rrf_score if seed else 0.0,
                            bm25_score=seed.bm25_score if seed else None,
                            dense_score=seed.dense_score if seed else None,
                            rerank_score=seed.rerank_score if seed else None,
                            score=seed.score if seed else bridge_score,
                            hierarchical_parent_id=seed.hierarchical_parent_id if seed else self.chunk_to_parent.get(cid),
                            hierarchical_parent_score=seed.hierarchical_parent_score if seed else None,
                            hierarchical_boosted=seed.hierarchical_boosted if seed else False,
                        )
                    )
                    meta[cid] = row
                ordered = expanded
        out: list[RetrievedChunk] = []
        for h in ordered:
            m = meta.get(h.chunk_id)
            if not m:
                continue
            header = build_header(m.get("doc_title"), m.get("section"), m.get("page"))
            out.append(
                RetrievedChunk(
                    chunk_id=h.chunk_id,
                    chunk_index=int(m["chunk_index"]),
                    text=m["text"],
                    document_id=m["document_id"],
                    doc_title=m["doc_title"],
                    doc_source=m["doc_source"],
                    page=m["page"],
                    section=m["section"],
                    rrf_score=h.rrf_score,
                    bm25_score=h.bm25_score,
                    dense_score=h.dense_score,
                    rerank_score=h.rerank_score,
                    score=h.score,
                    contextual_header=(
                        header.format(include_page=self.settings.contextual_headers_include_page)
                        if self.settings.enable_contextual_chunk_headers and not header.is_empty
                        else None
                    ),
                    rse_segment_start=m.get("rse_segment_start"),
                    rse_segment_end=m.get("rse_segment_end"),
                    rse_segment_score=m.get("rse_segment_score"),
                    rse_seed=bool(m.get("rse_seed", True)),
                    hierarchical_parent_id=h.hierarchical_parent_id,
                    hierarchical_parent_score=h.hierarchical_parent_score,
                    hierarchical_boosted=h.hierarchical_boosted,
                )
            )
        total_ms = (time.perf_counter() - started) * 1000
        self.last_retrieve_latency = {
            "bm25_ms": round(bm25_ms, 1),
            "dense_ms": round(dense_ms, 1),
            "fusion_ms": round(fusion_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "total_ms": round(total_ms, 1),
        }
        logger.info(
            "RAG_FLOW retrieval node=retrieve duration_ms=%.1f bm25_ms=%.1f dense_ms=%.1f fusion_ms=%.1f "
            "meta_ms=%.1f reranker_ms=%.1f query_chars=%s bm25_hits=%s dense_hits=%s fused_hits=%s "
            "parent_hits=%s candidates=%s reranked=%s returned=%s",
            total_ms,
            bm25_ms,
            dense_ms,
            fusion_ms,
            meta_ms,
            rerank_ms,
            len(query),
            len(bm25_hits),
            len(dense_hits),
            len(fused),
            len(parent_hits),
            len(candidates),
            len(rerank_map),
            len(out),
        )
        return out

    def stats(self) -> dict[str, Any]:
        docs = self.repo.list_documents()
        return {
            "documents": len(docs),
            "chunks": sum(d["n_chunks"] for d in docs),
            "vector_ready": self.vector.ready,
            "reranker": self.settings.use_reranker,
            "contextual_chunk_headers": self.settings.enable_contextual_chunk_headers,
            "relevant_segment_extraction": self.settings.enable_relevant_segment_extraction,
            "hierarchical_indices": self.settings.enable_hierarchical_indices,
            "hierarchical_parent_nodes": len(self.parent_nodes),
        }
