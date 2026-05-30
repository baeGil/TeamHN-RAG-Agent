"""KnowledgeBase: orchestrates ingestion, hybrid retrieval and persistence."""
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

from ..config import Settings, get_settings
from ..db.database import Database
from ..db.repo import Repo
from ..ingestion import loaders
from ..ingestion.chunker import chunk_blocks
from ..retrieval.hybrid import FusedHit, reciprocal_rank_fusion
from ..retrieval.reranker import Reranker
from .bm25_index import BM25Index
from .embeddings import Embedder
from .vector_index import VectorIndex


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

    # ---------------- persistence ----------------
    def _load_indexes(self) -> None:
        chunks = self.repo.all_chunks()
        if self.settings.vector_path.exists():
            try:
                self.vector = VectorIndex.load(
                    self.settings.vector_path, bit_width=self.settings.turbovec_bit_width
                )
                self.vector.set_count(len(chunks))
            except Exception:
                self.vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        if self.settings.bm25_path.exists():
            try:
                self.bm25 = BM25Index.load(self.settings.bm25_path)
            except Exception:
                self.bm25 = BM25Index()
        # If indexes are empty but DB has chunks, the BM25 (tokens-only) can be
        # rebuilt cheaply; the vector index cannot (needs embeddings) so we leave it.
        if self.bm25.count == 0 and chunks:
            for c in chunks:
                self.bm25.add(int(c["id"]), c["text"])
            self.bm25.save(self.settings.bm25_path)

    def _persist(self) -> None:
        self.vector.save(self.settings.vector_path)
        self.bm25.save(self.settings.bm25_path)
        with open(self.settings.meta_path, "w", encoding="utf-8") as f:
            json.dump({"embed_dim": self.embedder.dim}, f)

    # ---------------- ingestion ----------------
    def ingest_pdf(self, data: bytes, filename: str) -> dict[str, Any]:
        title, blocks = loaders.load_pdf(data, filename)
        return self._ingest(title, filename, "pdf", blocks)

    def ingest_url(self, url: str) -> dict[str, Any]:
        title, blocks = loaders.load_url(url)
        return self._ingest(title, url, "url", blocks)

    def ingest_text(self, text: str, title: str = "Văn bản") -> dict[str, Any]:
        title, blocks = loaders.load_text(text, title)
        return self._ingest(title, title, "text", blocks)

    def _ingest(self, title, source, source_type, blocks) -> dict[str, Any]:
        chunks = chunk_blocks(blocks)
        if not chunks:
            raise ValueError("Không tách được đoạn văn bản nào từ nguồn này.")
        with self._lock:
            doc_id = self.repo.add_document(title, source, source_type)
            chunk_ids: list[int] = []
            for i, ch in enumerate(chunks):
                cid = self.repo.add_chunk(doc_id, i, ch.text, ch.page, ch.section)
                chunk_ids.append(cid)
                self.bm25.add(cid, ch.text)
            self.repo.set_document_chunk_count(doc_id, len(chunks))
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.vector.add(vectors, chunk_ids)
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
            chunk_ids = self.repo.delete_document(doc_id)
            self.bm25.remove(set(chunk_ids))
            self.vector.remove(chunk_ids)
            self._persist()

    # ---------------- retrieval ----------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        s = self.settings
        top_k = top_k or s.final_top_k
        bm25_hits = self.bm25.search(query, s.bm25_top_k)
        dense_hits: list[tuple[int, float]] = []
        if self.vector.ready:
            qv = self.embedder.embed_query(query)
            dense_hits = self.vector.search(qv, s.dense_top_k)
        fused: list[FusedHit] = reciprocal_rank_fusion(bm25_hits, dense_hits, k=s.rrf_k)
        if not fused:
            return []

        candidates = fused[: s.rerank_top_n]
        meta = self.repo.get_chunks([h.chunk_id for h in candidates])

        rerank_map: dict[int, float] = {}
        if s.use_reranker:
            pairs = [
                (h.chunk_id, meta[h.chunk_id]["text"])
                for h in candidates
                if h.chunk_id in meta
            ]
            reranked = self.reranker.rerank(query, pairs, top_k=len(pairs))
            if reranked is not None:
                rerank_map = {cid: sc for cid, sc in reranked}

        def _final_score(h: FusedHit) -> float:
            return rerank_map.get(h.chunk_id, h.rrf_score)

        ordered = sorted(candidates, key=_final_score, reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for h in ordered:
            m = meta.get(h.chunk_id)
            if not m:
                continue
            out.append(
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
        }
