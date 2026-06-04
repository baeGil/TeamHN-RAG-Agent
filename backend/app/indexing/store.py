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
from .bm25_index import BM25Index
from .embeddings import Embedder
from .vector_index import VectorIndex

logger = logging.getLogger("rag.flow")

# Bump when tokenization or index_text formatting changes, to force a rebuild of
# persisted indexes on next load (otherwise stale tokens silently degrade recall).
_INDEX_VERSION = 4


def _index_text(title: Optional[str], section: Optional[str], text: str) -> str:
    """Contextual chunk header: prepend doc title + section path before the chunk
    so context-poor chunks still match queries (BM25 + dense). Display/citation
    text stays clean (this header is index-only)."""
    head = " › ".join(p for p in (title, section) if p)
    return f"{head}\n{text}" if head else text


def merge_overlapping_texts(text1: str, text2: str, max_overlap: int = 500) -> str:
    """Merge text1 and text2 smoothly by identifying the longest common overlap."""
    max_len = min(len(text1), len(text2), max_overlap)
    for l in range(max_len, 0, -1):
        if text1[-l:] == text2[:l]:
            return text1 + text2[l:]
    return text1 + "\n\n" + text2


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

    def _generate_hype_questions(self, text: str, num_q: Optional[int] = None) -> list[str]:
        if not self.settings.has_openai:
            return []
        num_q = num_q or self.settings.hype_num_questions
        from ..agent.llm import LLM
        llm = LLM()
        prompt = [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia về RAG. Hãy tạo ra các câu hỏi giả định bằng tiếng Việt "
                    "cho đoạn văn bản được cung cấp. Phản hồi của bạn phải là một danh sách JSON "
                    "hợp lệ chứa các chuỗi câu hỏi ngắn, ví dụ: [\"Câu hỏi 1?\", \"Câu hỏi 2?\"]."
                ),
            },
            {
                "role": "user",
                "content": f"Văn bản:\n---\n{text}\n---\nTạo đúng {num_q} câu hỏi giả định bằng tiếng Việt dưới dạng JSON list.",
            },
        ]
        try:
            res = llm.chat_json(prompt, fast=True)
            if isinstance(res, dict):
                for k in ["questions", "queries", "list"]:
                    if k in res and isinstance(res[k], list):
                        return [str(q).strip() for q in res[k] if q]
                for val in res.values():
                    if isinstance(val, list):
                        return [str(q).strip() for q in val if q]
            elif isinstance(res, list):
                return [str(q).strip() for q in res if q]
        except Exception as e:
            logger.warning("Error generating HyPE questions: %s", e)
        return []

    def embed_query(self, query: str) -> np.ndarray:
        s = self.settings
        if s.use_hyde:
            from ..agent.llm import LLM
            llm = LLM()
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là một chuyên gia. Hãy viết một câu trả lời giả định ngắn gọn "
                        "(2-3 câu) bằng tiếng Việt cho câu hỏi của người dùng. "
                        "Không thêm lời mở đầu hay kết luận."
                    ),
                },
                {"role": "user", "content": f"Câu hỏi: {query}"},
            ]
            try:
                hyde_answer = llm.chat(prompt, fast=True)
                return self.embedder.embed_query(hyde_answer)
            except Exception as e:
                logger.warning("Error generating HyDE answer: %s", e)
        return self.embedder.embed_query(query)

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
            _index_text(r.get("doc_title"), r.get("section"), r["text"]) for r in rows
        ]
        dense_texts = [
            _index_text(r.get("doc_title"), r.get("section"), r["embed_text"])
            if r.get("embed_text")
            else _index_text(r.get("doc_title"), r.get("section"), r["text"])
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

        # Handle HyPe questions
        hype_texts = []
        hype_ids = []

        if self.settings.use_hype:
            for r in rows:
                cid = int(r["id"])
                hqs = []
                if r.get("hype_questions"):
                    try:
                        hqs = json.loads(r["hype_questions"])
                    except Exception:
                        hqs = []
                if not hqs:
                    hqs = self._generate_hype_questions(r["text"])
                    if hqs:
                        self.repo.set_hype_questions(cid, hqs)
                for q in hqs:
                    hype_texts.append(q)
                    hype_ids.append(cid)

        if hype_texts:
            hype_vecs = self.embedder.embed_documents(hype_texts)
            all_vecs = list(vecs) + [hype_vecs[i] for i in range(len(hype_texts))]
            all_ids = [cid * 10 for cid in ids]
            hype_idx_map = {}
            for cid in hype_ids:
                idx = hype_idx_map.get(cid, 1)
                hype_idx_map[cid] = idx + 1
                all_ids.append(cid * 10 + idx)
            matrix = np.vstack(all_vecs).astype(np.float32)
        else:
            matrix = np.vstack(vecs).astype(np.float32)
            all_ids = [cid * 10 for cid in ids]

        self.bm25 = BM25Index()
        for cid, itext in zip(ids, index_texts):
            self.bm25.add(cid, itext)
        self.vector = VectorIndex(bit_width=self.settings.turbovec_bit_width)
        self.vector.rebuild(matrix, all_ids)
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
                itext = _index_text(title, ch.section, ch.text)
                index_texts.append(itext)
                embed_texts.append(ch.embed_text)
                self.bm25.add(cid, itext)
            self.repo.set_document_chunk_count(doc_id, len(chunks))

            dense_texts = [
                _index_text(title, ch.section, et) if et else index_texts[i]
                for i, (ch, et) in enumerate(zip(chunks, embed_texts))
            ]
            to_embed_texts = list(dense_texts)
            to_embed_ids = [cid * 10 for cid in chunk_ids]

            if self.settings.use_hype:
                for cid, ch in zip(chunk_ids, chunks):
                    hqs = self._generate_hype_questions(ch.text)
                    if hqs:
                        self.repo.set_hype_questions(cid, hqs)
                        for idx, q in enumerate(hqs, start=1):
                            to_embed_texts.append(q)
                            to_embed_ids.append(cid * 10 + idx)

            vectors = self.embedder.embed_documents(to_embed_texts)
            main_vectors = vectors[:len(chunk_ids)]
            self.repo.set_embeddings(chunk_ids, main_vectors)
            self.vector.add(vectors, to_embed_ids)
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
        if self.vector.ready:
            dense_started = time.perf_counter()
            qv = self.embed_query(query)
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

        ordered = sorted(candidates, key=_final_score, reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for h in ordered:
            m = meta.get(h.chunk_id)
            if not m:
                continue
            text = m["text"]
            if s.enable_context_window:
                doc_id = m["document_id"]
                current_idx = m["chunk_index"]
                num_neighbors = s.context_window_num_neighbors
                start_idx = max(0, current_idx - num_neighbors)
                end_idx = current_idx + num_neighbors

                neighbors = self.repo.get_neighboring_chunks(doc_id, start_idx, end_idx)
                if neighbors:
                    neighbors = sorted(neighbors, key=lambda x: x["chunk_index"])
                    text = neighbors[0]["text"]
                    for next_ch in neighbors[1:]:
                        text = merge_overlapping_texts(text, next_ch["text"], max_overlap=s.chunk_overlap + 100)

            out.append(
                RetrievedChunk(
                    chunk_id=h.chunk_id,
                    text=text,
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
        logger.info(
            "RAG_FLOW retrieval node=retrieve duration_ms=%.1f bm25_ms=%.1f dense_ms=%.1f fusion_ms=%.1f "
            "meta_ms=%.1f reranker_ms=%.1f query_chars=%s bm25_hits=%s dense_hits=%s fused_hits=%s "
            "candidates=%s reranked=%s returned=%s",
            (time.perf_counter() - started) * 1000,
            bm25_ms,
            dense_ms,
            fusion_ms,
            meta_ms,
            rerank_ms,
            len(query),
            len(bm25_hits),
            len(dense_hits),
            len(fused),
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
        }
