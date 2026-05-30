"""Reciprocal Rank Fusion (RRF) of BM25 + dense results."""
from dataclasses import dataclass, field


@dataclass
class FusedHit:
    chunk_id: int
    rrf_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    sources: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    bm25: list[tuple[int, float]],
    dense: list[tuple[int, float]],
    k: int = 60,
) -> list[FusedHit]:
    hits: dict[int, FusedHit] = {}

    def _ensure(cid: int) -> FusedHit:
        if cid not in hits:
            hits[cid] = FusedHit(chunk_id=cid, rrf_score=0.0)
        return hits[cid]

    for rank, (cid, score) in enumerate(bm25, start=1):
        h = _ensure(cid)
        h.rrf_score += 1.0 / (k + rank)
        h.bm25_rank = rank
        h.bm25_score = score
        h.sources.append("bm25")

    for rank, (cid, score) in enumerate(dense, start=1):
        h = _ensure(cid)
        h.rrf_score += 1.0 / (k + rank)
        h.dense_rank = rank
        h.dense_score = score
        h.sources.append("dense")

    return sorted(hits.values(), key=lambda h: h.rrf_score, reverse=True)
