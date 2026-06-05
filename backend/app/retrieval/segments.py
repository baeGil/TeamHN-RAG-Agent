"""Relevant Segment Extraction (RSE) for post-retrieval context building."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Segment:
    document_id: int
    start_index: int
    end_index: int
    score: float
    seed_chunk_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.end_index - self.start_index + 1


def _candidate_value(score: float, max_score: float, rank: int, penalty: float) -> float:
    normalized = score / max_score if max_score > 0 else 0.0
    rank_bonus = 1.0 / rank
    return (0.70 * normalized) + (0.30 * rank_bonus) - penalty


def _best_bounded_subarray(values: list[float], max_len: int) -> tuple[int, int, float]:
    best_start = 0
    best_end = 0
    best_score = float("-inf")
    max_len = max(1, max_len)
    for start in range(len(values)):
        total = 0.0
        for end in range(start, min(len(values), start + max_len)):
            total += values[end]
            if total > best_score:
                best_start = start
                best_end = end
                best_score = total
    return best_start, best_end, best_score


def extract_relevant_segments(
    ranked_hits: list[Any],
    meta: dict[int, dict[str, Any]],
    *,
    penalty: float = 0.20,
    max_segment_chunks: int = 4,
) -> list[Segment]:
    """Convert ranked chunk hits into contiguous relevant document segments.

    Retrieved chunks get positive evidence values based on normalized score and
    rank. Missing bridge chunks inside the span get a negative value, so the
    selected segment includes them only when they preserve enough nearby signal.
    """
    if not ranked_hits:
        return []

    max_score = max(float(getattr(h, "score", 0.0) or 0.0) for h in ranked_hits) or 1.0
    per_doc: dict[int, dict[int, dict[str, Any]]] = {}
    for rank, hit in enumerate(ranked_hits, start=1):
        row = meta.get(hit.chunk_id)
        if not row:
            continue
        doc_id = int(row["document_id"])
        chunk_index = int(row["chunk_index"])
        value = _candidate_value(float(getattr(hit, "score", 0.0) or 0.0), max_score, rank, penalty)
        existing = per_doc.setdefault(doc_id, {}).get(chunk_index)
        if existing is None or value > existing["value"]:
            per_doc[doc_id][chunk_index] = {
                "value": value,
                "chunk_id": int(hit.chunk_id),
            }

    segments: list[Segment] = []
    for doc_id, indexed in per_doc.items():
        if not indexed:
            continue
        lo = min(indexed)
        hi = max(indexed)
        values = [indexed.get(i, {"value": -penalty})["value"] for i in range(lo, hi + 1)]
        rel_start, rel_end, score = _best_bounded_subarray(values, max_segment_chunks)
        if score <= 0:
            continue
        start = lo + rel_start
        end = lo + rel_end
        seed_ids = tuple(
            int(v["chunk_id"])
            for idx, v in indexed.items()
            if start <= idx <= end
        )
        segments.append(Segment(doc_id, start, end, score, seed_ids))

    return sorted(segments, key=lambda s: s.score, reverse=True)


def expand_segments(
    segments: list[Segment],
    fetch_range: Callable[[int, int, int], list[dict[str, Any]]],
    *,
    max_context_chunks: int,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    seen: set[int] = set()
    for segment in segments:
        for row in fetch_range(segment.document_id, segment.start_index, segment.end_index):
            cid = int(row["id"])
            if cid in seen:
                continue
            enriched = dict(row)
            enriched["rse_segment_start"] = segment.start_index
            enriched["rse_segment_end"] = segment.end_index
            enriched["rse_segment_score"] = segment.score
            enriched["rse_seed"] = cid in segment.seed_chunk_ids
            expanded.append(enriched)
            seen.add(cid)
            if len(expanded) >= max_context_chunks:
                return expanded
    return expanded
