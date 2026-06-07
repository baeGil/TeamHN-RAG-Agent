"""Relevant Segment Extraction (RSE).

After reranking produces a scored list of chunks, RSE replaces the naive top-k
selection with *contiguous segments* from each document.  The insight (from
dsRAG / KITE benchmarks) is that relevant information tends to cluster in
documents: grabbing the surrounding context of a highly-scored chunk almost
always improves answer quality.

Algorithm (per document):
  1. Map retrieved chunks to their sequential chunk_index within the document.
  2. Build a score array over the full index range of interest:
       score[i] = retrieval_score[i]  if chunk i was retrieved
                = -irrelevant_penalty  otherwise
  3. Run a max-sum subarray search (Kadane's variant) with a length cap to
     find the best contiguous window.  Multiple non-overlapping segments can
     be returned per document.
  4. Fetch the full text for each segment (including any "bridge" chunks that
     were not retrieved but lie between two retrieved chunks).
  5. Return segments ordered by their total value.

Complexity: O(N * max_len) per document where N is the retrieved span width.
In practice N and max_len are small (< 100), so this takes < 10ms.

References:
  - dsRAG: https://github.com/D-Star-AI/dsRAG
  - KITE benchmark: 42.6% improvement over top-k retrieval
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RseSegment:
    doc_id: int
    doc_title: str
    doc_source: str
    start_idx: int        # chunk_index of first chunk in segment
    end_idx: int          # chunk_index of last chunk in segment (inclusive)
    text: str             # assembled text of all chunks in range
    page: Optional[int]   # page of first chunk
    section: Optional[str]
    value: float          # total relevance value of the segment
    anchor_chunk_id: int  # DB id of the highest-scoring chunk in segment
    chunk_ids: list[int] = field(default_factory=list)  # DB ids of all chunks


def _max_sum_subarrays(
    scores: list[float],
    max_len: int,
    min_value: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Find all non-overlapping max-sum subarrays with length ≤ max_len.

    Uses a greedy approach: repeatedly find the best subarray, record it, then
    zero out its scores so the next call finds the next-best non-overlapping one.

    Returns list of (start, end_inclusive, total_value) sorted by value desc.
    """
    n = len(scores)
    if n == 0:
        return []

    work = list(scores)
    results: list[tuple[int, int, float]] = []

    for _ in range(max(1, n // max(1, max_len))):
        best_val = min_value
        best_start = -1
        best_end = -1

        # Sliding window of size 1..max_len
        for start in range(n):
            total = 0.0
            for end in range(start, min(n, start + max_len)):
                total += work[end]
                if total > best_val:
                    best_val = total
                    best_start = start
                    best_end = end

        if best_start == -1:
            break  # No more positive-value segments

        results.append((best_start, best_end, best_val))

        # Zero out the found segment so next iteration finds next-best
        for i in range(best_start, best_end + 1):
            work[i] = 0.0

    return sorted(results, key=lambda x: x[2], reverse=True)


def relevant_segment_extraction(
    scored_chunks: list[Any],  # list[RetrievedChunk] – avoid circular import
    fetch_range_fn: Callable[[int, int, int], list[dict[str, Any]]],
    irrelevant_penalty: float = 0.2,
    max_segment_chunks: int = 15,
    overall_max_chunks: int = 30,
) -> list[RseSegment]:
    """Run RSE on a list of reranked chunks and return assembled segments.

    Args:
        scored_chunks: Reranked chunks from KnowledgeBase.retrieve() before
                       final slicing.  Each must have: chunk_id, chunk_index,
                       document_id, doc_title, doc_source, score, page, section.
        fetch_range_fn: Callable(doc_id, start_idx, end_idx) → list of DB row
                        dicts ordered by chunk_index (used to get bridge chunks).
        irrelevant_penalty: Value subtracted from every chunk's relevance score.
                            Non-retrieved chunks get -irrelevant_penalty.
                            Set higher to prefer tighter, more focused segments.
        max_segment_chunks: Maximum number of chunks in a single segment.
        overall_max_chunks: Soft cap on total chunks across all segments returned.

    Returns:
        list[RseSegment] ordered by segment value (descending).
    """
    if not scored_chunks:
        return []

    # --- Group retrieved chunks by document ---
    from collections import defaultdict
    doc_groups: dict[int, list[Any]] = defaultdict(list)
    for c in scored_chunks:
        doc_groups[c.document_id].append(c)

    all_segments: list[RseSegment] = []
    total_chunks_used = 0

    for doc_id, chunks in doc_groups.items():
        if total_chunks_used >= overall_max_chunks:
            break

        first = chunks[0]
        doc_title = first.doc_title
        doc_source = first.doc_source

        # Map chunk_index → (score, chunk_id, page, section)
        idx_map: dict[int, tuple[float, int, Optional[int], Optional[str]]] = {
            c.chunk_index: (c.score, c.chunk_id, c.page, c.section)
            for c in chunks
        }

        if not idx_map:
            continue

        min_idx = min(idx_map)
        max_idx = max(idx_map)

        # Extend the window a bit to allow bridging gaps
        window_start = max(0, min_idx)
        window_end = max_idx  # fetch_range_fn will handle out-of-bounds

        # Build score array over [window_start, window_end]
        n = window_end - window_start + 1
        score_arr: list[float] = []
        for abs_idx in range(window_start, window_end + 1):
            if abs_idx in idx_map:
                raw_score, _, _, _ = idx_map[abs_idx]
                score_arr.append(raw_score - irrelevant_penalty)
            else:
                score_arr.append(-irrelevant_penalty)

        # Find best non-overlapping segments
        segments_found = _max_sum_subarrays(
            score_arr,
            max_len=max_segment_chunks,
            min_value=0.0,
        )

        for rel_start, rel_end, seg_value in segments_found:
            if total_chunks_used >= overall_max_chunks:
                break

            abs_start = window_start + rel_start
            abs_end = window_start + rel_end

            # Fetch all chunks in this range (including bridge chunks)
            try:
                chunk_rows = fetch_range_fn(doc_id, abs_start, abs_end)
            except Exception:
                logger.exception("[RSE] fetch_range_fn failed for doc=%s range=[%s,%s]",
                                 doc_id, abs_start, abs_end)
                continue

            if not chunk_rows:
                continue

            # Assemble segment text
            texts = [row["text"] for row in chunk_rows if row.get("text")]
            assembled = "\n\n".join(t.strip() for t in texts if t.strip())
            if not assembled:
                continue

            first_row = chunk_rows[0]
            # Anchor = highest-scoring chunk in this segment
            anchor_id = max(
                (idx_map[r["chunk_index"]][1]
                 for r in chunk_rows
                 if r.get("chunk_index") in idx_map),
                key=lambda cid: next(
                    (idx_map[r["chunk_index"]][0]
                     for r in chunk_rows
                     if idx_map.get(r.get("chunk_index")) and idx_map[r["chunk_index"]][1] == cid),
                    0.0,
                ),
                default=first_row.get("id", 0),
            )

            all_segments.append(RseSegment(
                doc_id=doc_id,
                doc_title=doc_title,
                doc_source=doc_source,
                start_idx=abs_start,
                end_idx=abs_end,
                text=assembled,
                page=first_row.get("page"),
                section=first_row.get("section"),
                value=seg_value,
                anchor_chunk_id=int(anchor_id),
                chunk_ids=[int(r["id"]) for r in chunk_rows if r.get("id")],
            ))
            total_chunks_used += (rel_end - rel_start + 1)

    return sorted(all_segments, key=lambda s: s.value, reverse=True)
