from dataclasses import dataclass

from app.retrieval.segments import expand_segments, extract_relevant_segments


@dataclass
class Hit:
    chunk_id: int
    score: float


def test_rse_includes_bridge_chunk_between_relevant_seeds():
    hits = [Hit(1, 1.0), Hit(3, 0.9)]
    meta = {
        1: {"document_id": 10, "chunk_index": 0},
        3: {"document_id": 10, "chunk_index": 2},
    }

    segments = extract_relevant_segments(
        hits,
        meta,
        penalty=0.2,
        max_segment_chunks=4,
    )

    assert len(segments) == 1
    assert segments[0].start_index == 0
    assert segments[0].end_index == 2

    rows = [
        {"id": 1, "document_id": 10, "chunk_index": 0},
        {"id": 2, "document_id": 10, "chunk_index": 1},
        {"id": 3, "document_id": 10, "chunk_index": 2},
    ]
    expanded = expand_segments(
        segments,
        lambda document_id, start, end: [
            r for r in rows
            if r["document_id"] == document_id and start <= r["chunk_index"] <= end
        ],
        max_context_chunks=5,
    )

    assert [r["id"] for r in expanded] == [1, 2, 3]
    assert expanded[1]["rse_seed"] is False


def test_rse_respects_max_segment_length():
    hits = [Hit(1, 1.0), Hit(4, 0.95)]
    meta = {
        1: {"document_id": 10, "chunk_index": 0},
        4: {"document_id": 10, "chunk_index": 3},
    }

    segments = extract_relevant_segments(
        hits,
        meta,
        penalty=0.2,
        max_segment_chunks=2,
    )

    assert segments
    assert segments[0].length <= 2
