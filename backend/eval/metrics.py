"""Retrieval metrics: Recall@k and MRR@k against a relevant set."""


def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(relevant)


def hit_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0
