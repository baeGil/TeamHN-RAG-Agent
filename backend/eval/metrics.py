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


def precision_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for rid in topk if rid in relevant) / len(topk)


def ndcg_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            import math

            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    import math

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def context_precision_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    """Context Precision (RAGAS-style): Average Precision@k.

    Measures whether relevant chunks are ranked *before* irrelevant ones.
    AP = Σ (precision@i × rel_i) / min(|relevant|, k)
    Unlike precision@k (flat count), this rewards finding relevant chunks early.
    """
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            hits += 1
            score += hits / i          # precision at this rank
    denom = min(len(relevant), k)
    return score / denom if denom else 0.0


def redundancy_ratio(texts: list[str]) -> float:
    """Approximate top-k redundancy by duplicate normalized token overlap.

    Returns 0 for no redundancy and approaches 1 as later chunks repeat tokens
    already seen in earlier chunks.
    """
    seen: set[str] = set()
    repeated = 0
    total = 0
    for text in texts:
        tokens = {
            t.strip(".,:;!?()[]{}\"'`").lower()
            for t in text.split()
            if len(t.strip(".,:;!?()[]{}\"'`")) > 2
        }
        if not tokens:
            continue
        repeated += len(tokens & seen)
        total += len(tokens)
        seen.update(tokens)
    return repeated / total if total else 0.0


def average_score(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def tail_query_success(rows: list[dict], method: str, k: int, tail_label: str = "hard") -> float:
    tail = [r for r in rows if r.get("difficulty") == tail_label]
    if not tail:
        return 0.0
    return average_score([
        r.get("methods", {}).get(method, {}).get(f"hit@{k}", 0.0)
        for r in tail
    ])
