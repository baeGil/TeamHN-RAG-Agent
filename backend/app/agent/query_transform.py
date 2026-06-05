"""Query transformation utilities for query-time RAG enhancement."""
from dataclasses import dataclass
from typing import Iterable

from . import prompts
from .llm import LLM


@dataclass(frozen=True)
class QueryVariant:
    kind: str
    query: str


def _clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def _dedupe(variants: Iterable[QueryVariant], limit: int) -> list[QueryVariant]:
    out: list[QueryVariant] = []
    seen: set[str] = set()
    for v in variants:
        q = _clean(v.query)
        if not q:
            continue
        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(QueryVariant(v.kind, q))
        if len(out) >= limit:
            break
    return out


class QueryTransformer:
    def __init__(self, llm: LLM, max_variants: int = 3) -> None:
        self.llm = llm
        self.max_variants = max(1, max_variants)

    def transform(self, question: str, route: str) -> list[QueryVariant]:
        """Return original + rewrite/step-back variants for retrieval.

        Decomposition is handled by the planner for complex routes; this layer
        focuses on rewriting each retrievable query into stronger search inputs.
        """
        original = QueryVariant("original", question)
        if self.max_variants <= 1:
            return [original]

        msgs = [
            {"role": "system", "content": prompts.QUERY_TRANSFORM_SYSTEM},
            {"role": "user", "content": f"ROUTE: {route}\nCÂU HỎI GỐC:\n{question}"},
        ]
        try:
            data = self.llm.chat_json(msgs, fast=True, node="query_transform")
        except Exception:
            return [original]

        candidates = [original]
        rewrite = data.get("rewrite")
        if isinstance(rewrite, str):
            candidates.append(QueryVariant("rewrite", rewrite))
        step_back = data.get("step_back")
        if isinstance(step_back, str):
            candidates.append(QueryVariant("step_back", step_back))

        deduped = _dedupe(candidates, self.max_variants)

        # Reorder: LLM-selected variant goes first so callers can use deduped[0]
        best_kind = data.get("best", "original")
        selected = next((v for v in deduped if v.kind == best_kind), deduped[0])
        others = [v for v in deduped if v is not selected]
        return [selected] + others
