from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    route: str
    route_reason: str
    subquestions: list[str]
    steps: list[dict[str, Any]]
    retrieved: dict[int, dict[str, Any]]  # chunk_id -> chunk dict (deduped)
    context_chunks: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    iterations: int
