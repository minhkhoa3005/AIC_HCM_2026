"""Build retrieval text queries from a parsed QueryPlan."""

from __future__ import annotations

from .schemas import QueryPlan
from .task_types import TaskType


def build_clip_queries(plan: QueryPlan) -> list[str]:
    """Convert a ``QueryPlan`` into text queries for ``search_clip_text``.

    Retrieval accepts ``list[str]`` because one user query may become several
    CLIP searches, especially TRAKE where each event should be searched
    independently.
    """

    queries: list[str] = []

    if plan.task_type == TaskType.TRAKE and plan.events:
        queries.extend(event.description for event in plan.events)
    else:
        queries.append(plan.search_description)

    queries.extend(plan.positive_constraints)
    queries.extend(plan.metadata_keywords)
    return _dedupe_non_empty(queries)


def build_rerank_text(plan: QueryPlan) -> str:
    """Create a compact text summary for later rerank/QA prompts."""

    pieces = [plan.search_description]
    if plan.question:
        pieces.append(f"question: {plan.question}")
    if plan.events:
        event_text = " -> ".join(event.description for event in plan.events)
        pieces.append(f"events: {event_text}")
    if plan.negative_constraints:
        pieces.append("avoid: " + ", ".join(plan.negative_constraints))
    return " | ".join(piece for piece in pieces if piece)


def _dedupe_non_empty(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
