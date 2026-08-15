"""Build retrieval text queries from a parsed QueryPlan."""

from __future__ import annotations

from .schemas import QueryPlan
from .task_types import TaskType


_PROHIBITED_PURPOSE_PHRASES = (
    "showing clothing details",
    "showing their clothing details",
    "of a specific color",
    "showing if",
    "to identify",
    "for ocr",
)


def build_clip_queries(plan: QueryPlan) -> list[str]:
    """Return LLM-generated English CLIP queries for ``search_clip_text``."""

    queries = [query.strip() for query in plan.clip_queries if query.strip()]
    if not queries:
        raise ValueError("QueryPlan.clip_queries must contain English CLIP queries")
    if len(queries) != len(plan.clip_queries):
        raise ValueError("clip_queries must not contain empty strings")
    if len(set(queries)) != len(queries):
        raise ValueError("clip_queries must not contain duplicate queries")
    invalid = [query for query in queries if not _is_ascii(query)]
    if invalid:
        raise ValueError(f"clip_queries must be English-only ASCII strings: {invalid!r}")
    prohibited = [
        query
        for query in queries
        if any(phrase in query.lower() for phrase in _PROHIBITED_PURPOSE_PHRASES)
    ]
    if prohibited:
        raise ValueError(
            "clip_queries must describe concrete visual evidence, not inspection goals: "
            f"{prohibited!r}"
        )
    _validate_query_budget(plan, queries)
    return queries


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


def _validate_query_budget(plan: QueryPlan, queries: list[str]) -> None:
    if plan.task_type == TaskType.TEXTUAL_KIS and not 1 <= len(queries) <= 2:
        raise ValueError("TEXTUAL_KIS plans must contain one or two clip queries")
    if plan.task_type == TaskType.QA and not 1 <= len(queries) <= 4:
        raise ValueError("QA plans must contain between one and four clip queries")
    if plan.task_type == TaskType.TRAKE and len(queries) != len(plan.events):
        raise ValueError("TRAKE plans must contain exactly one clip query per event")


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True
