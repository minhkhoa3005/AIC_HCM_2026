"""Build retrieval text queries from a parsed QueryPlan."""

from __future__ import annotations

from .schemas import QueryPlan


def build_clip_queries(plan: QueryPlan) -> list[str]:
    """Return LLM-generated English CLIP queries for ``search_clip_text``."""

    queries = _dedupe_non_empty(plan.clip_queries)
    if not queries:
        raise ValueError("QueryPlan.clip_queries must contain English CLIP queries")
    invalid = [query for query in queries if not _is_ascii(query)]
    if invalid:
        raise ValueError(f"clip_queries must be English-only ASCII strings: {invalid!r}")
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


def _dedupe_non_empty(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True
