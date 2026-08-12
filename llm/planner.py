"""Hybrid query planner orchestration."""

from __future__ import annotations

from typing import Any

from .classifier import classify_intent
from .complexity import should_use_llm_parser
from .config import LLMConfig, get_llm_config
from .parser import parse_query_plan, parse_query_plan_with_llm
from .parser import parse_query_plan
from .schemas import QueryPlan


def plan_query(
    query: str,
    config: LLMConfig | None = None,
    gemini_client: Any | None = None,
) -> QueryPlan:
    """Plan a user query with local intent, optional Gemini parsing, and fallback."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_config = config or get_llm_config()
    intent = classify_intent(query)

    if not should_use_llm_parser(query, intent, resolved_config):
        return parse_query_plan(query, intent=intent)

    try:
        return parse_query_plan_with_llm(
            query,
            intent=intent,
            config=resolved_config,
            client=gemini_client,
        )
    except Exception:
        return parse_query_plan(query, intent=intent)
