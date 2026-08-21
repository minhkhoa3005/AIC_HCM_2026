"""Regression tests for malformed local Rewrite and Planner output."""

from dataclasses import replace

from llm.config import get_llm_config
from llm.rewrite import RewrittenQuery
from llm.task_types import TaskType


def test_rewrite_falls_back_to_raw_query(monkeypatch) -> None:
    from llm import rewrite

    def fail_generation(*args, **kwargs):
        raise ValueError("truncated local JSON")

    monkeypatch.setattr(rewrite, "generate_llm_json", fail_generation)
    result = rewrite.rewrite_query_with_llm(
        "tim nguoi mo laptop",
        config=get_llm_config(load_env=False),
    )

    assert result.raw_query == "tim nguoi mo laptop"
    assert result.rewritten_query == result.raw_query
    assert result.preserved_terms == []
    assert result.uncertain_terms == []


def test_planner_retries_once_after_invalid_json(monkeypatch) -> None:
    from llm import planner

    monkeypatch.setattr(
        planner,
        "rewrite_query_with_llm",
        lambda *args, **kwargs: RewrittenQuery(
            raw_query="Tìm bốn phi hành gia mặc áo đen.",
            rewritten_query="Tìm bốn phi hành gia mặc áo đen.",
        ),
    )
    calls = 0
    token_limits = []

    def generate_plan(*args, **kwargs):
        nonlocal calls
        calls += 1
        token_limits.append(kwargs["max_new_tokens"])
        if calls == 1:
            raise ValueError("incomplete JSON")
        return {
            "task_type": "TEXTUAL_KIS",
            "search_description": "Bốn phi hành gia mặc áo đen.",
            "question": None,
            "anchor": "four astronauts wearing black suits",
            "expansions": [
                f"four astronauts wearing black suits view {index}"
                for index in range(10)
            ],
            "objects": ["phi hành gia"],
            "actions": [],
            "negative_constraints": [],
            "metadata_keywords": [],
            "events": [],
            "event_queries": [],
        }

    monkeypatch.setattr(planner, "generate_llm_json", generate_plan)
    config = replace(
        get_llm_config(load_env=False),
        local_text_max_new_tokens=768,
    )
    result = planner.plan_query(
        "Tìm bốn phi hành gia mặc áo đen.",
        TaskType.TEXTUAL_KIS,
        config=config,
    )

    assert calls == 2
    assert token_limits == [1024, 1536]
    assert result.anchor == "four astronauts wearing black suits"
    assert len(result.expansions) == 10
