import pytest

from llm.query_builder import build_clip_queries
from llm.schemas import QueryPlan, TrakeEvent
from llm.task_types import TaskType


def test_build_clip_queries_returns_llm_generated_queries():
    plan = QueryPlan(
        task_type=TaskType.TEXTUAL_KIS,
        search_description="nguoi dan ong mac ao do mo cua xe mau trang",
        positive_constraints=["ao do", "xe mau trang"],
        clip_queries=[
            "a man wearing a red shirt opening the door of a white car",
            "white car with an open door",
        ],
    )

    assert build_clip_queries(plan) == [
        "a man wearing a red shirt opening the door of a white car",
        "white car with an open door",
    ]


def test_trake_clip_queries_keep_llm_event_order():
    plan = QueryPlan(
        task_type=TaskType.TRAKE,
        search_description="nguoi mo cua xe, buoc ra ngoai",
        events=[
            TrakeEvent(event_id=1, description="nguoi mo cua xe"),
            TrakeEvent(event_id=2, description="buoc ra ngoai"),
        ],
        clip_queries=["a person opening a car door", "stepping outside"],
    )

    assert build_clip_queries(plan) == [
        "a person opening a car door",
        "stepping outside",
    ]


def test_build_clip_queries_rejects_missing_llm_queries():
    plan = QueryPlan(
        task_type=TaskType.TEXTUAL_KIS,
        search_description="nguoi dan ong mac ao do mo cua xe mau trang",
    )

    with pytest.raises(ValueError, match="clip_queries"):
        build_clip_queries(plan)


def test_build_clip_queries_rejects_non_ascii_queries():
    plan = QueryPlan(
        task_type=TaskType.TEXTUAL_KIS,
        search_description="nguoi dan ong mac ao do",
        clip_queries=["người đàn ông mặc áo đỏ"],
    )

    with pytest.raises(ValueError, match="English-only"):
        build_clip_queries(plan)
