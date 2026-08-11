import json

import pytest

from llm.schemas import Candidate, Entity, EntityAction, QueryPlan, TrakeEvent
from llm.task_types import TaskType
from llm.validator import validate_query_plan
from retrieval.mock_retrieval import mock_search_clip_text


def _dump_model(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_create_query_plan_for_three_tasks():
    textual = QueryPlan(
        task_type=TaskType.TEXTUAL_KIS,
        search_description="nguoi dan ong mac ao do mo cua xe mau trang",
        objects=["nguoi dan ong", "xe", "cua xe"],
        actions=["mo cua"],
    )
    qa = QueryPlan(
        task_type=TaskType.QA,
        search_description="nguoi phu nu dung truoc xe buyt",
        question="bien so xe la gi?",
        objects=["nguoi phu nu", "xe buyt", "bien so"],
    )
    trake = QueryPlan(
        task_type=TaskType.TRAKE,
        search_description="nguoi mo cua xe, buoc ra ngoai, roi di vao toa nha",
        events=[
            TrakeEvent(event_id=1, description="nguoi mo cua xe"),
            TrakeEvent(event_id=2, description="nguoi buoc ra ngoai xe"),
            TrakeEvent(event_id=3, description="nguoi di vao toa nha"),
        ],
    )

    assert textual.task_type == TaskType.TEXTUAL_KIS
    assert qa.question == "bien so xe la gi?"
    assert [event.event_id for event in trake.events] == [1, 2, 3]


def test_entity_action_keeps_target_relationship():
    entity = Entity(
        name="nguoi dan ong",
        type="person",
        attributes=["ao do"],
        actions=[EntityAction(verb="mo", target="cua xe")],
    )

    assert entity.actions[0].verb == "mo"
    assert entity.actions[0].target == "cua xe"


def test_candidate_schema_and_json_round_trip():
    candidate = Candidate(
        video_id="L01_V001",
        frame_id=1500,
        keyframe_path="keyframes/L01_V001/01500.jpg",
        clip_score=0.82,
        metadata={"title": "street scene"},
        objects=["person", "car", "door"],
    )

    payload = json.loads(candidate.model_dump_json())
    recreated = Candidate(**payload)

    assert recreated.video_id == "L01_V001"
    assert recreated.metadata["title"] == "street scene"
    assert recreated.objects == ["person", "car", "door"]


def test_query_plan_json_round_trip():
    plan = QueryPlan(
        task_type=TaskType.TEXTUAL_KIS,
        search_description="nguoi cam o di qua vach sang duong",
        entities=[
            Entity(
                name="nguoi",
                type="person",
                actions=[EntityAction(verb="di qua", target="vach sang duong")],
            )
        ],
        confidence=0.86,
    )

    recreated = QueryPlan(**json.loads(plan.model_dump_json()))

    assert recreated.task_type == TaskType.TEXTUAL_KIS
    assert recreated.entities[0].actions[0].target == "vach sang duong"
    assert recreated.confidence == 0.86


def test_validator_repairs_missing_lists_and_confidence():
    plan = validate_query_plan(
        {
            "task_type": "QA",
            "search_description": "nguoi phu nu dung truoc xe buyt",
            "question": "bien so xe la gi?",
        }
    )

    assert plan.task_type == TaskType.QA
    assert plan.events == []
    assert plan.entities == []
    assert plan.confidence == 0.5


def test_validator_clamps_confidence():
    high = validate_query_plan(
        {
            "task_type": "TEXTUAL_KIS",
            "search_description": "xe buyt mau vang",
            "confidence": 1.5,
        }
    )
    low = validate_query_plan(
        {
            "task_type": "TEXTUAL_KIS",
            "search_description": "xe buyt mau vang",
            "confidence": -0.25,
        }
    )

    assert high.confidence == 1.0
    assert low.confidence == 0.0


def test_validator_rejects_invalid_task_type():
    with pytest.raises(ValueError, match="Invalid task_type"):
        validate_query_plan({"task_type": "IMAGE_SEARCH", "search_description": "test"})


def test_mock_retrieval_returns_candidates():
    candidates = mock_search_clip_text(["nguoi mo cua xe"], top_k=2)

    assert len(candidates) == 2
    assert all(isinstance(candidate, Candidate) for candidate in candidates)
    assert candidates[0].clip_score >= candidates[1].clip_score
