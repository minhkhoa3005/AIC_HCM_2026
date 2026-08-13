import json
from types import SimpleNamespace

import pytest

from llm.config import LLMConfig
from llm.planner import plan_query
from llm.query_builder import build_clip_queries, build_rerank_text
from llm.schemas import QueryPlan
from llm.task_types import TaskType
from retrieval.mock_retrieval import mock_search_clip_text


def _config():
    return LLMConfig(
        llm_provider="gemini",
        llm_api_keys=("test-key",),
        llm_model="gemini-2.5-flash",
        llm_rewrite_temperature=0,
        llm_planner_temperature=0,
    )


class MockLLMClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _rewrite_payload():
    return {
        "rewritten_query": (
            "Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng, lấy gói hàng ở cốp, "
            "sau đó đi vào tòa nhà có biển hiệu xanh; hỏi trên biển ghi chữ gì?"
        ),
        "preserved_terms": [],
        "uncertain_terms": [],
    }


def _planner_payload():
    return {
        "task_type": "QA",
        "search_description": (
            "người đàn ông mặc áo đỏ mở cửa xe trắng, lấy gói hàng ở cốp, "
            "sau đó đi vào tòa nhà có biển hiệu xanh"
        ),
        "question": "Trên biển hiệu ghi chữ gì?",
        "events": [
            {"event_id": 1, "description": "người đàn ông mặc áo đỏ mở cửa xe trắng"},
            {"event_id": 2, "description": "lấy gói hàng ở cốp xe"},
            {"event_id": 3, "description": "đi vào tòa nhà có biển hiệu xanh"},
        ],
        "entities": [
            {
                "name": "người đàn ông",
                "type": "person",
                "attributes": ["mặc áo đỏ"],
                "actions": [
                    {"verb": "mở cửa", "target": "xe trắng"},
                    {"verb": "lấy gói hàng", "target": "gói hàng"},
                    {"verb": "đi vào", "target": "tòa nhà"},
                ],
            }
        ],
        "objects": ["người đàn ông", "xe trắng", "gói hàng", "cốp xe", "tòa nhà", "biển hiệu"],
        "actions": ["mở cửa", "lấy gói hàng", "đi vào"],
        "scene": ["tòa nhà"],
        "positive_constraints": ["áo đỏ", "xe trắng", "biển hiệu xanh"],
        "negative_constraints": [],
        "metadata_keywords": ["biển hiệu"],
        "clip_queries": [
            "a man in a red shirt opening the door of a white car",
            "a man taking a package from the trunk of a white car",
            "a man entering a building with a green signboard",
            "a green signboard on a building with readable text",
        ],
        "confidence": 0.9,
    }


def test_plan_query_rewrites_then_builds_complete_plan():
    query = "Tim canh ng dan ong mac ao do mo cua xe trang; hoi tren bien ghi chu gi?"
    rewrite_client = MockLLMClient(_rewrite_payload())
    planner_client = MockLLMClient(_planner_payload())

    plan = plan_query(
        query,
        config=_config(),
        rewrite_client=rewrite_client,
        planner_client=planner_client,
    )

    assert rewrite_client.calls
    assert planner_client.calls
    assert isinstance(plan, QueryPlan)
    assert plan.raw_query == query
    assert "Tìm cảnh người đàn ông" in plan.rewritten_query
    assert plan.task_type == TaskType.QA
    assert plan.question == "Trên biển hiệu ghi chữ gì?"
    assert "biển hiệu xanh" in plan.positive_constraints
    assert all(_is_ascii(query) for query in build_clip_queries(plan))
    planner_prompt = planner_client.calls[0]["contents"]
    assert "tiếng Việt có dấu" in planner_prompt
    assert "người đàn ông" in planner_prompt
    assert "PHẦN A — VÍ DỤ THAM CHIẾU VỀ CÁCH TẠO QUERYPLAN" in planner_prompt
    assert "QUY TẮC VISUAL_HINTS" in planner_prompt
    assert "Không ép mọi query mở" in planner_prompt
    assert "a video frame showing" in planner_prompt
    assert "a close-up of a green signboard with readable text" in planner_prompt
    assert "clearly showing which hand is used" not in planner_prompt


def test_planner_owns_task_type_directly():
    rewrite_client = MockLLMClient(
        {"rewritten_query": "Tìm chuỗi sự kiện người mở cửa xe rồi đi vào tòa nhà."}
    )
    payload = _planner_payload()
    payload["task_type"] = "TRAKE"
    payload["question"] = None
    planner_client = MockLLMClient(payload)

    plan = plan_query(
        "Tim chuoi su kien nguoi mo cua xe roi di vao toa nha.",
        config=_config(),
        rewrite_client=rewrite_client,
        planner_client=planner_client,
    )

    assert plan.task_type == TaskType.TRAKE
    assert len(plan.events) == 3


def test_build_rerank_text_keeps_question_and_events():
    plan = QueryPlan(**_planner_payload())

    rerank_text = build_rerank_text(plan)

    assert "question:" in rerank_text
    assert "events:" in rerank_text
    assert "Trên biển hiệu ghi chữ gì?" in rerank_text


def test_plan_query_raises_when_planner_fails():
    rewrite_client = MockLLMClient({"rewritten_query": "Tìm cảnh người đàn ông mở cửa xe."})
    planner_client = MockLLMClient(error=RuntimeError("api unavailable"))

    with pytest.raises(RuntimeError, match="api unavailable"):
        plan_query(
            "Tìm cảnh người đàn ông mở cửa xe.",
            config=_config(),
            rewrite_client=rewrite_client,
            planner_client=planner_client,
        )


def test_plan_query_output_can_feed_mock_retrieval():
    plan = QueryPlan(**_planner_payload())
    candidates = mock_search_clip_text(build_clip_queries(plan), top_k=2)

    assert candidates
    assert candidates[0].metadata["query"]


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True
