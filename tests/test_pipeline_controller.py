import json
from types import SimpleNamespace

from llm.config import LLMConfig
from pipeline.checkpoint import QueryCheckpoint
from pipeline.controller import run_batch, run_query
from retrieval.mock_retrieval import mock_search_clip_text


def _offline_config():
    return LLMConfig(
        llm_provider="gemini",
        llm_api_keys=(),
        llm_model="gemini-2.5-flash",
        llm_rewrite_temperature=0,
        llm_planner_temperature=0,
    )


class MockLLMClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _rewrite_client():
    return MockLLMClient(
        {"rewritten_query": "Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng."}
    )


def _planner_client(task_type="TEXTUAL_KIS"):
    return MockLLMClient(
        {
            "task_type": task_type,
            "search_description": "người đàn ông mặc áo đỏ mở cửa xe trắng",
            "question": None,
            "events": [],
            "entities": [],
            "objects": ["người đàn ông", "xe trắng", "cửa xe"],
            "actions": ["mở cửa"],
            "scene": [],
            "positive_constraints": ["áo đỏ", "xe trắng"],
            "negative_constraints": [],
            "metadata_keywords": [],
            "clip_queries": [
                "a man in a red shirt opening the door of a white car",
                "a white car with an open door",
            ],
            "confidence": 0.92,
        }
    )


def test_run_query_textual_kis_returns_predictions():
    predictions = run_query(
        "Q001",
        "Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng.",
        top_k=2,
        search_fn=mock_search_clip_text,
        config=_offline_config(),
        llm_rewrite_client=_rewrite_client(),
        llm_planner_client=_planner_client(),
    )

    assert len(predictions) == 2
    assert predictions[0].query_id == "Q001"
    assert predictions[0].frame_ids == [1500]


def test_checkpoint_resume_skips_done_query(tmp_path):
    checkpoint = QueryCheckpoint(tmp_path / "checkpoint.jsonl")
    predictions = run_batch(
        [("Q001", "Tìm cảnh xe buýt màu vàng."), ("Q002", "Tìm cảnh xe máy.")],
        top_k=1,
        search_fn=mock_search_clip_text,
        checkpoint=checkpoint,
        config=_offline_config(),
        llm_rewrite_client=_rewrite_client(),
        llm_planner_client=_planner_client(),
    )
    resumed = run_batch(
        [("Q001", "Tìm cảnh xe buýt màu vàng."), ("Q002", "Tìm cảnh xe máy.")],
        top_k=1,
        search_fn=mock_search_clip_text,
        checkpoint=checkpoint,
        config=_offline_config(),
        llm_rewrite_client=_rewrite_client(),
        llm_planner_client=_planner_client(),
    )

    assert len(predictions) == 2
    assert resumed == []
