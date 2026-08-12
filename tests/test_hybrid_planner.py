import json
from types import SimpleNamespace

from llm.complexity import is_complex_query, should_use_llm_parser
from llm.config import LLMConfig
from llm.parser import parse_query_plan
from llm.planner import plan_query
from llm.query_builder import build_clip_queries
from llm.schemas import IntentClassification
from llm.task_types import TaskType
from retrieval.mock_retrieval import mock_search_clip_text


def _config(use_llm_classifier=False, use_llm_parser=True, mode="auto"):
    return LLMConfig(
        llm_provider="gemini",
        llm_api_key="test-key",
        llm_model="gemini-2.5-flash",
        use_llm_classifier=use_llm_classifier,
        llm_classifier_temperature=0,
        intent_confidence_threshold=0.8,
        use_llm_parser=use_llm_parser,
        llm_parser_mode=mode,
        llm_parser_temperature=0,
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
        return SimpleNamespace(text=json.dumps(self.payload))


def test_plan_query_local_mode_uses_local_parser_without_gemini():
    query = "Tim canh nguoi dan ong mac ao do mo cua xe mau trang."
    client = MockLLMClient()

    plan = plan_query(query, config=_config(use_llm_parser=False), llm_parser_client=client)

    assert plan == parse_query_plan(query)
    assert client.calls == []


def test_auto_gate_skips_simple_query_and_uses_complex_query():
    simple_intent = IntentClassification(task_type=TaskType.TEXTUAL_KIS, confidence=0.9)
    complex_intent = IntentClassification(task_type=TaskType.TRAKE, confidence=0.92)
    low_confidence = IntentClassification(task_type=TaskType.TEXTUAL_KIS, confidence=0.55)

    assert not is_complex_query("Tim canh xe buyt mau vang.", simple_intent)
    assert is_complex_query(
        "Tim chuoi su kien nguoi mo cua xe, buoc ra ngoai, roi di vao toa nha.",
        complex_intent,
    )
    assert is_complex_query("Canh gan quay trong sieu thi", low_confidence)

    assert not should_use_llm_parser(
        "Tim canh xe buyt mau vang.",
        simple_intent,
        _config(use_llm_parser=True, mode="auto"),
    )
    assert should_use_llm_parser(
        "Tim canh xe buyt mau vang.",
        simple_intent,
        _config(use_llm_parser=True, mode="always"),
    )


def test_plan_query_uses_mock_llm_parser_for_complex_query():
    payload = {
        "task_type": "TRAKE",
        "search_description": "nguoi mo cua xe, buoc ra ngoai, di vao toa nha",
        "question": None,
        "events": [
            {"event_id": 1, "description": "nguoi mo cua xe"},
            {"event_id": 2, "description": "nguoi buoc ra ngoai"},
            {"event_id": 3, "description": "nguoi di vao toa nha"},
        ],
        "entities": [],
        "objects": ["nguoi", "xe", "toa nha"],
        "actions": ["mo cua", "buoc ra ngoai", "di vao"],
        "scene": ["toa nha"],
        "positive_constraints": ["nguoi", "xe", "toa nha"],
        "negative_constraints": [],
        "metadata_keywords": [],
        "confidence": 0.93,
    }
    client = MockLLMClient(payload)

    plan = plan_query(
        "Tim chuoi su kien nguoi mo cua xe, buoc ra ngoai, roi di vao toa nha.",
        config=_config(use_llm_parser=True, mode="always"),
        llm_parser_client=client,
    )

    assert client.calls
    assert plan.task_type == TaskType.TRAKE
    assert [event.description for event in plan.events] == [
        "nguoi mo cua xe",
        "nguoi buoc ra ngoai",
        "nguoi di vao toa nha",
    ]


def test_plan_query_falls_back_to_local_parser_when_llm_parser_fails():
    query = "Tim canh xe dung lai, nguoi buoc xuong, roi lay hang tu cop xe."
    client = MockLLMClient(error=RuntimeError("api unavailable"))

    plan = plan_query(query, config=_config(mode="always"), llm_parser_client=client)

    assert client.calls
    assert plan == parse_query_plan(query)


def test_low_confidence_intent_uses_llm_classifier_before_parser():
    classifier_client = MockLLMClient(
        {
            "task_type": "QA",
            "confidence": 0.96,
            "reason": "The query asks what appears in the scene.",
        }
    )
    parser_client = MockLLMClient(
        {
            "task_type": "TEXTUAL_KIS",
            "search_description": "nguoi trong sieu thi cam vat",
            "question": "nguoi trong sieu thi cam cai gi?",
            "events": [],
            "entities": [],
            "objects": ["nguoi"],
            "actions": ["cam"],
            "scene": ["sieu thi"],
            "positive_constraints": ["nguoi", "cam", "sieu thi"],
            "negative_constraints": [],
            "metadata_keywords": [],
            "confidence": 0.91,
        }
    )

    plan = plan_query(
        "Canh gan quay trong sieu thi nguoi cam vat nao",
        config=_config(use_llm_classifier=True, use_llm_parser=True, mode="always"),
        llm_classifier_client=classifier_client,
        llm_parser_client=parser_client,
    )

    assert classifier_client.calls
    assert parser_client.calls
    assert plan.task_type == TaskType.QA
    assert plan.question == "nguoi trong sieu thi cam cai gi?"


def test_plan_query_output_feeds_mock_retrieval():
    plan = plan_query(
        "Tim canh nguoi dan ong mac ao do mo cua xe mau trang.",
        config=_config(use_llm_parser=False),
    )
    candidates = mock_search_clip_text(build_clip_queries(plan), top_k=2)

    assert candidates
    assert candidates[0].metadata["query"]
