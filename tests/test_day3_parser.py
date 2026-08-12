from llm.parser import parse_query_plan
from llm.examples import DAY1_QUERY_EXAMPLES
from llm.query_builder import build_clip_queries, build_rerank_text
from llm.schemas import QueryPlan
from llm.task_types import TaskType
from retrieval.mock_retrieval import mock_search_clip_text


def test_parse_textual_kis_query_plan():
    plan = parse_query_plan("Tim canh nguoi dan ong mac ao do mo cua xe mau trang.")

    assert plan.task_type == TaskType.TEXTUAL_KIS
    assert plan.search_description == "nguoi dan ong mac ao do mo cua xe mau trang"
    assert "nguoi dan ong" in plan.objects
    assert "xe" in plan.objects
    assert "mo cua" in plan.actions
    assert plan.question is None


def test_parse_qa_query_plan():
    plan = parse_query_plan(
        "Trong canh nguoi phu nu dung truoc xe buyt, bien so xe la gi?"
    )

    assert plan.task_type == TaskType.QA
    assert plan.search_description == "nguoi phu nu dung truoc xe buyt"
    assert plan.question == "bien so xe la gi?"
    assert "bien so" in plan.metadata_keywords


def test_parse_trake_query_plan_events():
    plan = parse_query_plan(
        "Tim chuoi su kien nguoi mo cua xe, buoc ra ngoai, roi di vao toa nha."
    )

    assert plan.task_type == TaskType.TRAKE
    assert [event.event_id for event in plan.events] == [1, 2, 3]
    assert plan.events[0].description == "nguoi mo cua xe"
    assert plan.events[-1].description == "di vao toa nha"


def test_entity_actions_keep_targets_after_parsing():
    plan = parse_query_plan("Tim canh nguoi dan ong mac ao do mo cua xe mau trang.")

    person = next(entity for entity in plan.entities if entity.name == "nguoi dan ong")

    assert person.type == "person"
    assert person.actions[0].verb == "mo cua"
    assert person.actions[0].target == "cua"


def test_build_clip_queries_for_textual_and_qa():
    textual = parse_query_plan("Tim canh xe may chay qua nga tu vao buoi toi.")
    qa = parse_query_plan("Dong chu tren bien hieu cua quan la gi?")

    assert textual.search_description in build_clip_queries(textual)
    assert "chay qua" in build_clip_queries(textual)
    assert qa.search_description in build_clip_queries(qa)
    assert "bien hieu" in build_clip_queries(qa)


def test_build_clip_queries_for_trake_uses_each_event():
    plan = parse_query_plan("Tim canh xe dung lai, nguoi buoc xuong, roi lay hang tu cop xe.")
    queries = build_clip_queries(plan)

    assert "xe dung lai" in queries
    assert "nguoi buoc xuong" in queries
    assert "lay hang tu cop xe" in queries


def test_day3_plan_can_feed_mock_retrieval():
    plan = parse_query_plan("Tim canh nguoi dan ong mac ao do mo cua xe mau trang.")
    candidates = mock_search_clip_text(build_clip_queries(plan), top_k=2)

    assert isinstance(plan, QueryPlan)
    assert len(candidates) == 2
    assert candidates[0].metadata["query"]


def test_build_rerank_text_contains_question_and_events():
    qa = parse_query_plan("Nguoi dan ong dang cam vat gi khi buoc vao cua hang?")
    trake = parse_query_plan(
        "Tim doan nguoi lay dien thoai ra, chup anh, sau do cat dien thoai vao tui."
    )

    assert "question:" in build_rerank_text(qa)
    assert "events:" in build_rerank_text(trake)


def test_parser_handles_all_day1_examples():
    for sample in DAY1_QUERY_EXAMPLES:
        plan = parse_query_plan(sample["input"])
        queries = build_clip_queries(plan)

        assert plan.task_type.value == sample["expected_task_type"]
        assert plan.search_description
        assert queries
