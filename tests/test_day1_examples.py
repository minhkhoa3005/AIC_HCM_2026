from collections import Counter

from llm.examples import DAY1_QUERY_EXAMPLES
from llm.task_types import TaskType


def test_examples_have_required_fields():
    for sample in DAY1_QUERY_EXAMPLES:
        assert sample["input"]
        assert sample["expected_task_type"]
        assert sample["notes"]


def test_examples_cover_three_tasks():
    task_types = {sample["expected_task_type"] for sample in DAY1_QUERY_EXAMPLES}

    assert task_types == {task.value for task in TaskType}


def test_examples_have_minimum_total_count():
    assert len(DAY1_QUERY_EXAMPLES) >= 50


def test_examples_have_required_count_per_task():
    counts = Counter(sample["expected_task_type"] for sample in DAY1_QUERY_EXAMPLES)

    assert counts["TEXTUAL_KIS"] >= 20
    assert counts["QA"] >= 15
    assert counts["TRAKE"] >= 15


def test_examples_expected_task_types_are_valid():
    valid_task_types = {task.value for task in TaskType}

    assert all(
        sample["expected_task_type"] in valid_task_types
        for sample in DAY1_QUERY_EXAMPLES
    )
