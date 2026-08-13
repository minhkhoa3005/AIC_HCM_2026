"""End-to-end query pipeline controller."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from llm.config import LLMConfig
from llm.planner import plan_query
from llm.query_builder import build_clip_queries
from llm.schemas import Candidate
from llm.task_types import TaskType
from retrieval.contract import search_clip_text

from .checkpoint import QueryCheckpoint
from .prediction import Prediction
from .qa import answer_qa
from .rerank import rerank_candidates
from .trake import align_trake

SearchFn = Callable[[list[str], int], list[Candidate]]


def run_query(
    query_id: str,
    query: str,
    *,
    top_k: int = 100,
    search_fn: SearchFn = search_clip_text,
    config: LLMConfig | None = None,
    llm_rewrite_client: Any | None = None,
    llm_planner_client: Any | None = None,
) -> list[Prediction]:
    """Run rewrite -> plan -> English CLIP queries -> search -> task handler."""

    plan = plan_query(
        query,
        config=config,
        rewrite_client=llm_rewrite_client,
        planner_client=llm_planner_client,
    )
    clip_queries = build_clip_queries(plan)
    candidates = rerank_candidates(search_fn(clip_queries, top_k), plan)

    if plan.task_type == TaskType.QA:
        return answer_qa(query_id, plan, candidates)
    if plan.task_type == TaskType.TRAKE:
        return align_trake(query_id, plan, candidates)
    return [
        Prediction(
            query_id=query_id,
            task_type=plan.task_type,
            rank=index,
            video_id=candidate.video_id,
            frame_ids=[candidate.frame_id],
            score=candidate.clip_score,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def run_batch(
    queries: Iterable[tuple[str, str]],
    *,
    top_k: int = 100,
    search_fn: SearchFn = search_clip_text,
    checkpoint: QueryCheckpoint | None = None,
    config: LLMConfig | None = None,
    llm_rewrite_client: Any | None = None,
    llm_planner_client: Any | None = None,
) -> list[Prediction]:
    """Run a batch, optionally skipping query IDs present in a checkpoint."""

    done = checkpoint.done_query_ids() if checkpoint else set()
    predictions: list[Prediction] = []
    for query_id, query in queries:
        if query_id in done:
            continue
        query_predictions = run_query(
            query_id,
            query,
            top_k=top_k,
            search_fn=search_fn,
            config=config,
            llm_rewrite_client=llm_rewrite_client,
            llm_planner_client=llm_planner_client,
        )
        predictions.extend(query_predictions)
        if checkpoint:
            checkpoint.append(query_id, query_predictions)
    return predictions
