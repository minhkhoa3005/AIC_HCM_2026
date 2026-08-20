"""End-to-end query pipeline controller."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from llm.config import LLMConfig
from llm.planner import plan_query
from llm.query_builder import TextEncoder, build_clip_queries, build_trake_queries
from llm.task_types import TaskType, infer_task_type_from_name, normalize_task_type
from retrieval.contract import search_clip_text, validate_retrieval_results
from retrieval.models import QueryRetrievalResult

from .checkpoint import QueryCheckpoint
from .kis import rank_kis
from .prediction import Prediction
from .qa import VQAFn, answer_qa
from .rerank import DEFAULT_RRF_WEIGHTS, VLMRankFn, fuse_retrieval_results
from .trake import align_trake

SearchFn = Callable[[list[str], int], list[QueryRetrievalResult]]


def run_query(
    query_id: str,
    query: str,
    *,
    task_type: TaskType | str | None = None,
    top_k: int = 100,
    search_fn: SearchFn = search_clip_text,
    config: LLMConfig | None = None,
    llm_rewrite_client: Any | None = None,
    llm_planner_client: Any | None = None,
    text_encoder: TextEncoder | None = None,
    vqa_fn: VQAFn | None = None,
    vlm_rank_fn: VLMRankFn | None = None,
    rrf_weights: list[float] | None = None,
    search_cache: dict[tuple[tuple[str, ...], int], list[QueryRetrievalResult]] | None = None,
) -> list[Prediction]:
    """Run rewrite -> plan -> English CLIP queries -> search -> task handler."""

    resolved_task_type = normalize_task_type(task_type) if task_type is not None else infer_task_type_from_name(query_id)
    plan = plan_query(
        query,
        task_type=resolved_task_type,
        config=config,
        rewrite_client=llm_rewrite_client,
        planner_client=llm_planner_client,
    )
    clip_queries = (
        build_trake_queries(plan)
        if plan.task_type == TaskType.TRAKE
        else build_clip_queries(plan, text_encoder=text_encoder)
    )
    coarse_top_k = max(200, top_k)
    cache_key = (tuple(clip_queries), coarse_top_k)
    if search_cache is not None and cache_key in search_cache:
        retrieval_results = search_cache[cache_key]
    else:
        retrieval_results = validate_retrieval_results(
            search_fn(clip_queries, coarse_top_k), clip_queries, coarse_top_k
        )
        if search_cache is not None:
            search_cache[cache_key] = retrieval_results

    if plan.task_type == TaskType.TRAKE:
        return align_trake(query_id, plan, retrieval_results)

    candidates = fuse_retrieval_results(
        retrieval_results,
        plan,
        query_weights=rrf_weights or list(DEFAULT_RRF_WEIGHTS),
        limit=300,
    )

    if plan.task_type == TaskType.QA:
        return answer_qa(
            query_id,
            plan,
            candidates,
            vqa_fn=vqa_fn,
            vlm_rank_fn=vlm_rank_fn,
            limit=100,
        )
    return rank_kis(query_id, plan, candidates, vlm_rank_fn=vlm_rank_fn, limit=100)


def run_batch(
    queries: Iterable[tuple[str, str] | tuple[str, str, TaskType | str]],
    *,
    top_k: int = 100,
    search_fn: SearchFn = search_clip_text,
    checkpoint: QueryCheckpoint | None = None,
    config: LLMConfig | None = None,
    llm_rewrite_client: Any | None = None,
    llm_planner_client: Any | None = None,
    text_encoder: TextEncoder | None = None,
    vqa_fn: VQAFn | None = None,
    vlm_rank_fn: VLMRankFn | None = None,
    rrf_weights: list[float] | None = None,
) -> list[Prediction]:
    """Run a batch, optionally skipping query IDs present in a checkpoint."""

    done = checkpoint.done_query_ids() if checkpoint else set()
    predictions: list[Prediction] = []
    search_cache: dict[tuple[tuple[str, ...], int], list[QueryRetrievalResult]] = {}
    for row in queries:
        if len(row) == 2:
            query_id, query = row
            task_type = infer_task_type_from_name(query_id)
        elif len(row) == 3:
            query_id, query, raw_task_type = row
            task_type = normalize_task_type(raw_task_type)
        else:
            raise ValueError("queries must contain (query_id, query) or (query_id, query, task_type)")

        if not isinstance(query_id, str):
            raise ValueError("query_id must be a string")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if query_id in done:
            continue
        query_predictions = run_query(
            query_id,
            query,
            task_type=task_type,
            top_k=top_k,
            search_fn=search_fn,
            config=config,
            llm_rewrite_client=llm_rewrite_client,
            llm_planner_client=llm_planner_client,
            text_encoder=text_encoder,
            vqa_fn=vqa_fn,
            vlm_rank_fn=vlm_rank_fn,
            rrf_weights=rrf_weights,
            search_cache=search_cache,
        )
        predictions.extend(query_predictions)
        if checkpoint:
            checkpoint.append(query_id, query_predictions)
    return predictions
