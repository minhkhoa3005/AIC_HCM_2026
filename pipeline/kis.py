"""TEXTUAL_KIS-specific ranking policy."""

from __future__ import annotations

from llm.schemas import QueryPlan

from .prediction import Prediction
from .rerank import VLMRankFn, apply_vlm_rerank, rerank_candidates, temporal_nms


def rank_kis(
    query_id: str,
    plan: QueryPlan,
    candidates: list,
    *,
    vlm_rank_fn: VLMRankFn | None = None,
    limit: int = 100,
) -> list[Prediction]:
    """Return Top-k keyframe predictions for a visual retrieval query."""
    ranked = temporal_nms(
        rerank_candidates(candidates, plan, limit=min(200, len(candidates))),
        window_seconds=1.0,
        limit=limit,
    )
    if vlm_rank_fn is not None:
        ranked = apply_vlm_rerank(
            plan.search_description,
            ranked,
            vlm_rank_fn,
            candidate_limit=min(30, len(ranked)),
        )
    return [
        Prediction(
            query_id=query_id,
            task_type=plan.task_type,
            rank=rank,
            video_id=item.candidate.video_id,
            frame_ids=[item.candidate.frame_id],
            score=item.score,
        )
        for rank, item in enumerate(ranked, 1)
    ]
