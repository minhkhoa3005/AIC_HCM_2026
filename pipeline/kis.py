"""TEXTUAL_KIS-specific ranking policy."""

from __future__ import annotations

from llm.schemas import QueryPlan
from llm.query_builder import build_rerank_text

from .prediction import Prediction
from .rerank import VLMRankFn, apply_vlm_rerank, rerank_candidates, temporal_nms


def rank_kis(
    query_id: str,
    plan: QueryPlan,
    candidates: list,
    *,
    vlm_rank_fn: VLMRankFn | None = None,
    vlm_candidate_limit: int = 40,
    vlm_weight: float = 0.65,
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
            build_rerank_text(plan),
            ranked,
            vlm_rank_fn,
            candidate_limit=min(vlm_candidate_limit, len(ranked)),
            vlm_weight=vlm_weight,
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
