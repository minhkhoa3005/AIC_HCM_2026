"""QA task handling over retrieved visual evidence."""

from __future__ import annotations

from collections.abc import Callable

from llm.schemas import Candidate, QueryPlan
from llm.task_types import TaskType
from retrieval.models import FusedCandidate

from .prediction import Prediction
from .rerank import VLMRankFn, apply_vlm_rerank, rerank_candidates, select_qa_evidence, temporal_nms

VQAFn = Callable[[str, Candidate], str]


def answer_qa(
    query_id: str,
    plan: QueryPlan,
    candidates: list[FusedCandidate],
    vqa_fn: VQAFn | None = None,
    vlm_rank_fn: VLMRankFn | None = None,
    vlm_candidate_limit: int = 40,
    vlm_weight: float = 0.65,
    limit: int = 100,
) -> list[Prediction]:
    """Create ranked QA predictions, using a VLM adapter when provided."""

    candidates = temporal_nms(
        rerank_candidates(candidates, plan, task_type=TaskType.QA, limit=min(200, len(candidates))),
        window_seconds=1.0,
        limit=limit,
    )
    if not candidates:
        return []
    if vlm_rank_fn is not None:
        candidates = apply_vlm_rerank(
            plan.question or plan.search_description,
            candidates,
            vlm_rank_fn,
            candidate_limit=min(vlm_candidate_limit, len(candidates)),
            vlm_weight=vlm_weight,
        )
    evidence = select_qa_evidence(candidates, limit=min(8, limit))
    answers: dict[tuple[str, int], str] = {}
    for item in evidence:
        candidate = item.candidate
        key = (candidate.video_id, candidate.frame_id)
        answers[key] = (
            vqa_fn(plan.question or "", candidate)
            if vqa_fn
            else _fallback_answer(candidate)
        )
    predictions = []
    for rank, fused_candidate in enumerate(candidates[:limit], start=1):
        candidate = fused_candidate.candidate
        answer = answers.get(
            (candidate.video_id, candidate.frame_id),
            _fallback_answer(candidate),
        )
        predictions.append(Prediction(
            query_id=query_id,
            task_type=TaskType.QA,
            rank=rank,
            video_id=candidate.video_id,
            frame_ids=[candidate.frame_id],
            answer=answer,
            score=fused_candidate.score,
        ))
    return predictions


def _fallback_answer(candidate: Candidate) -> str:
    if candidate.ocr:
        return candidate.ocr
    if candidate.asr:
        return candidate.asr
    return ""
