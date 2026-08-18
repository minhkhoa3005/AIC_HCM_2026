"""Fusion and task-specific reranking of FAISS candidates."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping

from llm.schemas import Candidate, QueryPlan
from llm.task_types import TaskType
from retrieval.models import FusedCandidate, QueryRetrievalResult, RetrievalEvidence

_RRF_OFFSET = 60
DEFAULT_RRF_WEIGHTS = (1.0, 0.85, 0.7)
_METADATA_BOOST_PER_MATCH = 0.001
_MAX_METADATA_BOOST = 0.003

VLMRankFn = Callable[[str, list[Candidate]], Mapping[tuple[str, int], float]]


def fuse_retrieval_results(
    results: list[QueryRetrievalResult],
    plan: QueryPlan,
    query_weights: list[float] | None = None,
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Union candidates by ``(video_id, frame_id)`` and apply weighted RRF."""
    evidence_by_frame: dict[tuple[str, int], list[RetrievalEvidence]] = {}
    for result in results:
        seen_frames: set[tuple[str, int]] = set()
        for rank, candidate in enumerate(result.candidates, start=1):
            key = (candidate.video_id, candidate.frame_id)
            if key in seen_frames:
                continue
            seen_frames.add(key)
            evidence_by_frame.setdefault(key, []).append(
                RetrievalEvidence(
                    query_index=result.query_index,
                    query_text=result.query_text,
                    rank=rank,
                    candidate=candidate,
                )
            )

    ranked = sorted(
        (_fuse_frame(evidence, plan, query_weights) for evidence in evidence_by_frame.values()),
        key=lambda item: (item.score, _best_clip_score(item.evidence)),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


def rerank_candidates(
    candidates: list[FusedCandidate],
    plan: QueryPlan,
    *,
    task_type: TaskType | None = None,
    limit: int | None = None,
) -> list[FusedCandidate]:
    """Rerank the FAISS/RRF shortlist using task-aware textual evidence.

    KIS favors visual similarity and explicit object/action evidence. QA gives
    OCR/ASR and question-term evidence more weight because those fields help
    choose the frame that can actually answer the question.
    """
    resolved_task = task_type or plan.task_type
    scored = []
    for item in candidates:
        candidate = item.candidate
        evidence_text = _candidate_text(candidate)
        term_score = _term_coverage(plan, evidence_text)
        clip_score = _bounded_clip_score(candidate.clip_score)
        support_score = min(len(item.evidence), 3) / 3.0
        if resolved_task == TaskType.QA:
            score = item.score + 0.10 * clip_score + 0.07 * term_score + 0.04 * support_score
        else:
            score = item.score + 0.14 * clip_score + 0.05 * term_score + 0.03 * support_score
        scored.append(item.model_copy(update={"score": score}))
    ranked = sorted(scored, key=lambda item: (item.score, _best_clip_score(item.evidence)), reverse=True)
    return ranked[:limit] if limit is not None else ranked


def temporal_nms(
    candidates: list[FusedCandidate],
    *,
    window_seconds: float = 1.0,
    limit: int = 100,
) -> list[FusedCandidate]:
    """Suppress near-duplicate frames from the same video.

    Candidates are already score-ranked. The first candidate in a temporal
    neighborhood therefore wins, while different videos remain independent.
    ``pts_time``/``timestamp`` is preferred; frame_id is only a fallback.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    selected: list[FusedCandidate] = []
    selected_times: dict[str, list[float]] = {}
    for item in candidates:
        video_id = item.candidate.video_id
        timestamp = _candidate_time(item.candidate)
        prior = selected_times.setdefault(video_id, [])
        if any(abs(timestamp - previous) <= window_seconds for previous in prior):
            continue
        selected.append(item)
        prior.append(timestamp)
        if len(selected) >= limit:
            break
    return selected


def apply_vlm_rerank(
    query: str,
    candidates: list[FusedCandidate],
    vlm_rank_fn: VLMRankFn,
    *,
    candidate_limit: int = 30,
    vlm_weight: float = 0.5,
) -> list[FusedCandidate]:
    """Apply one batched VLM call to a compact evidence shortlist."""
    shortlist = candidates[:candidate_limit]
    scores = vlm_rank_fn(query, [item.candidate for item in shortlist])
    reranked = []
    for item in shortlist:
        key = (item.candidate.video_id, item.candidate.frame_id)
        vlm_score = max(0.0, min(1.0, float(scores.get(key, 0.0))))
        reranked.append(item.model_copy(update={"score": (1 - vlm_weight) * item.score + vlm_weight * vlm_score}))
    reranked.sort(key=lambda item: item.score, reverse=True)
    remainder = candidates[candidate_limit:]
    return reranked + remainder


def select_qa_evidence(
    candidates: list[FusedCandidate],
    *,
    limit: int = 8,
    max_per_video: int = 2,
) -> list[FusedCandidate]:
    """Select compact, diverse evidence for VLM calls.

    This prevents spending a VLM call on 100 near-duplicate frames from one
    video while preserving more than one frame when temporal evidence helps.
    """
    selected: list[FusedCandidate] = []
    video_counts: dict[str, int] = {}
    for candidate in candidates:
        video_id = candidate.candidate.video_id
        if video_counts.get(video_id, 0) >= max_per_video:
            continue
        if any(_same_evidence(candidate, item) for item in selected):
            continue
        selected.append(candidate)
        video_counts[video_id] = video_counts.get(video_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _same_evidence(left: FusedCandidate, right: FusedCandidate) -> bool:
    return (
        left.candidate.video_id == right.candidate.video_id
        and abs(left.candidate.frame_id - right.candidate.frame_id) <= 1
    )


def _fuse_frame(
    evidence: list[RetrievalEvidence],
    plan: QueryPlan,
    query_weights: list[float] | None,
) -> FusedCandidate:
    representative = max(evidence, key=lambda item: item.candidate.clip_score).candidate
    rrf_score = sum(
        _query_weight(item.query_index, query_weights) / (_RRF_OFFSET + item.rank)
        for item in evidence
    )
    return FusedCandidate(
        candidate=representative,
        score=rrf_score + _metadata_boost(representative, plan),
        evidence=evidence,
    )


def _query_weight(query_index: int, query_weights: list[float] | None) -> float:
    if query_weights and 0 <= query_index < len(query_weights):
        return max(float(query_weights[query_index]), 0.0)
    return DEFAULT_RRF_WEIGHTS[query_index] if query_index < len(DEFAULT_RRF_WEIGHTS) else 0.7


def _best_clip_score(evidence: list[RetrievalEvidence]) -> float:
    return max((item.candidate.clip_score for item in evidence), default=0.0)


def _bounded_clip_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _candidate_text(candidate: Candidate) -> str:
    metadata = [str(value) for key, value in candidate.metadata.items() if key not in {"query", "query_text"}]
    return " ".join([*candidate.objects, candidate.ocr, candidate.asr, *metadata]).casefold()


def _candidate_time(candidate: Candidate) -> float:
    for key in ("pts_time", "timestamp", "start", "frame_time"):
        try:
            if key in candidate.metadata:
                return float(candidate.metadata[key])
        except (TypeError, ValueError):
            continue
    return float(candidate.frame_id)


def _term_coverage(plan: QueryPlan, text: str) -> float:
    terms = [*plan.objects, *plan.actions, *plan.metadata_keywords]
    if plan.question:
        terms.extend(re.findall(r"[\w]+", plan.question.casefold()))
    terms = [term.casefold().strip() for term in terms if term and term.strip()]
    if not terms:
        return 0.0
    return sum(term in text for term in terms) / len(terms)


def _metadata_boost(candidate: Candidate, plan: QueryPlan) -> float:
    matches = _term_coverage(plan, _candidate_text(candidate)) * len(
        [*plan.objects, *plan.actions, *plan.metadata_keywords]
    )
    return min(matches * _METADATA_BOOST_PER_MATCH, _MAX_METADATA_BOOST)
