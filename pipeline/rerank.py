"""Candidate merge, dedupe, and lightweight reranking."""

from __future__ import annotations

from llm.schemas import Candidate, QueryPlan


def rerank_candidates(candidates: list[Candidate], plan: QueryPlan) -> list[Candidate]:
    """Dedupe by video/frame and sort by score with simple metadata boosts."""

    best: dict[tuple[str, int], Candidate] = {}
    for candidate in candidates:
        key = (candidate.video_id, candidate.frame_id)
        current = best.get(key)
        if current is None or _score(candidate, plan) > _score(current, plan):
            best[key] = candidate
    return sorted(best.values(), key=lambda item: _score(item, plan), reverse=True)


def _score(candidate: Candidate, plan: QueryPlan) -> float:
    score = candidate.clip_score
    haystack = " ".join(
        [
            " ".join(candidate.objects),
            candidate.ocr,
            candidate.asr,
            " ".join(str(value) for value in candidate.metadata.values()),
        ]
    ).lower()
    for keyword in [*plan.objects, *plan.actions, *plan.metadata_keywords]:
        if keyword and keyword.lower() in haystack:
            score += 0.01
    return score
