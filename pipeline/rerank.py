"""Fuse per-query retrieval results into evidence-ranked frames."""

from __future__ import annotations

from llm.schemas import Candidate, QueryPlan
from retrieval.models import FusedCandidate, QueryRetrievalResult, RetrievalEvidence

_RRF_OFFSET = 60
_METADATA_BOOST_PER_MATCH = 0.001
_MAX_METADATA_BOOST = 0.003


def fuse_retrieval_results(
    results: list[QueryRetrievalResult], plan: QueryPlan
) -> list[FusedCandidate]:
    """Fuse independent query rankings with reciprocal-rank fusion.

    Raw CLIP scores from different text queries are not assumed comparable.
    They are used only as a deterministic tie-breaker after RRF.
    """

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

    fused = [
        _fuse_frame(evidence, plan)
        for evidence in evidence_by_frame.values()
    ]
    return sorted(
        fused,
        key=lambda item: (item.score, _best_clip_score(item.evidence)),
        reverse=True,
    )


def _fuse_frame(evidence: list[RetrievalEvidence], plan: QueryPlan) -> FusedCandidate:
    representative = max(evidence, key=lambda item: item.candidate.clip_score).candidate
    rrf_score = sum(1.0 / (_RRF_OFFSET + item.rank) for item in evidence)
    return FusedCandidate(
        candidate=representative,
        score=rrf_score + _metadata_boost(representative, plan),
        evidence=evidence,
    )


def _best_clip_score(evidence: list[RetrievalEvidence]) -> float:
    return max((item.candidate.clip_score for item in evidence), default=0.0)


def _metadata_boost(candidate: Candidate, plan: QueryPlan) -> float:
    """Apply a small boost only for explicit known metadata evidence."""

    metadata_values = [
        str(value)
        for key, value in candidate.metadata.items()
        if key not in {"query", "query_text"}
    ]
    haystack = " ".join(
        [" ".join(candidate.objects), candidate.ocr, candidate.asr, *metadata_values]
    ).lower()
    known_terms = [*plan.objects, *plan.actions, *plan.metadata_keywords]
    matches = sum(
        1 for term in known_terms if term and term.lower() in haystack
    )
    return min(matches * _METADATA_BOOST_PER_MATCH, _MAX_METADATA_BOOST)
