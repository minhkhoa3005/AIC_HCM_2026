"""TRAKE temporal alignment over retrieval results for ordered events."""

from __future__ import annotations

from collections import defaultdict

from llm.schemas import Candidate, QueryPlan
from llm.task_types import TaskType
from retrieval.models import QueryRetrievalResult

from .prediction import Prediction

_RRF_OFFSET = 60


def align_trake(
    query_id: str,
    plan: QueryPlan,
    results: list[QueryRetrievalResult],
) -> list[Prediction]:
    """Choose one increasing frame sequence per video for the ordered events.

    Query index ``i`` is the evidence query for event ``i + 1``. This invariant
    is enforced by query-plan validation before retrieval is called.
    """

    if not plan.events:
        return []

    results_by_index = _index_results(results, expected_queries=len(plan.events))
    event_hits = [
        _best_candidates_by_video(results_by_index[event_index])
        for event_index in range(len(plan.events))
    ]
    if not event_hits:
        return []

    common_video_ids = set(event_hits[0])
    for hits_by_video in event_hits[1:]:
        common_video_ids.intersection_update(hits_by_video)

    scored_sequences: list[tuple[str, tuple[float, list[int]]]] = []
    for video_id in common_video_ids:
        per_event_candidates = [hits_by_video[video_id] for hits_by_video in event_hits]
        sequence = _best_increasing_sequence(per_event_candidates)
        if sequence is not None:
            scored_sequences.append((video_id, sequence))

    scored_sequences.sort(key=lambda item: item[1][0], reverse=True)
    return [
        Prediction(
            query_id=query_id,
            task_type=TaskType.TRAKE,
            rank=rank,
            video_id=video_id,
            frame_ids=frame_ids,
            score=score,
        )
        for rank, (video_id, (score, frame_ids)) in enumerate(scored_sequences, start=1)
    ]


def _index_results(
    results: list[QueryRetrievalResult], *, expected_queries: int
) -> dict[int, QueryRetrievalResult]:
    indexed: dict[int, QueryRetrievalResult] = {}
    for result in results:
        if not 0 <= result.query_index < expected_queries:
            raise ValueError("TRAKE retrieval returned an unexpected query_index")
        if result.query_index in indexed:
            raise ValueError("TRAKE retrieval returned duplicate query results")
        indexed[result.query_index] = result
    missing = set(range(expected_queries)).difference(indexed)
    if missing:
        raise ValueError(f"TRAKE retrieval is missing results for query indexes: {sorted(missing)}")
    return indexed


def _best_candidates_by_video(
    result: QueryRetrievalResult,
) -> dict[str, list[tuple[int, float]]]:
    """Keep the highest-ranked instance of each frame for one event query."""

    by_video: dict[str, dict[int, float]] = defaultdict(dict)
    for rank, candidate in enumerate(result.candidates, start=1):
        score = _event_score(candidate, rank)
        current = by_video[candidate.video_id].get(candidate.frame_id)
        if current is None or score > current:
            by_video[candidate.video_id][candidate.frame_id] = score
    return {
        video_id: sorted(frames.items())
        for video_id, frames in by_video.items()
    }


def _event_score(candidate: Candidate, rank: int) -> float:
    """Use rank as the primary comparable evidence signal across event queries."""

    clip_tiebreak = min(max(candidate.clip_score, 0.0), 1.0) / 1_000_000
    return 1.0 / (_RRF_OFFSET + rank) + clip_tiebreak


def _best_increasing_sequence(
    per_event_candidates: list[list[tuple[int, float]]],
) -> tuple[float, list[int]] | None:
    """Dynamic program for the maximum-score strictly increasing frame path."""

    states: dict[int, tuple[float, list[int]]] = {}
    for frame_id, score in per_event_candidates[0]:
        current = states.get(frame_id)
        if current is None or score > current[0]:
            states[frame_id] = (score, [frame_id])

    for event_candidates in per_event_candidates[1:]:
        next_states: dict[int, tuple[float, list[int]]] = {}
        for frame_id, score in event_candidates:
            valid_predecessors = [
                state for previous_frame, state in states.items() if previous_frame < frame_id
            ]
            if not valid_predecessors:
                continue
            previous_score, previous_frames = max(valid_predecessors, key=lambda state: state[0])
            candidate_state = (previous_score + score, [*previous_frames, frame_id])
            current = next_states.get(frame_id)
            if current is None or candidate_state[0] > current[0]:
                next_states[frame_id] = candidate_state
        states = next_states
        if not states:
            return None

    return max(states.values(), key=lambda state: state[0], default=None)
