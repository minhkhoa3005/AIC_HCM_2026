"""TRAKE ordered-event retrieval and temporal path alignment."""

from __future__ import annotations

from collections import defaultdict

from llm.schemas import Candidate, QueryPlan
from llm.task_types import TaskType
from retrieval.models import QueryRetrievalResult

from .prediction import Prediction

_RRF_OFFSET = 60


def align_trake(query_id: str, plan: QueryPlan, results: list[QueryRetrievalResult]) -> list[Prediction]:
    """Select the highest-scoring strictly increasing frame path per video."""
    if len(results) != len(plan.events):
        raise ValueError("TRAKE needs one retrieval result for each event")
    indexed = _index_results(results, len(plan.events))
    event_hits = [_best_candidates_by_video(indexed[index]) for index in range(len(plan.events))]
    common_videos = set(event_hits[0])
    for hits in event_hits[1:]:
        common_videos.intersection_update(hits)

    scored: list[tuple[str, float, list[int]]] = []
    for video_id in common_videos:
        path = _best_increasing_sequence([hits[video_id] for hits in event_hits])
        if path is not None:
            score, frame_ids = path
            scored.append((video_id, score, frame_ids))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        Prediction(
            query_id=query_id,
            task_type=TaskType.TRAKE,
            rank=rank,
            video_id=video_id,
            frame_ids=frame_ids,
            score=score,
        )
        for rank, (video_id, score, frame_ids) in enumerate(scored[:100], start=1)
    ]


def _index_results(results: list[QueryRetrievalResult], expected: int) -> dict[int, QueryRetrievalResult]:
    indexed = {result.query_index: result for result in results}
    if len(indexed) != expected or set(indexed) != set(range(expected)):
        raise ValueError("TRAKE retrieval results must contain each event query exactly once")
    return indexed


def _best_candidates_by_video(result: QueryRetrievalResult) -> dict[str, list[tuple[int, float]]]:
    by_video: dict[str, dict[int, float]] = defaultdict(dict)
    for rank, candidate in enumerate(result.candidates, start=1):
        score = 1.0 / (_RRF_OFFSET + rank) + max(candidate.clip_score, 0.0) / 1_000_000
        current = by_video[candidate.video_id].get(candidate.frame_id)
        if current is None or score > current:
            by_video[candidate.video_id][candidate.frame_id] = score
    return {video_id: sorted(frames.items()) for video_id, frames in by_video.items()}


def _best_increasing_sequence(candidates: list[list[tuple[int, float]]]) -> tuple[float, list[int]] | None:
    states: dict[int, tuple[float, list[int]]] = {
        frame_id: (score, [frame_id]) for frame_id, score in candidates[0]
    }
    for event_candidates in candidates[1:]:
        next_states: dict[int, tuple[float, list[int]]] = {}
        for frame_id, score in event_candidates:
            predecessors = [state for previous, state in states.items() if previous < frame_id]
            if not predecessors:
                continue
            previous_score, previous_frames = max(predecessors, key=lambda state: state[0])
            candidate_state = (previous_score + score, [*previous_frames, frame_id])
            if frame_id not in next_states or candidate_state[0] > next_states[frame_id][0]:
                next_states[frame_id] = candidate_state
        states = next_states
        if not states:
            return None
    return max(states.values(), key=lambda state: state[0], default=None)
