"""Prompt and response helpers shared by the local VLM tasks."""

from __future__ import annotations

import json
from typing import Any

from llm.schemas import Candidate


def rank_prompt(query: str, candidates: list[Candidate]) -> str:
    rows = [
        {
            "index": index,
            "video_id": candidate.video_id,
            "frame_id": candidate.frame_id,
        }
        for index, candidate in enumerate(candidates)
    ]
    return (
        "You are a strict visual retrieval reranker. The images are provided in the "
        "same order as the candidate list. Judge each image against the query using "
        "only visible evidence. Return exactly one item for every candidate index, "
        "including irrelevant items. Return JSON only with an items array. Each item "
        "must contain index, video_id, frame_id, relevant (true/false), and confidence "
        "from 0 to 1. Keep the supplied IDs unchanged. Do not invent colors, identity, "
        "actions, or text.\n"
        f"Query: {query}\nCandidates: {json.dumps(rows, ensure_ascii=False)}"
    )


def parse_rank_response(
    payload: dict[str, Any], candidates: list[Candidate]
) -> dict[tuple[str, int], float]:
    by_key = {(item.video_id, item.frame_id): item for item in candidates}
    scores: dict[tuple[str, int], float] = {}
    for row in payload.get("items", []):
        if not isinstance(row, dict):
            continue
        try:
            row_index = int(row.get("index", -1))
        except (TypeError, ValueError):
            row_index = -1
        try:
            key = (str(row.get("video_id", "")), int(row.get("frame_id", -1)))
        except (TypeError, ValueError):
            key = ("", -1)
        if key not in by_key and 0 <= row_index < len(candidates):
            candidate = candidates[row_index]
            key = (candidate.video_id, candidate.frame_id)
        if key not in by_key:
            continue
        try:
            score = float(row.get("confidence", 1.0 if row.get("relevant") else 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores[key] = max(0.0, min(1.0, score))
    return scores
