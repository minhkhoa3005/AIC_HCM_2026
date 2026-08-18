"""Gemini Vision implementation for the pipeline's VLM adapter hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.config import LLMConfig, get_llm_config
from llm.llm_client import generate_vlm_json
from llm.schemas import Candidate


class GeminiVLM:
    """Cached Gemini VLM adapter for batch reranking and QA answers."""

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None):
        self.config = config or get_llm_config()
        self.client = client
        self._rank_cache: dict[tuple[str, tuple[tuple[str, int], ...]], dict[tuple[str, int], float]] = {}
        self._answer_cache: dict[tuple[str, str, int], str] = {}

    def rank(self, query: str, candidates: list[Candidate]) -> dict[tuple[str, int], float]:
        """Score a candidate batch with one Gemini request."""
        key = (query, tuple((item.video_id, item.frame_id) for item in candidates))
        if key in self._rank_cache:
            return self._rank_cache[key]
        usable = [item for item in candidates if Path(item.keyframe_path).exists()]
        if not usable:
            return {}
        prompt = _rank_prompt(query, usable)
        payload = generate_vlm_json(
            prompt,
            [item.keyframe_path for item in usable],
            self.config,
            client=self.client,
        )
        scores = _parse_rank_response(payload, usable)
        self._rank_cache[key] = scores
        return scores

    def answer(self, question: str, candidate: Candidate) -> str:
        """Answer a QA question from one evidence frame with cached results."""
        key = (question, candidate.video_id, candidate.frame_id)
        if key in self._answer_cache:
            return self._answer_cache[key]
        if not Path(candidate.keyframe_path).exists():
            return ""
        prompt = (
            "You are a visual question answering judge. Answer the question using only "
            "the visible evidence in the image. Return JSON only: "
            '{"answer": "short answer"}. Do not explain.\n'
            f"Question: {question}"
        )
        payload = generate_vlm_json(
            prompt,
            [candidate.keyframe_path],
            self.config,
            client=self.client,
        )
        answer = str(payload.get("answer", "")).strip()
        self._answer_cache[key] = answer
        return answer


def _rank_prompt(query: str, candidates: list[Candidate]) -> str:
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
        "only visible evidence. Return JSON only with an items array. Each item must "
        "contain video_id, frame_id, relevant (true/false), and confidence from 0 to 1. "
        "Do not invent colors, identity, actions, or text.\n"
        f"Query: {query}\nCandidates: {json.dumps(rows, ensure_ascii=False)}"
    )


def _parse_rank_response(payload: dict[str, Any], candidates: list[Candidate]) -> dict[tuple[str, int], float]:
    by_key = {(item.video_id, item.frame_id): item for item in candidates}
    scores: dict[tuple[str, int], float] = {}
    for row in payload.get("items", []):
        if not isinstance(row, dict):
            continue
        key = (str(row.get("video_id", "")), int(row.get("frame_id", -1)))
        if key not in by_key:
            continue
        try:
            score = float(row.get("confidence", 1.0 if row.get("relevant") else 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores[key] = max(0.0, min(1.0, score))
    return scores
