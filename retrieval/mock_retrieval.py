"""Mock retrieval implementation for Day 1 pipeline tests."""

from llm.schemas import Candidate

from .models import QueryRetrievalResult


def mock_search_clip_text(queries: list[str], top_k: int = 100) -> list[QueryRetrievalResult]:
    """Return deterministic per-query fake candidates for offline pipeline tests."""

    if top_k <= 0:
        return []

    return [
        QueryRetrievalResult(
            query_index=query_index,
            query_text=query_text,
            candidates=_mock_candidates(query_text, query_index)[:top_k],
        )
        for query_index, query_text in enumerate(queries)
    ]


def _mock_candidates(query_text: str, query_index: int) -> list[Candidate]:
    score_offset = query_index * 0.01
    return [
        Candidate(
            video_id="L01_V001",
            frame_id=1500,
            keyframe_path="keyframes/L01_V001/01500.jpg",
            clip_score=0.82 - score_offset,
            metadata={"title": "street scene", "query": query_text},
            objects=["person", "car", "door"],
            ocr="",
            asr="",
        ),
        Candidate(
            video_id="L02_V003",
            frame_id=2450,
            keyframe_path="keyframes/L02_V003/02450.jpg",
            clip_score=0.76 - score_offset,
            metadata={"title": "indoor scene", "query": query_text},
            objects=["person", "table", "bag"],
            ocr="OPEN",
            asr="",
        ),
        Candidate(
            video_id="L03_V007",
            frame_id=980,
            keyframe_path="keyframes/L03_V007/00980.jpg",
            clip_score=0.71 - score_offset,
            metadata={"title": "vehicle stop", "query": query_text},
            objects=["bus", "person", "sign"],
            ocr="BUS STOP",
            asr="",
        ),
    ]
