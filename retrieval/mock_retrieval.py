"""Mock retrieval implementation for Day 1 pipeline tests."""

from llm.schemas import Candidate


def mock_search_clip_text(queries: list[str], top_k: int = 100) -> list[Candidate]:
    """Return deterministic fake candidates that satisfy the retrieval schema."""

    if top_k <= 0:
        return []

    query_text = " | ".join(queries) if queries else "empty query"
    candidates = [
        Candidate(
            video_id="L01_V001",
            frame_id=1500,
            keyframe_path="keyframes/L01_V001/01500.jpg",
            clip_score=0.82,
            metadata={"title": "street scene", "query": query_text},
            objects=["person", "car", "door"],
            ocr="",
            asr="",
        ),
        Candidate(
            video_id="L02_V003",
            frame_id=2450,
            keyframe_path="keyframes/L02_V003/02450.jpg",
            clip_score=0.76,
            metadata={"title": "indoor scene", "query": query_text},
            objects=["person", "table", "bag"],
            ocr="OPEN",
            asr="",
        ),
        Candidate(
            video_id="L03_V007",
            frame_id=980,
            keyframe_path="keyframes/L03_V007/00980.jpg",
            clip_score=0.71,
            metadata={"title": "vehicle stop", "query": query_text},
            objects=["bus", "person", "sign"],
            ocr="BUS STOP",
            asr="",
        ),
    ]
    return candidates[:top_k]
