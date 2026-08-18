"""Retrieval interface for the external CLIP/FAISS backend."""

from .models import QueryRetrievalResult


def search_clip_text(queries: list[str], top_k: int = 100) -> list[QueryRetrievalResult]:
    """Return ranked candidates for every input query.

    ``top_k`` is applied per query. Implementations must preserve ``query_index``
    and ``query_text`` so fusion can use retrieval provenance.
    """

    raise NotImplementedError(
        "Inject a CLIP/FAISS search function, for example LocalBundleRetriever.search_many"
    )


def validate_retrieval_results(
    results: list[QueryRetrievalResult],
    queries: list[str],
    top_k: int,
) -> list[QueryRetrievalResult]:
    """Validate adapter output before RRF consumes retrieval results."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(results) != len(queries):
        raise ValueError(f"Expected {len(queries)} retrieval results, got {len(results)}")
    for expected_index, result in enumerate(results):
        if result.query_index != expected_index or result.query_text != queries[expected_index]:
            raise ValueError("retrieval results must preserve query order and query_index")
        if len(result.candidates) > top_k:
            raise ValueError("retrieval adapter returned more than top_k candidates")
        seen: set[tuple[str, int]] = set()
        for candidate in result.candidates:
            if not candidate.video_id.strip() or candidate.frame_id < 0:
                raise ValueError("candidate must contain video_id and non-negative frame_id")
            if float(candidate.metadata.get("pts_time", 0.0)) < 0:
                raise ValueError("candidate pts_time must be non-negative")
            key = (candidate.video_id, candidate.frame_id)
            if key in seen:
                raise ValueError(f"duplicate candidate in retrieval result: {key}")
            seen.add(key)
    return results
