"""Retrieval interface to be implemented by CLIP/FAISS integration."""

from .models import QueryRetrievalResult


def search_clip_text(queries: list[str], top_k: int = 100) -> list[QueryRetrievalResult]:
    """Return ranked candidates for every input query.

    ``top_k`` is applied per query. Implementations must preserve ``query_index``
    and ``query_text`` so fusion and TRAKE can use retrieval provenance.
    """

    raise NotImplementedError("CLIP/FAISS retrieval is not implemented in Day 1")
