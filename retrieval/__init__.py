"""Retrieval interfaces and mocks for the video query pipeline."""

from .contract import search_clip_text
from .mock_retrieval import mock_search_clip_text
from .models import FusedCandidate, QueryRetrievalResult, RetrievalEvidence

__all__ = [
    "FusedCandidate",
    "QueryRetrievalResult",
    "RetrievalEvidence",
    "mock_search_clip_text",
    "search_clip_text",
]
