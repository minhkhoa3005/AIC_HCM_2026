"""Retrieval interfaces and mocks for the video query pipeline."""

from .contract import search_clip_text, validate_retrieval_results
from .mock_retrieval import mock_search_clip_text
from .models import FusedCandidate, QueryRetrievalResult, RetrievalEvidence
from .bundle import BundleError, LocalBundleRetriever

__all__ = [
    "FusedCandidate",
    "QueryRetrievalResult",
    "RetrievalEvidence",
    "mock_search_clip_text",
    "search_clip_text",
    "validate_retrieval_results",
    "BundleError",
    "LocalBundleRetriever",
]
