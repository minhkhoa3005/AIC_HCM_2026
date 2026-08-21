"""Retrieval interfaces for the video query pipeline."""

from .contract import search_clip_text, validate_retrieval_results
from .models import FusedCandidate, QueryRetrievalResult, RetrievalEvidence
from .bundle import BundleError, LocalBundleRetriever

__all__ = [
    "FusedCandidate",
    "QueryRetrievalResult",
    "RetrievalEvidence",
    "search_clip_text",
    "validate_retrieval_results",
    "BundleError",
    "LocalBundleRetriever",
]
