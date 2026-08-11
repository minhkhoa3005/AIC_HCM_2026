"""Retrieval interfaces and mocks for the video query pipeline."""

from .contract import search_clip_text
from .mock_retrieval import mock_search_clip_text

__all__ = ["mock_search_clip_text", "search_clip_text"]
