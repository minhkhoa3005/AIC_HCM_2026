"""Vision adapters for KIS and QA."""

from .gemini import GeminiVLM
from .local import LocalVLM

__all__ = ["GeminiVLM", "LocalVLM"]
