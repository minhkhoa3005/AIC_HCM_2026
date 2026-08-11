"""Retrieval interface to be implemented by CLIP/FAISS integration."""

from llm.schemas import Candidate


def search_clip_text(queries: list[str], top_k: int = 100) -> list[Candidate]:
    """Search keyframes/clips by text and return normalized candidates."""

    raise NotImplementedError("CLIP/FAISS retrieval is not implemented in Day 1")
