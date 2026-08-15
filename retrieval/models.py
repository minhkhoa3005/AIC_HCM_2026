"""Typed retrieval results that preserve the query which produced each hit."""

from pydantic import BaseModel, Field

from llm.schemas import Candidate


class QueryRetrievalResult(BaseModel):
    """Ranked keyframe candidates returned for one CLIP text query."""

    query_index: int = Field(ge=0)
    query_text: str
    candidates: list[Candidate] = Field(default_factory=list)


class RetrievalEvidence(BaseModel):
    """One candidate together with the query and rank that retrieved it."""

    query_index: int = Field(ge=0)
    query_text: str
    rank: int = Field(ge=1)
    candidate: Candidate


class FusedCandidate(BaseModel):
    """A frame supported by one or more independent retrieval queries."""

    candidate: Candidate
    score: float
    evidence: list[RetrievalEvidence] = Field(default_factory=list)
