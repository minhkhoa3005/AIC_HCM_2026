"""Build three retrieval queries from an anchor and ten LLM expansions."""

from __future__ import annotations

import math
import re
import hashlib
from collections.abc import Callable, Sequence

from .schemas import QueryPlan

TextEncoder = Callable[[list[str]], Sequence[Sequence[float]]]
_PROHIBITED_PURPOSE_PHRASES = (
    "showing clothing details", "showing their clothing details",
    "of a specific color", "showing if", "to identify", "for ocr",
)


def build_clip_queries(plan: QueryPlan, text_encoder: TextEncoder | None = None) -> list[str]:
    """Return exactly three queries: anchor plus two diverse expansions."""
    anchor = _validate_english(plan.anchor, "anchor")
    expansions = [_validate_english(item, "expansion") for item in plan.expansions]
    if len(expansions) != 10:
        raise ValueError("QueryPlan.expansions must contain exactly 10 items")
    return [anchor, *select_expansions(anchor, expansions, text_encoder=text_encoder)]


def build_trake_queries(plan: QueryPlan) -> list[str]:
    """Return one literal English visual query for each ordered TRAKE event."""
    if plan.task_type.value != "TRAKE":
        raise ValueError("build_trake_queries requires a TRAKE plan")
    if len(plan.events) < 2 or len(plan.event_queries) != len(plan.events):
        raise ValueError("TRAKE events and event_queries must have the same length >= 2")
    return [_validate_english(query, f"event_queries[{index}]") for index, query in enumerate(plan.event_queries)]


def select_expansions(anchor: str, expansions: list[str], *, text_encoder: TextEncoder | None = None, count: int = 2, mmr_lambda: float = 0.72) -> list[str]:
    """Select relevant but diverse expansions using CLIP-style MMR/FQS."""
    if not 0 < count <= len(expansions):
        raise ValueError("count must be between 1 and the number of expansions")
    # One batched encoder call avoids ten separate tokenizer/model launches.
    vectors = list(text_encoder([anchor, *expansions])) if text_encoder else []
    if len(vectors) != len(expansions) + 1:
        vectors = [_lexical_vector(anchor), *(_lexical_vector(x) for x in expansions)]
    selected: list[int] = []
    remaining = set(range(len(expansions)))
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda index: (
                mmr_lambda * _cosine(vectors[0], vectors[index + 1])
                - (1 - mmr_lambda)
                * max(
                    (_cosine(vectors[index + 1], vectors[j + 1]) for j in selected),
                    default=0.0,
                ),
                -index,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return [expansions[index] for index in selected]


def build_rerank_text(plan: QueryPlan) -> str:
    pieces = [plan.search_description]
    if plan.question:
        pieces.append(f"question: {plan.question}")
    if plan.negative_constraints:
        pieces.append("avoid: " + ", ".join(plan.negative_constraints))
    return " | ".join(piece for piece in pieces if piece)


def _validate_english(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be English-only ASCII") from exc
    if any(phrase in value.lower() for phrase in _PROHIBITED_PURPOSE_PHRASES):
        raise ValueError(f"{label} contains an inspection-purpose phrase")
    return value


def _lexical_vector(value: str) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    vector = [0.0] * 256
    for token in tokens:
        digest = hashlib.sha256(token.encode("ascii")).digest()
        vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1.0
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    size = max(len(left), len(right))
    a = list(left) + [0.0] * (size - len(left))
    b = list(right) + [0.0] * (size - len(right))
    denominator = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0
