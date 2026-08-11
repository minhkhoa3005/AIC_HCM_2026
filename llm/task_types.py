"""Task type enum shared by LLM parsing, routing, and retrieval."""

from enum import Enum


class TaskType(str, Enum):
    TEXTUAL_KIS = "TEXTUAL_KIS"
    QA = "QA"
    TRAKE = "TRAKE"
