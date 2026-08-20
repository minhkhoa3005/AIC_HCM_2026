"""Task type enum shared by LLM parsing, routing, and retrieval."""

from enum import Enum
from pathlib import Path


class TaskType(str, Enum):
    TEXTUAL_KIS = "TEXTUAL_KIS"
    QA = "QA"
    TRAKE = "TRAKE"


def normalize_task_type(value: TaskType | str) -> TaskType:
    """Normalize task labels from filenames, CSV fields, or callers."""

    if isinstance(value, TaskType):
        return value

    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "KIS": TaskType.TEXTUAL_KIS,
        "TEXTUAL_KIS": TaskType.TEXTUAL_KIS,
        "TEXTUALKIS": TaskType.TEXTUAL_KIS,
        "QA": TaskType.QA,
        "Q_A": TaskType.QA,
        "QUESTION_ANSWERING": TaskType.QA,
        "TRAKE": TaskType.TRAKE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        expected = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported task type {value!r}. Expected one of: {expected}") from exc


def infer_task_type_from_name(name: str) -> TaskType:
    """Infer KIS/QA from an organizer filename or query ID."""

    stem = Path(str(name)).stem.upper().replace("-", "_").replace(" ", "_")
    tokens = [token for token in stem.split("_") if token]

    if "QA" in tokens or stem.startswith("QA") or stem.endswith("QA"):
        return TaskType.QA
    if "KIS" in tokens or "TEXTUAL_KIS" in stem or stem.startswith("KIS") or stem.endswith("KIS"):
        return TaskType.TEXTUAL_KIS
    if "TRAKE" in tokens or stem.startswith("TRAKE") or stem.endswith("TRAKE"):
        return TaskType.TRAKE

    raise ValueError(
        f"Cannot infer task type from {name!r}. Put KIS or QA in the filename/query_id, "
        "or pass task_type explicitly."
    )
