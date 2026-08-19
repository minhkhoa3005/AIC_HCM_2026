"""JSONL checkpointing for batch query runs."""

from __future__ import annotations

import json
from pathlib import Path

from .prediction import Prediction


class QueryCheckpoint:
    """Persist completed query predictions and allow resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_query_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        done = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            done.add(payload["query_id"])
        return done

    def append(self, query_id: str, predictions: list[Prediction]) -> None:
        payload = {
            "query_id": query_id,
            "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_predictions(self) -> list[Prediction]:
        """Load all predictions so a resumed run can export a complete CSV."""
        if not self.path.exists():
            return []
        predictions: list[Prediction] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            predictions.extend(Prediction.model_validate(item) for item in payload.get("predictions", []))
        return predictions
