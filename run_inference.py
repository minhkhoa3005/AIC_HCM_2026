"""Run the independent LLM KIS/QA pipeline against a local CLIP bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from adapters import ClipTextEncoder
from llm.config import load_project_env
from pipeline import run_batch
from pipeline.checkpoint import QueryCheckpoint
from retrieval.bundle import LocalBundleRetriever
from submission.csv_exporter import export_csv


def _read_queries(path: Path) -> list[tuple[str, str] | tuple[str, str, str]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append((payload["query_id"], payload["query"], payload["task_type"]) if payload.get("task_type") else (payload["query_id"], payload["query"]))
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if not row.get("query_id") or not row.get("query"):
                raise ValueError("Each query row needs query_id and query")
            rows.append((row["query_id"], row["query"], row["task_type"]) if row.get("task_type") else (row["query_id"], row["query"]))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent AIC KIS/QA inference")
    parser.add_argument("--queries", required=True, help="CSV or JSONL with query_id, query, optional task_type")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--checkpoint", default="outputs/checkpoint.jsonl")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--with-vlm", action="store_true", help="Enable Gemini VLM reranking and QA")
    args = parser.parse_args()

    load_project_env()
    encoder = ClipTextEncoder.from_env()
    retriever = LocalBundleRetriever.from_env(encoder.encode_one)
    checkpoint = QueryCheckpoint(args.checkpoint)
    vlm = None
    if args.with_vlm:
        from llm.vlm import GeminiVLM

        vlm = GeminiVLM()

    new_predictions = run_batch(
        _read_queries(Path(args.queries)),
        top_k=args.top_k,
        search_fn=retriever.search_many,
        checkpoint=checkpoint,
        text_encoder=encoder.encode_many,
        vlm_rank_fn=vlm.rank if vlm else None,
        vqa_fn=vlm.answer if vlm else None,
    )
    all_predictions = checkpoint.load_predictions()
    if not all_predictions:
        all_predictions = new_predictions
    export_csv(all_predictions, args.output)
    print(f"Wrote {len(all_predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
