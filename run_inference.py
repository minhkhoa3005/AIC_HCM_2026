"""Run the independent LLM KIS/QA pipeline against a local CLIP bundle."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from adapters import ClipTextEncoder
from llm.config import load_project_env
from pipeline import run_batch
from pipeline.checkpoint import QueryCheckpoint
from retrieval.bundle import LocalBundleRetriever
from submission.btc_exporter import export_btc_csv
from submission.csv_exporter import export_csv
from llm.task_types import infer_task_type_from_name


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


def _read_query_files(input_dir: Path) -> list[tuple[str, str, str]]:
    """Read BTC query package files and derive task type from each filename."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Query directory not found: {input_dir}")
    rows: list[tuple[str, str, str]] = []
    paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".txt", ".csv"}
        and re.search(r"-(kis|qa|trake)$", path.stem, flags=re.IGNORECASE)
    )
    for path in paths:
        query = path.read_text(encoding="utf-8-sig").strip()
        if not query:
            raise ValueError(f"Query file is empty: {path}")
        task_type = infer_task_type_from_name(path.stem)
        rows.append((path.stem, query, task_type.value))
    if not rows:
        raise ValueError(f"No query-*-kis/qa/trake.txt or .csv files found in {input_dir}")
    return rows


def _write_per_query_results(predictions, query_rows, output_dir: Path) -> None:
    """Write one CSV per query using the same stem as the input file."""
    by_query: dict[str, list] = {}
    for prediction in predictions:
        by_query.setdefault(prediction.query_id, []).append(prediction)
    output_dir.mkdir(parents=True, exist_ok=True)
    for query_id, _query, raw_task_type in query_rows:
        query_predictions = by_query.get(query_id, [])
        # The input stem already contains the BTC task suffix, e.g.
        # query-1-kis.txt -> query-1-kis.csv.
        output_path = output_dir / f"{query_id}.csv"
        export_btc_csv(query_predictions, output_path, task_type=raw_task_type)
        print(f"Wrote {len(query_predictions)} predictions to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent AIC KIS/QA inference")
    parser.add_argument("--queries", help="CSV or JSONL with query_id, query, optional task_type")
    parser.add_argument("--input-dir", help="BTC query package directory containing query-*-kis/qa/trake.txt files")
    parser.add_argument("--output", default="outputs/submission.csv")
    parser.add_argument("--output-dir", default="outputs/results", help="Per-query result directory for --input-dir")
    parser.add_argument("--checkpoint", default="outputs/checkpoint.jsonl")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--with-vlm", action="store_true", help="Enable Gemini VLM reranking and QA")
    args = parser.parse_args()

    if bool(args.queries) == bool(args.input_dir):
        parser.error("Provide exactly one of --queries or --input-dir")

    load_project_env()
    encoder = ClipTextEncoder.from_env()
    retriever = LocalBundleRetriever.from_env(encoder.encode_one)
    checkpoint = QueryCheckpoint(args.checkpoint)
    vlm = None
    if args.with_vlm:
        from llm.vlm import GeminiVLM

        vlm = GeminiVLM()

    query_rows = _read_query_files(Path(args.input_dir)) if args.input_dir else _read_queries(Path(args.queries))
    new_predictions = run_batch(
        query_rows,
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
    if args.input_dir:
        _write_per_query_results(all_predictions, query_rows, Path(args.output_dir))
    else:
        export_csv(all_predictions, args.output)
        print(f"Wrote {len(all_predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
