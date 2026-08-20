"""Evaluate per-query BTC result CSVs against answer text files.

The scorer follows the BTC rule: for each query, R@k is the maximum R-score
inside the first k ranked rows; Final Score is the mean of R@1, R@5, R@20,
R@50 and R@100. Ranges are inclusive and TRAKE paths require event order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


KS = (1, 5, 20, 50, 100)
VIDEO_RE = re.compile(r"(?im)^\s*Video\s*ID\s*:\s*([^\s\r\n]+)")
RANGE_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
EVENT_RE = re.compile(r"(?im)^\s*Event\s+\d+\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]")


@dataclass(frozen=True)
class Region:
    video_id: str
    start: int
    end: int
    answer: str = ""

    def accepts(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end


@dataclass(frozen=True)
class TrakePath:
    video_id: str
    events: tuple[tuple[int, int], ...]


def _task_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    if "trake" in stem:
        return "trake"
    if re.search(r"(^|[-_])qa($|[-_])", stem):
        return "qa"
    if "kis" in stem:
        return "kis"
    raise ValueError(f"Cannot infer task type from filename: {name}")


def _blocks(text: str) -> list[str]:
    matches = list(VIDEO_RE.finditer(text))
    return [text[item.start() : (matches[index + 1].start() if index + 1 < len(matches) else len(text))] for index, item in enumerate(matches)]


def parse_answer(path: Path, task: str) -> tuple[list[Region], list[TrakePath]]:
    text = path.read_text(encoding="utf-8-sig")
    regions: list[Region] = []
    paths: list[TrakePath] = []
    for block in _blocks(text):
        video_match = VIDEO_RE.search(block)
        if not video_match:
            continue
        video_id = video_match.group(1).strip()
        if task == "trake":
            events = tuple((int(match.group(1)), int(match.group(2))) for match in EVENT_RE.finditer(block))
            if events:
                paths.append(TrakePath(video_id, events))
            continue

        frame_match = RANGE_RE.search(block)
        if not frame_match:
            raise ValueError(f"Missing Frame Range in {path}: {block[:120]!r}")
        answer_match = re.search(r"(?ims)^\s*Answer\s*:\s*(.*?)(?=^\s*Video\s*ID\s*:|\Z)", block)
        answer = " ".join(answer_match.group(1).split()) if answer_match else ""
        regions.append(Region(video_id, int(frame_match.group(1)), int(frame_match.group(2)), answer))
    if task == "trake" and not paths:
        raise ValueError(f"No TRAKE events found in {path}")
    if task != "trake" and not regions:
        raise ValueError(f"No answer regions found in {path}")
    return regions, paths


def _read_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            frames = [int(value) for value in re.findall(r"-?\d+", row.get("frame_ids", ""))]
            rows.append({
                "rank": int(row.get("rank", len(rows) + 1)),
                "video_id": str(row.get("video_id", "")).removesuffix(".mp4"),
                "frame_ids": frames,
                "answer": row.get("answer", "") or "",
            })
        return sorted(rows, key=lambda row: row["rank"])


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.findall(r"[\w]+", plain, flags=re.UNICODE))


def _answer_matches(predicted: str, expected: str) -> bool:
    predicted_norm = _normalize(predicted)
    alternatives = re.split(r"\s*(?:hoặc|or)\s*", expected, flags=re.IGNORECASE)
    for alternative in alternatives:
        expected_norm = _normalize(alternative)
        if not predicted_norm or not expected_norm:
            continue
        if predicted_norm == expected_norm or predicted_norm in expected_norm or expected_norm in predicted_norm:
            return True
        overlap = len(set(predicted_norm.split()) & set(expected_norm.split())) / max(len(set(expected_norm.split())), 1)
        if overlap >= 0.8:
            return True
    return False


def _kis_qa_hit(row: dict, regions: list[Region], task: str) -> bool:
    for region in regions:
        if row["video_id"] != region.video_id:
            continue
        if not any(region.accepts(frame) for frame in row["frame_ids"]):
            continue
        if task == "kis" or _answer_matches(row["answer"], region.answer):
            return True
    return False


def _trake_hit(row: dict, paths: list[TrakePath]) -> bool:
    frames = row["frame_ids"]
    if not frames or any(left >= right for left, right in zip(frames, frames[1:])):
        return False
    return any(
        row["video_id"] == path.video_id
        and len(frames) == len(path.events)
        and all(start <= frame <= end for frame, (start, end) in zip(frames, path.events))
        for path in paths
    )


def evaluate(answer_dir: Path, result_dir: Path) -> dict:
    per_query: dict[str, dict] = {}
    for answer_path in sorted(answer_dir.glob("*.txt")):
        query_id = answer_path.stem
        task = _task_from_name(query_id)
        result_path = result_dir / f"{query_id}-result.csv"
        if not result_path.is_file():
            # Also accept the combined exporter filename for backward compatibility.
            result_path = result_dir / f"{query_id}.csv"
        regions, paths = parse_answer(answer_path, task)
        rows = _read_results(result_path) if result_path.exists() else []
        hits = [
            _trake_hit(row, paths) if task == "trake" else _kis_qa_hit(row, regions, task)
            for row in rows[:100]
        ]
        metrics = {f"R@{k}": float(any(hits[:k])) for k in KS}
        correct_ranks = [index + 1 for index, hit in enumerate(hits) if hit]
        correct_predictions = [
            {
                "rank": row["rank"],
                "video_id": row["video_id"],
                "frame_ids": row["frame_ids"],
                "answer": row["answer"],
            }
            for row, hit in zip(rows[:100], hits)
            if hit
        ]
        per_query[query_id] = {
            "task_type": task,
            **metrics,
            "first_hit_rank": correct_ranks[0] if correct_ranks else None,
            "correct_ranks": correct_ranks,
            "correct_predictions": correct_predictions,
            "prediction_count": len(rows),
            "status": "hit" if any(hits) else "miss",
        }

    aggregate = {f"R@{k}": sum(item[f"R@{k}"] for item in per_query.values()) / max(len(per_query), 1) for k in KS}
    return {**aggregate, "Final Score": sum(aggregate.values()) / len(KS), "per_query": per_query}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score BTC KIS/QA/TRAKE result CSVs")
    parser.add_argument("--answer-dir", default="answer")
    parser.add_argument("--result-dir", default="outputs/results")
    parser.add_argument("--report", default="outputs/evaluation_report.json")
    args = parser.parse_args()
    report = evaluate(Path(args.answer_dir), Path(args.result_dir))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in (*[f"R@{k}" for k in KS], "Final Score"):
        print(f"{key}: {report[key]:.4f}")
    print("Correct ranks:")
    for query_id, item in report["per_query"].items():
        ranks = item["correct_ranks"]
        print(f"  {query_id}: {', '.join(f'Top-{rank}' for rank in ranks) if ranks else 'no hit'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
