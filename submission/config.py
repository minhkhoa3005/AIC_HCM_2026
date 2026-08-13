"""Configurable CSV submission formatting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubmissionConfig:
    frame_base: int = 0
    include_mp4_extension: bool = False
    header: tuple[str, ...] = ("query_id", "rank", "video_id", "frame_ids", "answer")
    separator: str = ","
