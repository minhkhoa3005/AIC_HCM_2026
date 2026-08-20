"""Environment-based configuration for LLM components."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


def load_project_env(env_path: str | Path = ".env") -> None:
    """Load a local .env file when python-dotenv is installed."""

    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=env_path)


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    """Runtime knobs for text LLM and local/API VLM calls."""

    llm_provider: str
    llm_api_keys: tuple[str, ...]
    llm_model: str
    llm_vlm_model: str
    vlm_provider: str
    vlm_local_model: str
    vlm_local_device: str
    vlm_local_load_in_4bit: bool
    vlm_local_batch_size: int
    vlm_local_max_new_tokens: int
    vlm_local_min_pixels: int
    vlm_local_max_pixels: int
    llm_rewrite_temperature: float
    llm_planner_temperature: float
    vlm_candidate_limit: int
    vlm_weight: float


def get_llm_config(load_env: bool = True) -> LLMConfig:
    """Read LLM settings from environment variables."""

    if load_env:
        load_project_env()

    api_keys = tuple(
        key.strip()
        for key in os.getenv("LLM_API_KEYS", "").replace("\n", ",").split(",")
        if key.strip()
    )

    llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    return LLMConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_api_keys=api_keys,
        llm_model=llm_model,
        llm_vlm_model=os.getenv("LLM_VLM_MODEL", "gemini-3.1-pro-preview").strip(),
        vlm_provider=os.getenv("VLM_PROVIDER", "local").strip().lower(),
        vlm_local_model=os.getenv(
            "VLM_LOCAL_MODEL", "Qwen/Qwen3-VL-2B-Instruct"
        ).strip(),
        vlm_local_device=os.getenv(
            "VLM_LOCAL_DEVICE", os.getenv("AIC_DEVICE", "cuda")
        ).strip().lower(),
        vlm_local_load_in_4bit=_read_bool("VLM_LOCAL_LOAD_IN_4BIT", True),
        vlm_local_batch_size=max(1, int(os.getenv("VLM_LOCAL_BATCH_SIZE", "8"))),
        vlm_local_max_new_tokens=max(
            16, int(os.getenv("VLM_LOCAL_MAX_NEW_TOKENS", "192"))
        ),
        vlm_local_min_pixels=max(
            28 * 28, int(os.getenv("VLM_LOCAL_MIN_PIXELS", str(256 * 28 * 28)))
        ),
        vlm_local_max_pixels=max(
            28 * 28, int(os.getenv("VLM_LOCAL_MAX_PIXELS", str(512 * 28 * 28)))
        ),
        llm_rewrite_temperature=float(os.getenv("LLM_REWRITE_TEMPERATURE", "0")),
        llm_planner_temperature=float(os.getenv("LLM_PLANNER_TEMPERATURE", "0")),
        vlm_candidate_limit=max(1, int(os.getenv("VLM_CANDIDATE_LIMIT", "40"))),
        vlm_weight=min(1.0, max(0.0, float(os.getenv("VLM_WEIGHT", "0.65")))),
    )
