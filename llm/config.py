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
    """Runtime knobs for future API-backed LLM calls."""

    openai_api_key: str | None
    openai_model: str
    use_llm_classifier: bool
    classifier_temperature: float


def get_llm_config(load_env: bool = True) -> LLMConfig:
    """Read LLM settings from environment variables."""

    if load_env:
        load_project_env()

    return LLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        use_llm_classifier=_read_bool("USE_LLM_CLASSIFIER", False),
        classifier_temperature=float(os.getenv("CLASSIFIER_TEMPERATURE", "0")),
    )
