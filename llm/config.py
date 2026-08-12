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
    """Runtime knobs for API-backed LLM calls."""

    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    use_llm_classifier: bool
    llm_classifier_temperature: float
    intent_confidence_threshold: float
    use_llm_parser: bool
    llm_parser_mode: str
    llm_parser_temperature: float


def get_llm_config(load_env: bool = True) -> LLMConfig:
    """Read LLM settings from environment variables."""

    if load_env:
        load_project_env()

    return LLMConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        use_llm_classifier=_read_bool("USE_LLM_CLASSIFIER", False),
        llm_classifier_temperature=float(os.getenv("LLM_CLASSIFIER_TEMPERATURE", "0")),
        intent_confidence_threshold=float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.8")),
        use_llm_parser=_read_bool("USE_LLM_PARSER", False),
        llm_parser_mode=os.getenv("LLM_PARSER_MODE", "auto").strip().lower(),
        llm_parser_temperature=float(os.getenv("LLM_PARSER_TEMPERATURE", "0")),
    )
