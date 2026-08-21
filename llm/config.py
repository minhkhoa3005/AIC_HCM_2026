"""Environment-based configuration for LLM components."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def resolve_model_cache_dir() -> Path | None:
    """Resolve the shared model cache relative to the project root."""

    raw = os.getenv("AIC_MODEL_CACHE_DIR", "model_cache").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def configure_huggingface_cache(cache_root: Path | None = None) -> Path | None:
    """Route Hugging Face Hub and Xet files into the shared model cache."""

    resolved = cache_root if cache_root is not None else resolve_model_cache_dir()
    if resolved is None:
        return None
    hf_home = resolved / "huggingface"
    hub_cache = hf_home / "hub"
    xet_cache = hf_home / "xet"
    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    return hub_cache


@dataclass(frozen=True)
class LLMConfig:
    """Runtime knobs for the shared local text and vision-language model."""

    llm_provider: str
    vlm_provider: str
    local_model: str
    model_cache_dir: Path | None
    local_device: str
    local_load_in_4bit: bool
    local_batch_size: int
    local_max_new_tokens: int
    local_text_max_new_tokens: int
    local_min_pixels: int
    local_max_pixels: int
    llm_rewrite_temperature: float
    llm_planner_temperature: float
    vlm_candidate_limit: int
    vlm_weight: float


def get_llm_config(load_env: bool = True) -> LLMConfig:
    """Read LLM settings from environment variables."""

    if load_env:
        load_project_env()

    return LLMConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "local").strip().lower(),
        vlm_provider=os.getenv("VLM_PROVIDER", "local").strip().lower(),
        local_model=os.getenv(
            "LOCAL_MODEL", os.getenv("VLM_LOCAL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
        ).strip(),
        model_cache_dir=resolve_model_cache_dir(),
        local_device=os.getenv(
            "LOCAL_DEVICE", os.getenv("VLM_LOCAL_DEVICE", os.getenv("AIC_DEVICE", "cpu"))
        ).strip().lower(),
        local_load_in_4bit=_read_bool(
            "LOCAL_LOAD_IN_4BIT", _read_bool("VLM_LOCAL_LOAD_IN_4BIT", False)
        ),
        local_batch_size=max(1, int(os.getenv("VLM_LOCAL_BATCH_SIZE", "8"))),
        local_max_new_tokens=max(16, int(os.getenv("VLM_LOCAL_MAX_NEW_TOKENS", "192"))),
        local_text_max_new_tokens=max(
            64, int(os.getenv("LLM_LOCAL_MAX_NEW_TOKENS", "1536"))
        ),
        local_min_pixels=max(
            28 * 28, int(os.getenv("VLM_LOCAL_MIN_PIXELS", str(256 * 28 * 28)))
        ),
        local_max_pixels=max(
            28 * 28, int(os.getenv("VLM_LOCAL_MAX_PIXELS", str(512 * 28 * 28)))
        ),
        llm_rewrite_temperature=float(os.getenv("LLM_REWRITE_TEMPERATURE", "0")),
        llm_planner_temperature=float(os.getenv("LLM_PLANNER_TEMPERATURE", "0")),
        vlm_candidate_limit=max(1, int(os.getenv("VLM_CANDIDATE_LIMIT", "40"))),
        vlm_weight=min(1.0, max(0.0, float(os.getenv("VLM_WEIGHT", "0.65")))),
    )
