"""Local JSON generation for rewrite, planning, reranking, and QA."""

from __future__ import annotations

from typing import Any

from .config import LLMConfig


def generate_llm_json(
    prompt: str,
    config: LLMConfig,
    temperature: float,
    client: Any | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate text-only JSON with the shared local model."""

    if client is not None:
        raise ValueError("External LLM clients are disabled; use the local model")
    if config.llm_provider != "local":
        raise ValueError(
            f"Only local inference is supported; set LLM_PROVIDER=local, got {config.llm_provider!r}"
        )
    from .vlm.local import get_local_vlm

    return get_local_vlm(config).generate_text_json(
        prompt,
        max_new_tokens=max_new_tokens,
    )


def generate_vlm_json(
    prompt: str,
    image_paths: list[str],
    config: LLMConfig,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate multimodal JSON with the shared local model."""

    if client is not None:
        raise ValueError("External VLM clients are disabled; use the local model")
    if config.vlm_provider != "local":
        raise ValueError(
            f"Only local inference is supported; set VLM_PROVIDER=local, got {config.vlm_provider!r}"
        )
    from .vlm.local import get_local_vlm

    return get_local_vlm(config)._generate_json(prompt, image_paths)
