"""Shared LLM client adapter.

The application layer uses generic LLM names, while the current backend adapter
uses Google Gemini through ``google-genai``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import LLMConfig


def generate_llm_json(
    prompt: str,
    config: LLMConfig,
    temperature: float,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate and parse a JSON object from the configured LLM provider."""

    resolved_client = client or _build_llm_client(config)
    response = _generate_content(resolved_client, prompt, config, temperature)
    payload = _parse_json_response(_response_text(response))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _build_llm_client(config: LLMConfig) -> Any:
    if config.llm_provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER {config.llm_provider!r}")
    if not config.llm_api_key:
        raise ValueError("LLM_API_KEY is required when an LLM feature is enabled")

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "google-genai is required for the configured LLM provider."
        ) from exc

    return genai.Client(api_key=config.llm_api_key)


def _generate_content(
    client: Any,
    prompt: str,
    config: LLMConfig,
    temperature: float,
) -> Any:
    generation_config = _build_generation_config(temperature)
    return client.models.generate_content(
        model=config.llm_model,
        contents=prompt,
        config=generation_config,
    )


def _build_generation_config(temperature: float) -> Any:
    try:
        from google.genai import types
    except ImportError:  # pragma: no cover - mocked tests do not need SDK types
        return {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }

    return types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    if isinstance(response, str):
        return response
    raise ValueError("LLM response did not contain text")


def _parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid JSON") from exc
