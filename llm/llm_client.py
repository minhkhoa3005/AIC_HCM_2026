"""Shared LLM client adapter.

The application layer uses generic LLM names, while the current backend adapter
uses Google Gemini through ``google-genai``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import LLMConfig
from .providers.key_pool import APIKeyPool

_KEY_POOLS: dict[tuple[str, ...], APIKeyPool] = {}


def generate_llm_json(
    prompt: str,
    config: LLMConfig,
    temperature: float,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate and parse a JSON object from the configured LLM provider."""

    if client is not None:
        response = _generate_content(client, prompt, config, temperature)
    else:
        response = _generate_content_with_key_rotation(prompt, config, temperature)
    response_text = _response_text(response)
    payload = _parse_json_response(response_text)
    if isinstance(payload, list):
        # Some model/API combinations return several valid alternatives even
        # when the prompt requests one object. Keep the first deterministic
        # object; rewrite/planner validation still checks its schema strictly.
        objects = [item for item in payload if isinstance(item, dict)]
        if objects:
            payload = objects[0]
    if not isinstance(payload, dict):
        preview = " ".join(response_text.split())[:240]
        raise ValueError(
            "LLM response must be a JSON object; "
            f"received {type(payload).__name__}: {preview!r}"
        )
    return payload


def generate_vlm_json(
    prompt: str,
    image_paths: list[str],
    config: LLMConfig,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate JSON from Gemini using a text prompt plus local images."""
    if client is not None:
        response = _generate_multimodal_content(client, prompt, image_paths, config)
    else:
        response = _generate_multimodal_with_key_rotation(prompt, image_paths, config)
    response_text = _response_text(response)
    payload = _parse_json_response(response_text)
    if isinstance(payload, list):
        objects = [item for item in payload if isinstance(item, dict)]
        if objects:
            payload = objects[0]
    if not isinstance(payload, dict):
        preview = " ".join(response_text.split())[:240]
        raise ValueError(
            "VLM response must be a JSON object; "
            f"received {type(payload).__name__}: {preview!r}"
        )
    return payload


def _build_llm_client_for_key(config: LLMConfig, api_key: str) -> Any:
    if config.llm_provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER {config.llm_provider!r}")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "google-genai is required for the configured LLM provider."
        ) from exc

    return genai.Client(api_key=api_key)


def _get_key_pool(keys: tuple[str, ...]) -> APIKeyPool:
    if keys not in _KEY_POOLS:
        _KEY_POOLS[keys] = APIKeyPool(list(keys))
    return _KEY_POOLS[keys]


def _generate_content_with_key_rotation(
    prompt: str,
    config: LLMConfig,
    temperature: float,
) -> Any:
    if not config.llm_api_keys:
        raise ValueError("LLM_API_KEYS is required when an LLM feature is enabled")

    pool = _get_key_pool(config.llm_api_keys)
    last_error: Exception | None = None
    for _ in range(len(config.llm_api_keys)):
        api_key = pool.next_key()
        try:
            client = _build_llm_client_for_key(config, api_key)
            response = _generate_content(client, prompt, config, temperature)
            pool.mark_available(api_key)
            return response
        except Exception as exc:
            pool.mark_failed(api_key)
            last_error = exc

    raise RuntimeError(
        "All configured LLM API keys failed or are cooling down. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _generate_multimodal_with_key_rotation(
    prompt: str,
    image_paths: list[str],
    config: LLMConfig,
) -> Any:
    if not config.llm_api_keys:
        raise ValueError("LLM_API_KEYS is required when a VLM feature is enabled")

    pool = _get_key_pool(config.llm_api_keys)
    last_error: Exception | None = None
    for _ in range(len(config.llm_api_keys)):
        api_key = pool.next_key()
        try:
            client = _build_llm_client_for_key(config, api_key)
            response = _generate_multimodal_content(client, prompt, image_paths, config)
            pool.mark_available(api_key)
            return response
        except Exception as exc:
            pool.mark_failed(api_key)
            last_error = exc
    raise RuntimeError("All configured VLM API keys failed or are cooling down") from last_error


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


def _generate_multimodal_content(
    client: Any,
    prompt: str,
    image_paths: list[str],
    config: LLMConfig,
) -> Any:
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError("google-genai is required for Gemini VLM") from exc

    contents: list[Any] = [prompt]
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"VLM image not found: {path}")
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
    return client.models.generate_content(
        model=config.llm_vlm_model,
        contents=contents,
        config=_build_generation_config(0.0),
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
    if not cleaned:
        raise ValueError("LLM returned an empty response")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover JSON when a provider adds a short explanation before/after
        # the object despite the response MIME type and prompt contract.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", cleaned):
            try:
                value, _ = decoder.raw_decode(cleaned[match.start():])
                return value
            except json.JSONDecodeError:
                continue
        preview = " ".join(cleaned.split())[:240]
        raise ValueError(f"LLM returned invalid JSON: {preview!r}") from None
