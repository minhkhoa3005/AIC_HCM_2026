"""Local JSON generation for rewrite, planning, reranking, and QA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import LLMConfig


def generate_llm_json(
    prompt: str,
    config: LLMConfig,
    temperature: float,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate text-only JSON with the shared local model."""

    if client is not None:
        raise ValueError("External LLM clients are disabled; use the local model")
    if config.llm_provider != "local":
        raise ValueError(
            f"Only local inference is supported; set LLM_PROVIDER=local, got {config.llm_provider!r}"
        )
    from .vlm.local import get_local_vlm

    return get_local_vlm(config).generate_text_json(prompt)


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


def _build_llm_client_for_key(config: LLMConfig, api_key: str) -> Any:
    raise RuntimeError("External API inference is disabled; use the local model")


def _get_key_pool(keys: tuple[str, ...], model: str) -> Any:
    raise RuntimeError("External API inference is disabled; use the local model")


def _generate_content_with_key_rotation(
    prompt: str,
    config: LLMConfig,
    temperature: float,
) -> Any:
    if not config.llm_api_keys:
        raise RuntimeError("External API inference is disabled; use the local model")

    pool = _get_key_pool(config.llm_api_keys, config.llm_model)
    last_error: Exception | None = None
    for _ in range(len(config.llm_api_keys)):
        try:
            api_key = pool.next_key()
        except RuntimeError as exc:
            wait = pool.cooldown_remaining()
            raise RuntimeError(
                f"No API keys available for model {config.llm_model!r}; "
                f"all keys are cooling down. Retry in about {wait:.0f}s."
            ) from exc
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
    raise RuntimeError("External API inference is disabled; use the local model")

    raise RuntimeError("External API inference is disabled; use the local model")

    raise RuntimeError("External API inference is disabled; use the local model")

    if not config.llm_api_keys:
        raise RuntimeError("External API inference is disabled; use the local model")

    pool = _get_key_pool(config.llm_api_keys, config.llm_vlm_model)
    last_error: Exception | None = None
    for _ in range(len(config.llm_api_keys)):
        try:
            api_key = pool.next_key()
        except RuntimeError as exc:
            wait = pool.cooldown_remaining()
            raise RuntimeError(
                f"No API keys available for VLM model {config.llm_vlm_model!r}; "
                f"all keys are cooling down. Retry in about {wait:.0f}s."
            ) from exc
        try:
            client = _build_llm_client_for_key(config, api_key)
            response = _generate_multimodal_content(client, prompt, image_paths, config)
            pool.mark_available(api_key)
            return response
        except Exception as exc:
            pool.mark_failed(api_key)
            last_error = exc
    raise RuntimeError(
        "All configured VLM API keys failed or are cooling down. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


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
    raise RuntimeError("External API inference is disabled; use the local model")

    raise RuntimeError("External API inference is disabled; use the local model")

    raise RuntimeError("External API inference is disabled; use the local model")

    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("External API inference is disabled; use the local model") from exc

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
