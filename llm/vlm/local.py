"""Local vision-language adapter for KIS reranking and QA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm.config import LLMConfig, configure_huggingface_cache, get_llm_config
from llm.schemas import Candidate

from .common import parse_rank_response, rank_prompt


_LOCAL_VLM_INSTANCES: dict[tuple[Any, ...], "LocalVLM"] = {}


def get_local_vlm(config: LLMConfig | None = None) -> "LocalVLM":
    """Return one shared local model instance for text and image prompts."""

    resolved = config or get_llm_config()
    key = (
        resolved.local_model,
        str(resolved.model_cache_dir),
        resolved.local_device,
        resolved.local_load_in_4bit,
        resolved.local_min_pixels,
        resolved.local_max_pixels,
    )
    if key not in _LOCAL_VLM_INSTANCES:
        _LOCAL_VLM_INSTANCES[key] = LocalVLM(resolved)
    return _LOCAL_VLM_INSTANCES[key]


class LocalVLM:
    """Qwen-VL adapter with lazy model loading and bounded image batches."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or get_llm_config()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._rank_cache: dict[
            tuple[str, tuple[tuple[str, int], ...]], dict[tuple[str, int], float]
        ] = {}
        self._answer_cache: dict[tuple[str, str, int], str] = {}

    def rank(self, query: str, candidates: list[Candidate]) -> dict[tuple[str, int], float]:
        """Score candidates locally, using one bounded generation per chunk."""

        key = (query, tuple((item.video_id, item.frame_id) for item in candidates))
        if key in self._rank_cache:
            return self._rank_cache[key]

        usable = [item for item in candidates if Path(item.keyframe_path).exists()]
        scores: dict[tuple[str, int], float] = {}
        for start in range(0, len(usable), self.config.local_batch_size):
            chunk = usable[start : start + self.config.local_batch_size]
            payload = self._generate_json(
                rank_prompt(query, chunk),
                [item.keyframe_path for item in chunk],
            )
            scores.update(parse_rank_response(payload, chunk))
        self._rank_cache[key] = scores
        return scores

    def answer(self, question: str, candidate: Candidate) -> str:
        """Answer a QA question from one evidence frame without an API call."""

        key = (question, candidate.video_id, candidate.frame_id)
        if key in self._answer_cache:
            return self._answer_cache[key]
        if not Path(candidate.keyframe_path).exists():
            return ""

        prompt = (
            "You are a visual question answering judge. Answer the question using "
            "only the visible evidence in the image. Return JSON only: "
            '{"answer": "short answer"}. Do not explain.\n'
            f"Question: {question}"
        )
        payload = self._generate_json(prompt, [candidate.keyframe_path], max_new_tokens=64)
        answer = str(payload.get("answer", "")).strip()
        self._answer_cache[key] = answer
        return answer

    def generate_text_json(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON object from text only using the shared local model."""

        return self._generate_json(
            prompt,
            [],
            max_new_tokens=max_new_tokens or self.config.local_text_max_new_tokens,
        )

    def _generate_json(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        content = [
                {"type": "image", "image": str(Path(image_path))}
            for image_path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.local_max_new_tokens,
                do_sample=False,
            )
        generated_trimmed = [
            output_ids[input_ids.shape[-1] :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        response = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _parse_json_object(response)

    def _load(self) -> None:
        if self._model is not None:
            return

        hf_cache_dir = configure_huggingface_cache(self.config.model_cache_dir)
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForMultimodalLM as ModelClass
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ImportError(
                "Local VLM requires a recent Transformers build with Qwen3-VL "
                "support, accelerate, and torch. Install requirements.txt."
            ) from exc

        use_cuda = self.config.local_device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "LOCAL_DEVICE=cuda but CUDA is unavailable. "
                "Install a CUDA PyTorch build or set LOCAL_DEVICE=cpu."
            )
        if self.config.local_load_in_4bit and not use_cuda:
            raise RuntimeError(
                "LOCAL_LOAD_IN_4BIT=true requires LOCAL_DEVICE=cuda"
            )

        model_kwargs = _base_model_load_kwargs(self.config.local_device, torch)
        if hf_cache_dir is not None:
            model_kwargs["cache_dir"] = str(hf_cache_dir)
        if self.config.local_load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "4-bit local VLM loading requires bitsandbytes. "
                    "Install requirements.txt or set LOCAL_LOAD_IN_4BIT=false."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        try:
            self._model = ModelClass.from_pretrained(
                self.config.local_model,
                **model_kwargs,
            )
        except TypeError:
            # Older Transformers releases use torch_dtype instead of dtype.
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self._model = ModelClass.from_pretrained(
                self.config.local_model,
                **model_kwargs,
            )
        processor_kwargs: dict[str, Any] = {
            "min_pixels": self.config.local_min_pixels,
            "max_pixels": self.config.local_max_pixels,
        }
        if hf_cache_dir is not None:
            processor_kwargs["cache_dir"] = str(hf_cache_dir)
        self._processor = AutoProcessor.from_pretrained(
            self.config.local_model,
            **processor_kwargs,
        )
        self._torch = torch


def _base_model_load_kwargs(local_device: str, torch_module: Any) -> dict[str, Any]:
    """Build a device map that cannot silently ignore the selected profile."""

    if local_device.startswith("cuda"):
        return {"device_map": {"": local_device}, "dtype": "auto"}
    if local_device != "cpu":
        raise ValueError(f"Unsupported LOCAL_DEVICE: {local_device!r}")
    return {"device_map": {"": "cpu"}, "dtype": torch_module.float32}


def _parse_json_object(text: str) -> dict[str, Any]:
    """Recover and normalize a JSON object emitted by a local model."""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        if cleaned.startswith("{"):
            partial = _recover_partial_object(cleaned)
            if partial:
                return partial
        decoder = json.JSONDecoder()
        matches = list(re.finditer(r"[\[{]", cleaned))
        # If the response starts as an object/array, decoding a nested array
        # after that outer value fails would silently return the wrong shape.
        # This is common when generation is truncated inside preserved_terms.
        if cleaned.startswith(("{", "[")):
            matches = matches[:1]
        for match in matches:
            try:
                value, _ = decoder.raw_decode(cleaned[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            preview = " ".join(cleaned.split())[:240]
            raise ValueError(f"Local VLM returned invalid JSON: {preview!r}")
    return _coerce_json_object(value, cleaned)


def _recover_partial_object(text: str) -> dict[str, Any]:
    """Keep complete top-level fields when generation ends mid-object."""

    decoder = json.JSONDecoder()
    recovered: dict[str, Any] = {}
    index = 1
    length = len(text)
    while index < length:
        while index < length and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= length or text[index] == "}":
            break
        try:
            key, key_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if not isinstance(key, str):
            break
        index = key_end
        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] != ":":
            break
        index += 1
        while index < length and text[index].isspace():
            index += 1
        try:
            field_value, value_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        recovered[key] = field_value
        index = value_end
    return recovered


def _coerce_json_object(value: Any, source_text: str) -> dict[str, Any]:
    """Repair common shape deviations without inventing semantic fields."""

    if isinstance(value, dict):
        if len(value) == 1:
            wrapped = next(iter(value.values()))
            if isinstance(wrapped, dict):
                return wrapped
        return value

    if isinstance(value, list):
        object_items = [item for item in value if isinstance(item, dict)]
        if object_items:
            rank_keys = {"index", "video_id", "frame_id", "relevant", "confidence"}
            if all(rank_keys.intersection(item) for item in object_items):
                return {"items": object_items}
            return object_items[0]

    preview = " ".join(source_text.split())[:240]
    raise ValueError(
        "Local VLM response could not be normalized to a JSON object; "
        f"received {type(value).__name__}: {preview!r}"
    )
