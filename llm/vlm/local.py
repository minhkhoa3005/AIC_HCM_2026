"""Local vision-language adapter for KIS reranking and QA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm.config import LLMConfig, get_llm_config
from llm.schemas import Candidate

from .gemini import _parse_rank_response, _rank_prompt


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
        for start in range(0, len(usable), self.config.vlm_local_batch_size):
            chunk = usable[start : start + self.config.vlm_local_batch_size]
            payload = self._generate_json(
                _rank_prompt(query, chunk),
                [item.keyframe_path for item in chunk],
            )
            scores.update(_parse_rank_response(payload, chunk))
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
                max_new_tokens=max_new_tokens or self.config.vlm_local_max_new_tokens,
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
        payload = _parse_json_object(response)
        if not isinstance(payload, dict):
            raise ValueError("Local VLM response must be a JSON object")
        return payload

    def _load(self) -> None:
        if self._model is not None:
            return

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

        if self.config.vlm_local_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "VLM_LOCAL_DEVICE=cuda but CUDA is unavailable. "
                "Install a CUDA PyTorch build or set VLM_LOCAL_DEVICE=cpu."
            )
        if self.config.vlm_local_load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError("VLM_LOCAL_LOAD_IN_4BIT=true requires an NVIDIA CUDA device")

        model_kwargs: dict[str, Any] = {"device_map": "auto", "dtype": "auto"}
        if self.config.vlm_local_load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "4-bit local VLM loading requires bitsandbytes. "
                    "Install requirements.txt or set "
                    "VLM_LOCAL_LOAD_IN_4BIT=false."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        try:
            self._model = ModelClass.from_pretrained(
                self.config.vlm_local_model,
                **model_kwargs,
            )
        except TypeError:
            # Older Transformers releases use torch_dtype instead of dtype.
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self._model = ModelClass.from_pretrained(
                self.config.vlm_local_model,
                **model_kwargs,
            )
        self._processor = AutoProcessor.from_pretrained(
            self.config.vlm_local_model,
            min_pixels=self.config.vlm_local_min_pixels,
            max_pixels=self.config.vlm_local_max_pixels,
        )
        self._torch = torch


def _parse_json_object(text: str) -> Any:
    """Recover a JSON object when a local model emits a short wrapper."""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", cleaned):
            try:
                value, _ = decoder.raw_decode(cleaned[match.start() :])
                return value
            except json.JSONDecodeError:
                continue
    preview = " ".join(cleaned.split())[:240]
    raise ValueError(f"Local VLM returned invalid JSON: {preview!r}")
