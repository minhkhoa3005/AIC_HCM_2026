"""Tests for explicit CPU and CUDA runtime profiles."""

from types import SimpleNamespace

import pytest

from llm.config import get_llm_config
from llm.vlm.local import _base_model_load_kwargs


def test_default_profile_is_cpu(monkeypatch) -> None:
    for name in (
        "LOCAL_DEVICE",
        "VLM_LOCAL_DEVICE",
        "AIC_DEVICE",
        "LOCAL_LOAD_IN_4BIT",
        "VLM_LOCAL_LOAD_IN_4BIT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = get_llm_config(load_env=False)
    assert config.local_device == "cpu"
    assert config.local_load_in_4bit is False


def test_cpu_profile_forces_cpu_device_map() -> None:
    float32 = object()
    fake_torch = SimpleNamespace(float32=float32)
    kwargs = _base_model_load_kwargs("cpu", fake_torch)

    assert kwargs == {"device_map": {"": "cpu"}, "dtype": float32}


def test_cuda_profile_keeps_requested_device() -> None:
    kwargs = _base_model_load_kwargs("cuda:0", SimpleNamespace())
    assert kwargs == {"device_map": {"": "cuda:0"}, "dtype": "auto"}


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported LOCAL_DEVICE"):
        _base_model_load_kwargs("directml", SimpleNamespace(float32=object()))
