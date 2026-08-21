"""Tests for explicit CPU and CUDA runtime profiles."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm.config import configure_huggingface_cache, get_llm_config
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


def test_relative_model_cache_is_resolved_from_project_root(monkeypatch) -> None:
    monkeypatch.setenv("AIC_MODEL_CACHE_DIR", "runtime_models")
    config = get_llm_config(load_env=False)

    expected_root = Path(__file__).resolve().parents[1] / "runtime_models"
    assert config.model_cache_dir == expected_root.resolve()


def test_model_cache_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AIC_MODEL_CACHE_DIR", "")
    config = get_llm_config(load_env=False)
    assert config.model_cache_dir is None


def test_huggingface_and_xet_caches_follow_model_cache(tmp_path, monkeypatch) -> None:
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.delenv(name, raising=False)

    hub_cache = configure_huggingface_cache(tmp_path)

    assert hub_cache == tmp_path / "huggingface" / "hub"
    assert Path(os.environ["HF_HOME"]) == tmp_path / "huggingface"
    assert Path(os.environ["HF_HUB_CACHE"]) == hub_cache
    assert Path(os.environ["HF_XET_CACHE"]) == tmp_path / "huggingface" / "xet"
