import pytest
from types import SimpleNamespace

from llm.config import LLMConfig
from llm import llm_client
from llm.providers.key_pool import APIKeyPool


def test_key_pool_round_robin_and_repr_hides_keys():
    pool = APIKeyPool(["alpha", "beta"], cooldown_seconds=10)

    assert pool.next_key() == "alpha"
    assert pool.next_key() == "beta"
    assert pool.next_key() == "alpha"
    assert "alpha" not in repr(pool)


def test_key_pool_cools_down_failed_key():
    pool = APIKeyPool(["alpha", "beta"], cooldown_seconds=10)

    failed = pool.next_key()
    pool.mark_failed(failed)

    assert pool.next_key() == "beta"
    with pytest.raises(RuntimeError):
        pool.mark_failed("beta")
        pool.next_key()


def _config_with_keys():
    return LLMConfig(
        llm_provider="gemini",
        llm_api_keys=("alpha", "beta"),
        llm_model="gemini-2.5-flash",
        llm_rewrite_temperature=0,
        llm_planner_temperature=0,
    )


def test_llm_client_reuses_key_pool_for_round_robin(monkeypatch):
    used_keys = []
    llm_client._KEY_POOLS.clear()

    def fake_build_client(_config, api_key):
        used_keys.append(api_key)
        return SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs: SimpleNamespace(text='{"ok": true}')
            )
        )

    monkeypatch.setattr(llm_client, "_build_llm_client_for_key", fake_build_client)
    config = _config_with_keys()

    assert llm_client.generate_llm_json("prompt", config, 0) == {"ok": True}
    assert llm_client.generate_llm_json("prompt", config, 0) == {"ok": True}

    assert used_keys == ["alpha", "beta"]


def test_llm_client_retries_next_key_after_provider_error(monkeypatch):
    used_keys = []
    llm_client._KEY_POOLS.clear()

    def fake_build_client(_config, api_key):
        used_keys.append(api_key)
        if api_key == "alpha":
            return SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("quota")
                    )
                )
            )
        return SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs: SimpleNamespace(text='{"ok": true}')
            )
        )

    monkeypatch.setattr(llm_client, "_build_llm_client_for_key", fake_build_client)

    assert llm_client.generate_llm_json("prompt", _config_with_keys(), 0) == {"ok": True}
    assert used_keys == ["alpha", "beta"]
