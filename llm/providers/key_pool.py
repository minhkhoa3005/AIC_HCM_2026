"""Round-robin API key pool with retry cooldowns."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class _KeyState:
    value: str
    cooldown_until: float = 0.0


class APIKeyPool:
    """Rotate valid API keys without exposing raw keys in logs or repr."""

    def __init__(self, keys: list[str], cooldown_seconds: float = 60.0) -> None:
        cleaned = [key.strip() for key in keys if isinstance(key, str) and key.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty API key is required")
        self._keys = [_KeyState(value=key) for key in cleaned]
        self._index = 0
        self.cooldown_seconds = cooldown_seconds

    def __repr__(self) -> str:
        return f"APIKeyPool(size={len(self._keys)}, cooldown_seconds={self.cooldown_seconds})"

    def next_key(self) -> str:
        now = monotonic()
        total = len(self._keys)
        for offset in range(total):
            index = (self._index + offset) % total
            state = self._keys[index]
            if state.cooldown_until <= now:
                self._index = (index + 1) % total
                return state.value
        raise RuntimeError("No API keys are currently available")

    def mark_failed(self, key: str) -> None:
        for state in self._keys:
            if state.value == key:
                state.cooldown_until = monotonic() + self.cooldown_seconds
                return

    def mark_available(self, key: str) -> None:
        for state in self._keys:
            if state.value == key:
                state.cooldown_until = 0.0
                return
