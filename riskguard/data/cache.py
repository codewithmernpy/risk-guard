"""Thread-safe in-memory TTL cache for market data and analytics."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class TTLCache:
    """Simple thread-safe time-to-live in-memory key-value cache."""

    def __init__(self, default_ttl_seconds: float = 60.0, max_entries: int = 256) -> None:
        self._default_ttl = float(default_ttl_seconds)
        self._max_entries = int(max_entries)
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    @property
    def default_ttl_seconds(self) -> float:
        """Configured default TTL in seconds."""
        return self._default_ttl

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        ttl = self._default_ttl if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            return
        expires_at = time.monotonic() + ttl
        with self._lock:
            if len(self._store) >= self._max_entries and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest_key]
            self._store[key] = (expires_at, value)

    def get_or_set(self, key: str, factory: Callable[[], T], ttl_seconds: float | None = None) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl_seconds=ttl_seconds)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
