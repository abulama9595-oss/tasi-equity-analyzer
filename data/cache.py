"""TTL disk cache for provider fetches.

Keeps the app fast on warm cache and protects rate-limited sources (SAHMK free tier
~100 req/day). Values are pickled; DataFrames round-trip fine. Keys are namespaced by
a logical "kind" so different TTLs apply (price_history vs fundamentals vs ...).
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, cache_dir: Path, ttl_seconds: dict[str, int], enabled: bool = True):
        self.dir = Path(cache_dir)
        self.ttl = dict(ttl_seconds)
        self.enabled = enabled
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _path(self, kind: str, key: str) -> Path:
        digest = hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()[:24]
        return self.dir / f"{kind}__{digest}.pkl"

    def _ttl_for(self, kind: str) -> int:
        # Unknown kinds fall back to a conservative 1h TTL.
        return int(self.ttl.get(kind, 3600))

    def get(self, kind: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(kind, key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl_for(kind):
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # corrupt cache entry — ignore and refetch
            log.warning("cache read failed for %s/%s: %s", kind, key, exc)
            return None

    def set(self, kind: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(kind, key)
        try:
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        except Exception as exc:
            log.warning("cache write failed for %s/%s: %s", kind, key, exc)

    def get_or_compute(self, kind: str, key: str, compute: Callable[[], Any]) -> Any:
        """Return cached value if fresh, else compute, store, and return it."""
        hit = self.get(kind, key)
        if hit is not None:
            return hit
        value = compute()
        if value is not None:
            self.set(kind, key, value)
        return value

    def clear(self) -> int:
        """Delete all cache entries. Returns the number of files removed."""
        if not self.dir.exists():
            return 0
        n = 0
        for f in self.dir.glob("*.pkl"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        return n
