"""diskcache wrapper with per-cadence TTL for the Macro Regime Engine."""

from __future__ import annotations

import os
from typing import Any, Callable

import diskcache
import yaml
from loguru import logger

# Load settings once at import time
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

_TTL_MAP: dict[str, int] = _SETTINGS["cache"]["ttl_seconds"]
_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", os.getenv("CACHE_DIR", _SETTINGS["cache"]["directory"])
)

_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    """Return the singleton diskcache.Cache instance, creating it if needed."""
    global _cache
    if _cache is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _cache = diskcache.Cache(_CACHE_DIR)
        logger.debug(f"Opened diskcache at {_CACHE_DIR}")
    return _cache


def ttl_for_cadence(cadence: str) -> int:
    """Return TTL in seconds for a given refresh cadence string."""
    return _TTL_MAP.get(cadence, _TTL_MAP["daily"])


def cached(key: str, cadence: str, fetch_fn: Callable[[], Any]) -> Any:
    """
    Return cached value for *key* if still fresh, otherwise call *fetch_fn*,
    store the result, and return it.

    Args:
        key: Cache key string.
        cadence: Refresh cadence ("realtime" | "daily" | "weekly" | "monthly").
        fetch_fn: Zero-argument callable that returns the fresh value.

    Returns:
        The cached or freshly fetched value.
    """
    cache = get_cache()
    value = cache.get(key)
    if value is not None:
        logger.debug(f"Cache HIT  [{cadence}] {key}")
        return value

    logger.debug(f"Cache MISS [{cadence}] {key} — fetching live")
    value = fetch_fn()
    ttl = ttl_for_cadence(cadence)
    cache.set(key, value, expire=ttl)
    logger.debug(f"Cached [{cadence}] {key} for {ttl}s")
    return value


def invalidate(key: str) -> None:
    """Remove a specific key from the cache."""
    get_cache().delete(key)
    logger.debug(f"Invalidated cache key: {key}")


def clear_all() -> None:
    """Wipe the entire cache. Use with caution."""
    get_cache().clear()
    logger.warning("Cleared entire diskcache")
