# cache.py — Synthesis result caching for CrispTTS.
#
# Caches synthesized audio by hashing (model_id, voice, text, params).
# Uses a simple file-based LRU cache with configurable max size.

import hashlib
import json
import logging
import os
import shutil

logger = logging.getLogger("CrispTTS.cache")

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "crisptts", "synthesis")
_DEFAULT_MAX_MB = 500

_cache_dir = _DEFAULT_CACHE_DIR
_max_bytes = _DEFAULT_MAX_MB * 1024 * 1024
_enabled = False


def configure(cache_dir: str | None = None, max_mb: int = 500, enabled: bool = True):
    """Configure the synthesis cache."""
    global _cache_dir, _max_bytes, _enabled
    _cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    _max_bytes = max_mb * 1024 * 1024
    _enabled = enabled
    if _enabled:
        os.makedirs(_cache_dir, exist_ok=True)
        logger.info("Synthesis cache: %s (max %d MB)", _cache_dir, max_mb)


# Include version in cache key so upgrades auto-invalidate stale entries.
# This prevents serving unwatermarked audio from a pre-watermark cache.
_VERSION = "0.8.0"
try:
    import importlib.metadata
    _VERSION = importlib.metadata.version("crisptts")
except Exception:  # noqa: S110 — version lookup is best-effort
    pass


def _cache_key(model_id: str, voice: str | None, text: str, params: str | None) -> str:
    """Compute a cache key from synthesis parameters + version."""
    raw = json.dumps({"m": model_id, "v": voice or "", "t": text, "p": params or "",
                      "_ver": _VERSION},
                     sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def lookup(model_id: str, voice: str | None, text: str, params: str | None,
           output_ext: str = ".wav") -> str | None:
    """Check if a cached result exists. Returns the cached file path or None."""
    if not _enabled:
        return None
    key = _cache_key(model_id, voice, text, params)
    cached_path = os.path.join(_cache_dir, f"{key}{output_ext}")
    if os.path.isfile(cached_path) and os.path.getsize(cached_path) > 100:
        # Touch access time for LRU
        os.utime(cached_path, None)
        logger.info("Cache hit: %s", key)
        return cached_path
    return None


def store(model_id: str, voice: str | None, text: str, params: str | None,
          source_path: str, output_ext: str = ".wav") -> str | None:
    """Store a synthesis result in the cache. Returns the cached path."""
    if not _enabled:
        return None
    if not os.path.isfile(source_path) or os.path.getsize(source_path) < 100:
        return None
    key = _cache_key(model_id, voice, text, params)
    cached_path = os.path.join(_cache_dir, f"{key}{output_ext}")
    try:
        shutil.copy2(source_path, cached_path)
        logger.debug("Cache store: %s → %s", key, cached_path)
        _evict_if_needed()
        return cached_path
    except OSError as e:
        logger.debug("Cache store failed: %s", e)
        return None


def _evict_if_needed():
    """Evict oldest entries if total cache size exceeds the limit."""
    try:
        entries = []
        total = 0
        for name in os.listdir(_cache_dir):
            path = os.path.join(_cache_dir, name)
            if os.path.isfile(path):
                stat = os.stat(path)
                entries.append((stat.st_atime, stat.st_size, path))
                total += stat.st_size

        if total <= _max_bytes:
            return

        # Sort by access time (oldest first) and evict
        entries.sort()
        for _atime, size, path in entries:
            if total <= _max_bytes * 0.8:  # evict to 80% to avoid thrashing
                break
            try:
                os.unlink(path)
                total -= size
                logger.debug("Cache evict: %s", path)
            except OSError:
                pass
    except OSError:
        pass
