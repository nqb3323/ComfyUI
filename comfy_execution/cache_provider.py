from typing import Any, Optional, Tuple, List
import hashlib
import json
import logging
import math
import threading

# Public types — source of truth is comfy_api.latest._caching
from comfy_api.latest._caching import CacheProvider, CacheContext, CacheValue

_logger = logging.getLogger(__name__)


_providers: List[CacheProvider] = []
_providers_lock = threading.Lock()
_providers_snapshot: Tuple[CacheProvider, ...] = ()


def register_cache_provider(provider: CacheProvider) -> None:
    """Register an external cache provider. Providers are called in registration order."""
    global _providers_snapshot
    with _providers_lock:
        if provider in _providers:
            _logger.warning(f"Provider {provider.__class__.__name__} already registered")
            return
        _providers.append(provider)
        _providers_snapshot = tuple(_providers)
        _logger.info(f"Registered cache provider: {provider.__class__.__name__}")


def unregister_cache_provider(provider: CacheProvider) -> None:
    global _providers_snapshot
    with _providers_lock:
        try:
            _providers.remove(provider)
            _providers_snapshot = tuple(_providers)
            _logger.info(f"Unregistered cache provider: {provider.__class__.__name__}")
        except ValueError:
            _logger.warning(f"Provider {provider.__class__.__name__} was not registered")


def _get_cache_providers() -> Tuple[CacheProvider, ...]:
    return _providers_snapshot


def _has_cache_providers() -> bool:
    return bool(_providers_snapshot)


def _clear_cache_providers() -> None:
    global _providers_snapshot
    with _providers_lock:
        _providers.clear()
        _providers_snapshot = ()


def _canonicalize(obj: Any) -> Any:
    # Convert to canonical JSON-serializable form with deterministic ordering.
    # Frozensets have non-deterministic iteration order between Python sessions.
    if isinstance(obj, frozenset):
        # Sort frozenset items for deterministic ordering
        return ("__frozenset__", sorted(
            [_canonicalize(item) for item in obj],
            key=lambda x: json.dumps(x, sort_keys=True)
        ))
    elif isinstance(obj, set):
        return ("__set__", sorted(
            [_canonicalize(item) for item in obj],
            key=lambda x: json.dumps(x, sort_keys=True)
        ))
    elif isinstance(obj, tuple):
        return ("__tuple__", [_canonicalize(item) for item in obj])
    elif isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, bytes):
        return ("__bytes__", obj.hex())
    elif hasattr(obj, 'value'):
        # Handle Unhashable class from ComfyUI
        return ("__unhashable__", _canonicalize(getattr(obj, 'value', None)))
    else:
        # For other types, use repr as fallback
        return ("__repr__", repr(obj))


def _serialize_cache_key(cache_key: Any) -> Optional[str]:
    # Returns deterministic SHA256 hex digest, or None on failure.
    # Uses JSON (not pickle) because pickle is non-deterministic across sessions.
    try:
        canonical = _canonicalize(cache_key)
        json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    except Exception as e:
        _logger.warning(f"Failed to serialize cache key: {e}")
        return None


def _contains_nan(obj: Any) -> bool:
    # NaN != NaN so local cache never hits, but serialized NaN would match.
    # Skip external caching for keys containing NaN.
    if isinstance(obj, float):
        try:
            return math.isnan(obj)
        except (TypeError, ValueError):
            return False
    if hasattr(obj, 'value'):  # Unhashable class
        val = getattr(obj, 'value', None)
        if isinstance(val, float):
            try:
                return math.isnan(val)
            except (TypeError, ValueError):
                return False
    if isinstance(obj, (frozenset, tuple, list, set)):
        return any(_contains_nan(item) for item in obj)
    if isinstance(obj, dict):
        return any(_contains_nan(k) or _contains_nan(v) for k, v in obj.items())
    return False


def _estimate_value_size(value: CacheValue) -> int:
    try:
        import torch
    except ImportError:
        return 0

    total = 0

    def estimate(obj):
        nonlocal total
        if isinstance(obj, torch.Tensor):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                estimate(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                estimate(item)

    for output in value.outputs:
        estimate(output)
    return total
