import threading
from collections import OrderedDict

_MAXSIZE = 32

_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_lock = threading.Lock()


def _profile_cache_key(profile) -> tuple:
    return (
        profile.age, profile.height, profile.weight, profile.gender,
        profile.pregnant,
        tuple(sorted(profile.conditions)),
        tuple(sorted(profile.allergies)),
        tuple(sorted(profile.preferences)),
        profile.goal, profile.activity_level, profile.meal_type,
    )


def get_filtered_recipes(real_expert_system, profile) -> dict:
    key = _profile_cache_key(profile)

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    result = real_expert_system.filter_recipes(profile)

    with _lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > _MAXSIZE:
            _cache.popitem(last=False)

    return result


class _CachingExpertSystem:

    def __init__(self, real_expert_system):
        self._real = real_expert_system

    def filter_recipes(self, profile):
        return get_filtered_recipes(self._real, profile)

    def __getattr__(self, name):
        return getattr(self._real, name)


_wrapped = None
_wrap_lock = threading.Lock()


def get_cached_expert_system():
    global _wrapped
    if _wrapped is None:
        with _wrap_lock:
            if _wrapped is None:
                from recipes.services.ai_runtime import get_expert_system
                _wrapped = _CachingExpertSystem(get_expert_system())
    return _wrapped


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def cache_size() -> int:
    with _lock:
        return len(_cache)
