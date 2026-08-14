"""
filter_cache.py — Backend-side mitigation for DietaryExpertSystem.filter_recipes()'s
per-request cost.

BACKGROUND (see the performance investigation): filter_recipes() computes
_expert_score and _reason via two row-wise pandas .apply() calls inside
Expert System/engine/scorer.py — pure-Python, ~200-300k rows per call,
~50+ seconds on a cold profile. That is Expert System core logic and is
explicitly out of scope to modify here. This module is a mitigation, not a
fix: it avoids RE-PAYING that cost on every request, without touching
Expert System/ or Nlp/ at all.

WHY THIS IS SAFE TO CACHE
    filter_recipes(profile) is a pure function of (self.df, profile) for the
    lifetime of a process:
      - self.df (the 384,541-row corpus) never changes after the
        DietaryExpertSystem singleton is built.
      - Its output does NOT depend on free-text search filters at all
        (sugar_max, main_ingredient, ...) — those are applied by
        Nlp/pipeline.py as a pure post-filter on filter_recipes()'s
        already-returned safe_recipes, never fed back into it.
    So the result is safe to cache, keyed on exactly the profile fields (+
    meal_type) that influence it — nothing else.

WHY A PROXY, NOT JUST A WRAPPED FUNCTION
    Nlp/pipeline.py's search_recipes() calls `expert_system.filter_recipes(
    merged_profile)` directly on whatever object is passed in as
    `expert_system` — it is not ours to modify. _CachingExpertSystem is a
    duck-typed proxy exposing the same .filter_recipes(profile) interface;
    passing it into search_recipes() intercepts that call transparently, so
    SearchView benefits from the cache with zero changes to Nlp/.
    RecommendationsView calls .filter_recipes() the same way, directly.

Thread-safe, bounded LRU (default 32 entries): each entry holds a
~200-300k-row DataFrame, so the cache is deliberately small to bound memory
rather than sized for a large fleet of distinct profiles. The lock only
guards dict mutation, not the (slow) call to the real filter_recipes(), so
concurrent misses on the same new key may both compute — an accepted
trade-off for a mitigation, not a correctness issue.
"""

import threading
from collections import OrderedDict

_MAXSIZE = 32

_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_lock = threading.Lock()


def _profile_cache_key(profile) -> tuple:
    """Exactly the profile fields (+ meal_type) that influence
    filter_recipes()'s output — nothing query-specific."""
    return (
        profile.age, profile.height, profile.weight, profile.gender,
        profile.pregnant,
        tuple(sorted(profile.conditions)),
        tuple(sorted(profile.allergies)),
        tuple(sorted(profile.preferences)),
        profile.goal, profile.activity_level, profile.meal_type,
    )


def get_filtered_recipes(real_expert_system, profile) -> dict:
    """Thread-safe LRU cache around DietaryExpertSystem.filter_recipes()."""
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
    """Duck-typed proxy: same .filter_recipes(profile) interface as
    DietaryExpertSystem, served from the cache above. Any other attribute
    access falls through to the real instance unchanged."""

    def __init__(self, real_expert_system):
        self._real = real_expert_system

    def filter_recipes(self, profile):
        return get_filtered_recipes(self._real, profile)

    def __getattr__(self, name):
        return getattr(self._real, name)


_wrapped = None
_wrap_lock = threading.Lock()


def get_cached_expert_system():
    """Process-wide singleton _CachingExpertSystem wrapping
    ai_runtime.get_expert_system(). Use this in place of
    ai_runtime.get_expert_system() anywhere filter_recipes() would
    otherwise be called directly or handed to Nlp/pipeline.py."""
    global _wrapped
    if _wrapped is None:
        with _wrap_lock:
            if _wrapped is None:
                from recipes.services.ai_runtime import get_expert_system
                _wrapped = _CachingExpertSystem(get_expert_system())
    return _wrapped


def clear_cache() -> None:
    """Test/ops helper: drop all cached entries."""
    with _lock:
        _cache.clear()


def cache_size() -> int:
    with _lock:
        return len(_cache)
