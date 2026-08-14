"""
AI runtime layer for the recipes app.

Single well-known place for the heavy AI resources used by the meal-planning
endpoints (built in the next step). Everything here is LAZY by default: the
384,541-row recipe CSV is loaded and the DietaryExpertSystem is constructed on
the FIRST call to get_expert_system() (several seconds, as documented in the AI
project), and the SHAP explainer pays its one-time ~8 s numba-JIT warmup on the
first warm_up_explainer() call. Subsequent calls return the cached object /
no-op instantly, so the cost is paid exactly once per process.

Two opt-in environment flags (both default to UNSET = lazy behavior):

    AI_EAGER_WARMUP
        If set to a truthy value ("1", "true", "yes", "on" — case-insensitive),
        the expert system is loaded at Django startup from recipes/apps.py's
        ready() instead of on the first request.
    AI_EAGER_EXPLAIN_WARMUP
        Same idea for the SHAP explainer warmup (one-time ~8 s cost).
    AI_EAGER_NLP_WARMUP
        Same idea for the Nlp/ search-query parser's Sentence-BERT model
        (all-MiniLM-L6-v2). Requires internet access the first time it runs,
        same as documented in Nlp/query_parser.py.

Usage::

    from recipes.services.ai_runtime import get_expert_system, warm_up_explainer

    system = get_expert_system()      # first call loads the CSV (~seconds)
    result = system.filter_recipes(profile)
"""

import os
import threading

# ---------------------------------------------------------------------------
# module-level singletons + their guards
# ---------------------------------------------------------------------------

_expert_system = None
_expert_system_lock = threading.Lock()

_explainer_warmed = False
_explainer_lock = threading.Lock()

_nlp_warmed = False
_nlp_lock = threading.Lock()


def _env_flag(name: str) -> bool:
    """Truthy env flags: '1', 'true', 'yes', 'on' (case-insensitive). Default False."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Public flag values, read once at import time so apps.py and callers agree.
AI_EAGER_WARMUP = _env_flag("AI_EAGER_WARMUP")
AI_EAGER_EXPLAIN_WARMUP = _env_flag("AI_EAGER_EXPLAIN_WARMUP")
AI_EAGER_NLP_WARMUP = _env_flag("AI_EAGER_NLP_WARMUP")


def _build_expert_system():
    """Reuse the AI project's own construction path (TOPSIS/main.py, TOPSIS/
    sanity_check.py do exactly this): load_data() from Expert System's main
    module, then DietaryExpertSystem(df). Heavy imports stay inside this
    function so importing ai_runtime itself is cheap."""
    from main import load_data                      # Expert System/main.py
    from engine.filtering_engine import DietaryExpertSystem

    df = load_data()
    return DietaryExpertSystem(df)


def get_expert_system():
    """Thread-safe lazy singleton for DietaryExpertSystem.

    First call loads the 384,541-row CSV and constructs the expert system
    (several seconds). Concurrent callers during that first load are blocked on
    the lock and receive the SAME instance once it is built. Later calls return
    the cached instance immediately."""
    global _expert_system
    if _expert_system is None:
        with _expert_system_lock:
            if _expert_system is None:
                _expert_system = _build_expert_system()
    return _expert_system


def warm_up_explainer() -> None:
    """One-time SHAP explainer warmup (approx. 8 s, numba JIT).

    Safe to call any number of times: the AI project's explain.warm_up() is
    invoked at most once per process, and only after it succeeds is the flag
    set — a failed attempt is retried on the next call."""
    global _explainer_warmed
    if _explainer_warmed:
        return
    with _explainer_lock:
        if not _explainer_warmed:
            from ml.health_classifier.explain import warm_up
            warm_up()
            _explainer_warmed = True


def warm_up_nlp_search() -> None:
    """One-time warmup of the Nlp/ search-query parser's Sentence-BERT model.

    Calls the module's real public entry point (parse_query) with a short
    phrase that reaches the semantic-matching step, so the same lazy
    singleton Nlp/query_parser.py already owns gets loaded here instead of
    on the first real search request. The singleton itself stays inside
    Nlp/ — this function only triggers it early.

    Safe to call any number of times: warmed at most once per process, and
    only after it succeeds is the flag set — a failed attempt (e.g. no
    internet on first run) is retried on the next call."""
    global _nlp_warmed
    if _nlp_warmed:
        return
    with _nlp_lock:
        if not _nlp_warmed:
            from Nlp.query_parser import parse_query
            parse_query("chicken breakfast")
            _nlp_warmed = True