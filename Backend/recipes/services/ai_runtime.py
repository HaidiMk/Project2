import os
import threading

_expert_system = None
_expert_system_lock = threading.Lock()

_explainer_warmed = False
_explainer_lock = threading.Lock()

_nlp_warmed = False
_nlp_lock = threading.Lock()


def _env_flag(name: str) -> bool:
    """Truthy env flags: '1', 'true', 'yes', 'on' (case-insensitive). Default False."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


AI_EAGER_WARMUP = _env_flag("AI_EAGER_WARMUP")
AI_EAGER_EXPLAIN_WARMUP = _env_flag("AI_EAGER_EXPLAIN_WARMUP")
AI_EAGER_NLP_WARMUP = _env_flag("AI_EAGER_NLP_WARMUP")


def _build_expert_system():
    from main import load_data                    
    from engine.filtering_engine import DietaryExpertSystem

    df = load_data()
    return DietaryExpertSystem(df)


def get_expert_system():
    global _expert_system
    if _expert_system is None:
        with _expert_system_lock:
            if _expert_system is None:
                _expert_system = _build_expert_system()
    return _expert_system


def warm_up_explainer() -> None:
    global _explainer_warmed
    if _explainer_warmed:
        return
    with _explainer_lock:
        if not _explainer_warmed:
            from ml.health_classifier.explain import warm_up
            warm_up()
            _explainer_warmed = True


def warm_up_nlp_search() -> None:
    global _nlp_warmed
    if _nlp_warmed:
        return
    with _nlp_lock:
        if not _nlp_warmed:
            from Nlp.query_parser import parse_query
            parse_query("chicken breakfast")
            _nlp_warmed = True