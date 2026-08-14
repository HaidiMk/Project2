import threading

from django.apps import AppConfig


class RecipesConfig(AppConfig):
    name = 'recipes'

    def ready(self):
        """Opt-in eager warmup of the AI runtime singletons.

        Both AI_EAGER_WARMUP and AI_EAGER_EXPLAIN_WARMUP default to unset —
        lazy on-first-request behavior, and ready() is a no-op then. When a
        flag IS set, the warmup runs in a daemon background thread instead of
        blocking Django's startup for several seconds: under `runserver` with
        auto-reload, ready() runs on every reload, and blocking startup for
        the CSV load (seconds) + SHAP JIT (~8 s) each time is poor UX. First
        requests served while loading fall back to the same lazy path, so the
        degraded window is exactly what the default configuration already
        experiences.
        """
        from .services import ai_runtime

        def _background(fn):
            threading.Thread(target=fn, daemon=True).start()

        if ai_runtime.AI_EAGER_WARMUP:
            _background(ai_runtime.get_expert_system)
        if ai_runtime.AI_EAGER_EXPLAIN_WARMUP:
            _background(ai_runtime.warm_up_explainer)
        if ai_runtime.AI_EAGER_NLP_WARMUP:
            _background(ai_runtime.warm_up_nlp_search)