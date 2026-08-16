import threading

from django.apps import AppConfig


class RecipesConfig(AppConfig):
    name = 'recipes'

    def ready(self):

        from .services import ai_runtime

        def _background(fn):
            threading.Thread(target=fn, daemon=True).start()

        if ai_runtime.AI_EAGER_WARMUP:
            _background(ai_runtime.get_expert_system)
        if ai_runtime.AI_EAGER_EXPLAIN_WARMUP:
            _background(ai_runtime.warm_up_explainer)
        if ai_runtime.AI_EAGER_NLP_WARMUP:
            _background(ai_runtime.warm_up_nlp_search)