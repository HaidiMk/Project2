from unittest.mock import patch

from django.apps import apps as django_apps
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from profiles.models import UserProfile
from recipes.services import ai_runtime, filter_cache
from recipes.services.profile_translator import (
    PREFERENCE_ALIASES,
    ProfileTranslationError,
    translate_profile,
)


def _make_user_with_profile(username, **profile_overrides):
    user = User.objects.create_user(
        username=username, email=f"{username}@example.com", password="pass12345",
    )
    defaults = dict(
        age=30, height=175, weight=70, gender="male", pregnant=False,
        conditions=[], allergies=[], preferences=[],
        goal="lose_weight", activity_level="moderate", meal_type="standard",
    )
    defaults.update(profile_overrides)
    UserProfile.objects.create(user=user, **defaults)
    return user


class SearchViewTests(TestCase):
    """
    GET /api/recipes/search/ — covers the contract documented in
    recipes/views.py: SearchView (param validation, error paths, meal_type
    precedence, taste_text pass-through, filters_applied, the anti-
    truncation-bias fix, and NLP-model singleton reuse).
    """

    URL = "/api/recipes/search/"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Pay the Expert System's CSV-load cost once for the whole class,
        # the same singleton every real request shares.
        ai_runtime.get_expert_system()

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user_with_profile("searchuser")
        token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
        filter_cache.clear_cache()

    def _get(self, **params):
        return self.client.get(self.URL, params)

    # ------------------------------------------------------------------ #
    # error paths
    # ------------------------------------------------------------------ #
    def test_missing_query_is_400(self):
        self.assertEqual(self._get().status_code, 400)

    def test_blank_query_is_400(self):
        self.assertEqual(self._get(query="   ").status_code, 400)

    def test_invalid_meal_type_is_400(self):
        self.assertEqual(
            self._get(query="chicken", meal_type="brunch").status_code, 400)

    def test_invalid_limit_is_400(self):
        self.assertEqual(
            self._get(query="chicken", limit="abc").status_code, 400)
        self.assertEqual(
            self._get(query="chicken", limit="0").status_code, 400)

    def test_unauthenticated_is_401(self):
        client = APIClient()
        resp = client.get(self.URL, {"query": "chicken"})
        self.assertIn(resp.status_code, (401, 403))

    def test_no_profile_is_404(self):
        user2 = User.objects.create_user(username="noprofile", password="pass12345")
        token2 = Token.objects.create(user=user2)
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION=f"Token {token2.key}")
        resp = client.get(self.URL, {"query": "chicken"})
        self.assertEqual(resp.status_code, 404)

    def test_profile_translation_error_is_400(self):
        self.user.profile.conditions = ["not_a_real_condition"]
        self.user.profile.save()
        resp = self._get(query="chicken")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("conditions", resp.json()["detail"])

    def test_unexpected_ai_error_is_500(self):
        with patch("recipes.views.rank_with_topsis", side_effect=RuntimeError("boom")):
            resp = self._get(query="chicken")
        self.assertEqual(resp.status_code, 500)

    # ------------------------------------------------------------------ #
    # happy paths / contract shape
    # ------------------------------------------------------------------ #
    def test_basic_search_returns_expected_shape(self):
        resp = self._get(query="chicken", limit=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ("count", "total_safe", "total_original", "meal_type",
                    "filters_applied", "results"):
            self.assertIn(key, data)
        self.assertLessEqual(data["count"], 5)
        self.assertEqual(data["filters_applied"]["main_ingredient"], "chicken")

    def test_meal_type_explicit_param_wins_when_no_conflict(self):
        resp = self._get(query="chicken", meal_type="breakfast")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["meal_type"], "breakfast")

    def test_meal_type_falls_back_to_parsed_when_no_param(self):
        resp = self._get(query="breakfast chicken")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["filters_applied"]["meal_type"], "breakfast")
        self.assertEqual(data["meal_type"], "breakfast")

    def test_meal_type_defaults_to_any_with_no_signal(self):
        resp = self._get(query="chicken")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["meal_type"], "any")

    def test_taste_text_adds_taste_score_and_changes_blend(self):
        resp_no_taste = self._get(query="chicken", limit=10)
        resp_taste = self._get(
            query="chicken", limit=10, taste_text="I love garlic, dislike seafood")
        self.assertEqual(resp_no_taste.status_code, 200)
        self.assertEqual(resp_taste.status_code, 200)
        results_taste = resp_taste.json()["results"]
        results_no_taste = resp_no_taste.json()["results"]
        self.assertTrue(results_taste)
        self.assertTrue(all("_taste_score" in r for r in results_taste))
        self.assertTrue(all("_taste_score" not in r for r in results_no_taste))

    def test_nlp_filters_only_narrow_results(self):
        # NLP query refinement can only narrow the Expert System's safe set,
        # never widen it (NFR-010) — total_safe for a constrained query must
        # be <= the unconstrained recommendations total for the same profile.
        reco = self.client.get("/api/recipes/recommendations/")
        search = self._get(query="low sugar breakfast")
        self.assertEqual(reco.status_code, 200)
        self.assertEqual(search.status_code, 200)
        self.assertLessEqual(search.json()["total_safe"], reco.json()["total_safe"])

    # ------------------------------------------------------------------ #
    # anti-truncation-bias regression (the reason for _NLP_PRERANK_LIMIT)
    # ------------------------------------------------------------------ #
    def test_full_filtered_set_passed_to_topsis_not_pre_truncated_by_limit(self):
        with patch("recipes.views.rank_with_topsis") as mock_rank:
            mock_rank.side_effect = lambda df, goal, user_taste_text=None: df.assign(
                final_score=1.0, _topsis_score=1.0, _ai_health_score=1.0,
            )
            resp = self._get(query="chicken", limit=3)
        self.assertEqual(resp.status_code, 200)
        passed_df = mock_rank.call_args[0][0]
        total_safe = resp.json()["total_safe"]
        # rank_with_topsis must see the FULL filtered set (== total_safe),
        # not just `limit` rows pre-selected by _expert_score.
        self.assertEqual(len(passed_df), total_safe)
        self.assertGreater(len(passed_df), 3)

    # ------------------------------------------------------------------ #
    # NLP Sentence-BERT singleton reuse
    # ------------------------------------------------------------------ #
    def test_nlp_model_singleton_reused_across_requests(self):
        import Nlp.query_parser as query_parser

        self._get(query="chicken breakfast")
        self.assertIsNotNone(query_parser._MODEL)

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            resp = self._get(query="beef dinner")
            self.assertEqual(resp.status_code, 200)
            mock_st.assert_not_called()

    # ------------------------------------------------------------------ #
    # RecommendationsView smoke test — guards the shared json_safe refactor
    # ------------------------------------------------------------------ #
    def test_recommendations_still_works_after_json_safe_extraction(self):
        resp = self.client.get("/api/recipes/recommendations/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ("count", "total_safe", "total_original", "meal_type", "results"):
            self.assertIn(key, data)

    # ------------------------------------------------------------------ #
    # filter_cache integration — repeat requests for the same profile
    # signature + meal_type must not recompute filter_recipes()
    # ------------------------------------------------------------------ #
    def test_repeat_search_requests_hit_the_cache(self):
        real_system = ai_runtime.get_expert_system()
        with patch.object(real_system, "filter_recipes",
                           wraps=real_system.filter_recipes) as spy:
            resp1 = self._get(query="chicken", meal_type="dinner")
            resp2 = self._get(query="beef", meal_type="dinner")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)
        # Same profile signature + meal_type -> filter_recipes() computed
        # once even though the query text differs between requests.
        self.assertEqual(spy.call_count, 1)

    def test_cache_shared_between_search_and_recommendations(self):
        real_system = ai_runtime.get_expert_system()
        with patch.object(real_system, "filter_recipes",
                           wraps=real_system.filter_recipes) as spy:
            self.client.get("/api/recipes/recommendations/")
            self._get(query="chicken")
        self.assertEqual(spy.call_count, 1)


class NlpWarmupTests(TestCase):
    """AI_EAGER_NLP_WARMUP flag + warm_up_nlp_search() singleton trigger."""

    def test_warm_up_nlp_search_is_idempotent(self):
        previous = ai_runtime._nlp_warmed
        ai_runtime._nlp_warmed = False
        try:
            with patch("Nlp.query_parser.parse_query") as mock_parse:
                ai_runtime.warm_up_nlp_search()
                ai_runtime.warm_up_nlp_search()
                self.assertEqual(mock_parse.call_count, 1)
        finally:
            ai_runtime._nlp_warmed = previous

    def test_apps_ready_triggers_nlp_warmup_when_flag_set(self):
        config = django_apps.get_app_config("recipes")
        with patch.object(ai_runtime, "AI_EAGER_NLP_WARMUP", True), \
             patch.object(ai_runtime, "AI_EAGER_WARMUP", False), \
             patch.object(ai_runtime, "AI_EAGER_EXPLAIN_WARMUP", False), \
             patch("recipes.apps.threading.Thread") as mock_thread:
            config.ready()
            targets = [c.kwargs.get("target") for c in mock_thread.call_args_list]
            self.assertIn(ai_runtime.warm_up_nlp_search, targets)

    def test_apps_ready_does_not_trigger_nlp_warmup_when_flag_unset(self):
        config = django_apps.get_app_config("recipes")
        with patch.object(ai_runtime, "AI_EAGER_NLP_WARMUP", False), \
             patch.object(ai_runtime, "AI_EAGER_WARMUP", False), \
             patch.object(ai_runtime, "AI_EAGER_EXPLAIN_WARMUP", False), \
             patch("recipes.apps.threading.Thread") as mock_thread:
            config.ready()
            mock_thread.assert_not_called()


def _fake_expert_system(compute_result=None):
    """A minimal duck-typed stand-in for DietaryExpertSystem: only
    .filter_recipes() is ever called on it by filter_cache."""
    fake = type("FakeExpertSystem", (), {})()
    calls = []

    def filter_recipes(profile):
        calls.append(profile)
        return compute_result(profile, len(calls)) if compute_result else {"n": len(calls)}

    fake.filter_recipes = filter_recipes
    fake.calls = calls
    return fake


def _ai_profile(**overrides):
    from core.user_profile import UserProfile as AIUserProfile
    defaults = dict(age=30, height=175, weight=70, gender="male",
                     goal="weight_loss", activity_level="moderate")
    defaults.update(overrides)
    return AIUserProfile(**defaults)


class FilterCacheTests(TestCase):
    """Unit tests for recipes/services/filter_cache.py in isolation — no
    HTTP layer, no real Expert System, so these run instantly."""

    def setUp(self):
        filter_cache.clear_cache()

    def tearDown(self):
        filter_cache.clear_cache()

    def test_second_call_with_identical_signature_is_a_cache_hit(self):
        fake = _fake_expert_system()
        r1 = filter_cache.get_filtered_recipes(fake, _ai_profile())
        r2 = filter_cache.get_filtered_recipes(fake, _ai_profile())  # new object, same values
        self.assertEqual(len(fake.calls), 1)
        self.assertIs(r1, r2)

    def test_different_meal_type_is_a_cache_miss(self):
        fake = _fake_expert_system()
        filter_cache.get_filtered_recipes(fake, _ai_profile(meal_type="breakfast"))
        filter_cache.get_filtered_recipes(fake, _ai_profile(meal_type="dinner"))
        self.assertEqual(len(fake.calls), 2)

    def test_different_goal_is_a_cache_miss(self):
        fake = _fake_expert_system()
        filter_cache.get_filtered_recipes(fake, _ai_profile(goal="weight_loss"))
        filter_cache.get_filtered_recipes(fake, _ai_profile(goal="muscle_gain"))
        self.assertEqual(len(fake.calls), 2)

    def test_condition_list_order_does_not_affect_cache_key(self):
        fake = _fake_expert_system()
        filter_cache.get_filtered_recipes(
            fake, _ai_profile(conditions=["diabetes", "hypertension"]))
        filter_cache.get_filtered_recipes(
            fake, _ai_profile(conditions=["hypertension", "diabetes"]))
        self.assertEqual(len(fake.calls), 1)  # sorted() makes order irrelevant

    def test_lru_eviction_bounds_cache_size(self):
        fake = _fake_expert_system(compute_result=lambda profile, n: {"age": profile.age})
        for age in range(1, filter_cache._MAXSIZE + 10):
            filter_cache.get_filtered_recipes(fake, _ai_profile(age=age))
        self.assertEqual(filter_cache.cache_size(), filter_cache._MAXSIZE)

    def test_caching_expert_system_proxy_delegates_and_caches(self):
        fake = _fake_expert_system()
        fake.corpus_size = 384541  # arbitrary passthrough attribute
        proxy = filter_cache._CachingExpertSystem(fake)

        proxy.filter_recipes(_ai_profile())
        proxy.filter_recipes(_ai_profile())  # same signature -> cache hit

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(proxy.corpus_size, 384541)  # __getattr__ fallback

    def test_get_cached_expert_system_is_a_process_wide_singleton(self):
        a = filter_cache.get_cached_expert_system()
        b = filter_cache.get_cached_expert_system()
        self.assertIs(a, b)


class PreferenceAliasTests(TestCase):
    """profile_translator.py's PREFERENCE_ALIASES — the 10 additions from
    the alias-map investigation, plus a regression check on the pre-existing
    aliases and the deliberately-unmapped allergen phrases."""

    NEW_ALIASES = {
        "low carb":       "low_carb",
        "high protein":   "no_preference",
        "low sugar":      "no_preference",
        "healthy":        "no_preference",
        "healthy eating": "no_preference",
        "omnivore":       "no_preference",
        "pescatarian":    "seafood_lover",
        "pescetarian":    "seafood_lover",
        "chicken only":   "chicken_lover",
        "poultry only":   "chicken_lover",
    }

    EXISTING_ALIASES = {
        "keto":               "low_carb",
        "lowcarb":            "low_carb",
        "low-carb":           "low_carb",
        "mediterranean diet": "mediterranean",
        "no preference":      "no_preference",
        "none":               "no_preference",
    }

    def _django_profile(self, preferences):
        return UserProfile(
            age=30, height=175, weight=70, gender="male", pregnant=False,
            conditions=[], allergies=[], preferences=preferences,
            goal="lose_weight", activity_level="moderate", meal_type="standard",
        )

    def test_new_aliases_are_registered_with_the_expected_canonical_value(self):
        for phrase, expected in self.NEW_ALIASES.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(PREFERENCE_ALIASES.get(phrase), expected)

    def test_all_ten_new_alias_phrases_translate_successfully(self):
        for phrase, expected in self.NEW_ALIASES.items():
            with self.subTest(phrase=phrase):
                ai_profile = translate_profile(self._django_profile([phrase]))
                self.assertIn(expected, ai_profile.preferences)

    def test_existing_aliases_still_work(self):
        for phrase, expected in self.EXISTING_ALIASES.items():
            with self.subTest(phrase=phrase):
                ai_profile = translate_profile(self._django_profile([phrase]))
                self.assertIn(expected, ai_profile.preferences)

    def test_gluten_dairy_nut_free_are_deliberately_left_unmapped(self):
        for phrase in ("gluten free", "dairy free", "nut free"):
            with self.subTest(phrase=phrase):
                with self.assertRaises(ProfileTranslationError):
                    translate_profile(self._django_profile([phrase]))
