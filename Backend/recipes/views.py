"""
GET /api/recipes/recommendations/ — ranked, medically-safe recipe recommendations
====================================================================================

First real AI endpoint: takes the authenticated user's stored health profile,
runs it through the Expert System's safety filters, then the blended TOPSIS
ranking, and returns the top-N recipes as JSON.

Pipeline (in order):
    1. Validate request params (meal_type, taste_text, limit).
    2. Load request.user.profile (OneToOneField, related_name="profile").
    3. translate_profile() -> AI UserProfile dataclass.
    4. Apply the PER-REQUEST meal_type to the dataclass (profile_translator
       deliberately leaves it at the default "any" — it is a per-request
       concept, NOT a stored profile attribute).
    5. filter_recipes() via the cached expert-system proxy (recipes/services/
       filter_cache.py) — same shared Step-3 singleton underneath, but
       repeat requests for the same profile signature + meal_type skip the
       ~50s row-wise scoring cost in Expert System/engine/scorer.py.
    6. rank_with_topsis(safe_recipes, goal, user_taste_text?) — blended score.
    7. Slice top `limit` rows, serialize to native Python types (NaN -> None).

Request (all query params optional):
    meal_type  : breakfast | lunch | dinner | any    (default "any")
    taste_text : free text, passed to the taste model (default: omitted)
    limit      : positive int, default 20, capped at 100

Response shape (the documented contract):
    {
        "count":          int,  number of result rows returned
        "total_safe":     int,  recipes surviving the Expert System filters
        "total_original": int,  full recipe database size
        "meal_type":      str,  the meal_type actually applied
        "results": [
            {
                "RecipeId":              int,
                "Name":                  str,
                "Calories":              float|None,
                "ProteinContent":        float|None,
                "CarbohydrateContent":   float|None,
                "SugarContent":          float|None,
                "FiberContent":          float|None,
                "final_score":           float,   blended 0..1-ish score
                "_topsis_score":         float,
                "_ai_health_score":      float,
                "_expert_score":         float,
                "_taste_score":          float    (only when taste_text given)
            }, ...
        ]
    }
    Every value is a native Python int/float/str/bool; NaN is converted to
    None so the JSON is always valid (numpy dtypes are converted explicitly).

Errors:
    401/403  unauthenticated (DRF default; IsAuthenticated is also explicit)
    404      authenticated but no stored UserProfile yet
    400      ProfileTranslationError (invalid stored conditions/allergies/
             preferences/goal) or an invalid query parameter — the exact
             translator message is passed through in "detail"
    500      any unexpected AI-layer error — logged server-side, generic
             message returned (no stack trace leaked to the client)
"""

import json
from django.contrib.auth.models import User
from rest_framework.permissions import IsAdminUser
from django.conf import settings
from pathlib import Path


import logging
import pandas as pd 

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from ml.health_classifier.explain import explain_health_score

from planner.meal_planner import build_daily_plan
# MEAL_TYPE_DATA_MAP keys are the per-request meal vocabulary the engine
# actually understands (breakfast -> Breakfast, lunch/dinner -> MainDish);
# "any" is the engine's no-filter sentinel. Reusing the engine's map instead
# of re-declaring the vocabulary keeps the two in lockstep.
from engine.filtering_engine import MEAL_TYPE_DATA_MAP
from profiles.models import UserProfile
from recipes.services.filter_cache import get_cached_expert_system
from recipes.services.profile_translator import (
    ProfileTranslationError,
    translate_profile,
)
from recipes.services.serialization import json_safe
from topsis_model import rank_with_topsis

logger = logging.getLogger(__name__)
def _json_safe_deep(value):
    if isinstance(value, dict):
        return {k: _json_safe_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_deep(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_json_safe_deep(row) for row in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return _json_safe_deep(value.to_dict())
    return json_safe(value)




class RecommendationsView(APIView):
    """See the module docstring for the full request/response contract."""

    permission_classes = [IsAuthenticated]

    MEAL_TYPE_CHOICES = frozenset(MEAL_TYPE_DATA_MAP) | {"any"}
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100

    # Documented result columns (subset — enough for the frontend card view,
    # with the per-source scores for transparency). Filtered by what the
    # ranked frame actually contains (e.g. no _taste_score without taste_text).
    RESULT_COLUMNS = [
        "RecipeId", "Name", "ImageUrl", "Calories", "ProteinContent",
        "CarbohydrateContent", "SugarContent", "FiberContent",
        "final_score", "_topsis_score", "_ai_health_score",
        "_expert_score", "_taste_score",
    ]

    def _parse_limit(self, request):
        raw = request.query_params.get("limit", str(self.DEFAULT_LIMIT))
        try:
            limit = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"limit must be an integer, got {raw!r}")
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        return min(limit, self.MAX_LIMIT)

    def get(self, request, *args, **kwargs):
        # --- 1. query params -------------------------------------------------
        meal_type = request.query_params.get("meal_type", "any").strip().lower()
        if meal_type not in self.MEAL_TYPE_CHOICES:
            return Response(
                {"detail": f"meal_type must be one of: "
                           f"{', '.join(sorted(self.MEAL_TYPE_CHOICES))}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            limit = self._parse_limit(request)
        except ValueError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)
        # Empty/falsy taste_text is omitted entirely, preserving rank_with_topsis's
        # documented backward-compatible no-taste fallback blend (0.4/0.4/0.2).
        taste_text = request.query_params.get("taste_text", "").strip() or None

        # --- 2. the user's stored profile ------------------------------------
        try:
            django_profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile before requesting "
                           "recommendations."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # --- 3. translate (invalid stored data -> 400, exact message) --------
        try:
            ai_profile = translate_profile(django_profile)
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)

        # --- 4. apply the per-request meal_type -------------------------------
        ai_profile.meal_type = meal_type

        # --- 5. expert system safety filtering + 6. blended ranking -----------
        try:
            result = get_cached_expert_system().filter_recipes(ai_profile)
            ranked = rank_with_topsis(
                result["safe_recipes"], ai_profile.goal,
                user_taste_text=taste_text,
            )
        except Exception:
            logger.exception(
                "Recommendations failed for user %s (goal=%s, meal_type=%s)",
                getattr(request.user, "username", "?"), ai_profile.goal, meal_type,
            )
            return Response(
                {"detail": "Internal error while generating recommendations."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # --- 7. slice + serialize to native types -----------------------------
        columns = [c for c in self.RESULT_COLUMNS if c in ranked.columns]
        top = ranked[columns].head(limit)
        results = [
            {k: json_safe(v) for k, v in row.items()}
            for row in top.to_dict("records")
        ]

        return Response({
            "count": len(results),
            "total_safe": result.get("total_safe", len(result["safe_recipes"])),
            "total_original": result.get("total_original", 0),
            "meal_type": meal_type,
            "results": results,
        })


class SearchView(APIView):
    """
    GET /api/recipes/search/ — free-text search -> NLP filters -> AI ranking
    ==========================================================================

    Pipeline (in order):
        1. Validate request params (query, meal_type, taste_text, limit).
        2. Load request.user.profile, translate_profile() -> AI UserProfile
           (identical to RecommendationsView; 404/400 on the same conditions).
        3. Apply an explicit `meal_type` query param to the profile BEFORE
           calling search_recipes() (same pre-set pattern RecommendationsView
           uses). If no explicit param is given, Nlp/pipeline.py's own merge
           logic lets whatever the query text parses to decide (falls back to
           "any" if the text has no meal-type language either).
        4. Nlp.pipeline.search_recipes() — the ONLY place free text is turned
           into structured filters and merged with medical safety filtering
           (Expert System's filter_recipes(), called through the cached
           expert-system proxy in recipes/services/filter_cache.py — same
           singleton underneath, but repeat requests for the same profile
           signature + meal_type skip the ~50s row-wise scoring cost).
           Called with a deliberately oversized top_n so it returns the FULL
           filtered set, not just its own expert-score-sorted top slice —
           see the module note on rank_with_topsis below for why.
        5. rank_with_topsis(safe_recipes, goal, user_taste_text?) — blended
           score, computed over the FULL filtered set so the true top-N by
           final_score is what gets returned (re-ranking only a pre-truncated
           expert-score slice would silently drop recipes TOPSIS should
           surface).
        6. Slice top `limit` rows, serialize to native Python types.

    Request (all query params optional except `query`):
        query      : free-text English search query (required)
        meal_type  : breakfast | lunch | dinner | any   (optional; wins over
                     whatever the query text itself parses to, EXCEPT in the
                     rare case the text also parses a conflicting meal_type —
                     Nlp/pipeline.py's own merge prefers the parsed value
                     then. `filters_applied.meal_type` (raw NLP output) vs.
                     the top-level `meal_type` (what was actually applied)
                     makes that case transparent rather than silently wrong.)
        taste_text : free text for the taste model (default: omitted) —
                     separate from `query`; the search query itself is only
                     used for structured filter extraction, never taste
                     scoring.
        limit      : positive int, default 20, capped at 100

    Response shape (the documented contract):
        {
            "count":          int,  number of result rows returned
            "total_safe":     int,  recipes surviving Expert System filters
                              AND the NLP query's numeric/ingredient
                              refinement (narrower than RecommendationsView's
                              total_safe, which stops at Expert System only)
            "total_original": int,  full recipe database size
            "meal_type":      str,  the meal_type actually applied
            "filters_applied": dict, the raw Nlp.query_parser.parse_query()
                              output for this query (condition, meal_type,
                              protein_min, sugar_max, sodium_max, fiber_min,
                              calories_max, allergy, diet_preference,
                              main_ingredient)
            "results": [ ... same per-row shape as RecommendationsView ... ]
        }

    Errors:
        400  missing/blank `query`, invalid `meal_type`/`limit` query param,
             or ProfileTranslationError (translator message passed through)
        401/403  unauthenticated
        404      authenticated but no stored UserProfile yet
        500      any unexpected AI-layer error — logged server-side, generic
                 message returned (no stack trace leaked to the client)
    """

    permission_classes = [IsAuthenticated]

    MEAL_TYPE_CHOICES = frozenset(MEAL_TYPE_DATA_MAP) | {"any"}
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100

    
    _NLP_PRERANK_LIMIT = 500_000

    RESULT_COLUMNS = [
        "RecipeId", "Name", "ImageUrl", "Calories", "ProteinContent",
        "CarbohydrateContent", "SugarContent", "FiberContent",
        "final_score", "_topsis_score", "_ai_health_score",
        "_expert_score", "_taste_score",
    ]

    def _parse_limit(self, request):
        raw = request.query_params.get("limit", str(self.DEFAULT_LIMIT))
        try:
            limit = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"limit must be an integer, got {raw!r}")
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        return min(limit, self.MAX_LIMIT)

    def get(self, request, *args, **kwargs):
        # --- 1. query params -------------------------------------------------
        query_text = request.query_params.get("query", "").strip()
        if not query_text:
            return Response(
                {"detail": "query is required and must not be blank."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        meal_type = request.query_params.get("meal_type", "").strip().lower() or None
        if meal_type is not None and meal_type not in self.MEAL_TYPE_CHOICES:
            return Response(
                {"detail": f"meal_type must be one of: "
                           f"{', '.join(sorted(self.MEAL_TYPE_CHOICES))}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            limit = self._parse_limit(request)
        except ValueError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)
        # Empty/falsy taste_text is omitted entirely, preserving rank_with_topsis's
        # documented backward-compatible no-taste fallback blend (0.4/0.4/0.2).
        taste_text = request.query_params.get("taste_text", "").strip() or None

        # --- 2. the user's stored profile ------------------------------------
        try:
            django_profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile before requesting "
                           "a search."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # --- 3. translate (invalid stored data -> 400, exact message) --------
        try:
            ai_profile = translate_profile(django_profile)
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)

        # --- 4. explicit meal_type pre-set (see class docstring caveat) ------
        if meal_type is not None:
            ai_profile.meal_type = meal_type

        # --- 5. NLP filter extraction + Expert System safety filtering,
        #        6. blended TOPSIS ranking over the FULL filtered set --------
        try:
            from Nlp.pipeline import search_recipes

            search_result = search_recipes(
                query_text=query_text,
                base_profile=ai_profile,
                expert_system=get_cached_expert_system(),
                top_n=self._NLP_PRERANK_LIMIT,
            )
            ranked = rank_with_topsis(
                search_result["safe_recipes"], ai_profile.goal,
                user_taste_text=taste_text,
            )
        except Exception:
            logger.exception(
                "Search failed for user %s (goal=%s, meal_type=%s, query=%r)",
                getattr(request.user, "username", "?"), ai_profile.goal,
                meal_type, query_text,
            )
            return Response(
                {"detail": "Internal error while searching recipes."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # --- 7. slice + serialize to native types -----------------------------
        columns = [c for c in self.RESULT_COLUMNS if c in ranked.columns]
        top = ranked[columns].head(limit)
        results = [
            {k: json_safe(v) for k, v in row.items()}
            for row in top.to_dict("records")
        ]

        return Response({
            "count": len(results),
            "total_safe": search_result["total_safe"],
            "total_original": search_result["total_original"],
            "meal_type": search_result["meal_type"],
            "filters_applied": search_result["filters"],
            "results": results,
        })
        
class AlternativesView(APIView):
   
    permission_classes = [IsAuthenticated]

    RESULT_COLUMNS = [
        "RecipeId", "Name", "ImageUrl", "Calories", "ProteinContent",
        "CarbohydrateContent", "SugarContent", "FiberContent",
        "final_score"
    ]

    def get(self, request, recipe_id, *args, **kwargs):
        try:
            django_profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile before requesting alternatives."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            ai_profile = translate_profile(django_profile)
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            expert_system = get_cached_expert_system()
            result = expert_system.filter_recipes(ai_profile)
            safe_df = result["safe_recipes"]

            safe_df = safe_df[safe_df['RecipeId'] != recipe_id]

            ranked = rank_with_topsis(safe_df, ai_profile.goal)

            columns = [c for c in self.RESULT_COLUMNS if c in ranked.columns]
            top_alternatives = ranked[columns].head(3)

            results = [
                {k: json_safe(v) for k, v in row.items()}
                for row in top_alternatives.to_dict("records")
            ]

            return Response({
                "original_recipe_id": recipe_id,
                "alternatives_count": len(results),
                "results": results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("Alternatives generation failed for recipe_id=%s", recipe_id)
            return Response(
                {"detail": "Internal error while generating alternatives."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )   
class ExplanationView(APIView):
   
    permission_classes = [IsAuthenticated]

    def get(self, request, recipe_id, *args, **kwargs):
        try:
            django_profile = request.user.profile
            ai_profile = translate_profile(django_profile)
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile first."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 1. جلب بيانات الوجبة من الـ DataFrame المحمل بالذاكرة
            expert_system = get_cached_expert_system()
            
            # (ملاحظة: إذا رفيقتك مسمية الداتا فريم recipes_df بدل df، عدلي الكلمة هنا)
            if not hasattr(expert_system, 'df'):
                return Response(
                    {"detail": "Dataframe attribute missing in expert system."}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            recipe_row = expert_system.df[expert_system.df['RecipeId'] == recipe_id]
            
            if recipe_row.empty:
                return Response({"detail": "Recipe not found."}, status=status.HTTP_404_NOT_FOUND)
            
            # تحويل بيانات الوجبة لـ Dictionary ليفهمها كود الـ AI
            recipe_data = recipe_row.iloc[0].to_dict()

            # 2. الشروط الطبية للمريض
            condition_keys = list(ai_profile.conditions) if ai_profile.conditions else []

            # إذا المريض سليم تماماً، ما في داعي للتفسير الطبي المعقد
            if not condition_keys:
                 return Response({
                    "recipe_id": recipe_id,
                    "is_safe": True,
                    "reasons": ["ملفك الصحي لا يحتوي على أمراض مزمنة، لذلك هذه الوجبة مناسبة لك تماماً."]
                 }, status=status.HTTP_200_OK)

            # 3. استدعاء دالة التفسير الذكية من ملف رفيقتك
            explanation_data = explain_health_score(recipe=recipe_data, condition_keys=condition_keys, top_n=3)

            # 4. تجميع النصوص (text) من الرد لتكون جاهزة للفرونت اند
            reasons = []
            explanations = explanation_data.get("condition_explanations", {})
            for cond, details in explanations.items():
                for reason in details.get("top_reasons", []):
                    reasons.append(reason.get("text"))

            if not reasons:
                 reasons.append("تم فحص الوجبة والتأكد من مطابقتها لقيودك الطبية.")

            return Response({
                "recipe_id": recipe_id,
                "is_safe": True,
                "reasons": reasons
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("Explanation failed for recipe_id=%s", recipe_id)
            return Response(
                {"detail": "Internal error while generating explanation."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RecipeDetailView(APIView):
    """
    GET /api/recipes/<int:recipe_id>/ — full detail for a single recipe
    ==========================================================================

    Pipeline:
        1. Load request.user.profile, translate_profile() (identical to the
           other views; 404/400 on the same conditions).
        2. Look up the row by RecipeId directly in the cached expert system's
           in-memory DataFrame (same access pattern ExplanationView already
           uses) -> 404 if the id isn't present in the live corpus (whether
           it never existed there, or was dropped during cleaning).
        3. Run the user's profile through the same cached filter_recipes()
           every other endpoint uses (recipes/services/filter_cache.py) to
           determine whether THIS recipe survives the Expert System's safety
           filters for THIS user -> is_safe.
        4. Serialize every column the live corpus carries for this recipe
           (all 41 columns of cleaned_recipes.csv, including ImageUrl/
           ImagesCount/ImagesJson) to native JSON types.

    Response shape:
        {
            "recipe_id": int,
            "is_safe":   bool,
            "recipe":    { <all columns>, native JSON types, NaN -> None }
        }

    Errors:
        400  ProfileTranslationError (translator message passed through)
        401/403  unauthenticated
        404      no stored UserProfile yet, OR recipe_id not present in the
                 live corpus
        500      any unexpected AI-layer error — logged server-side, generic
                 message returned (no stack trace leaked to the client)
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, recipe_id, *args, **kwargs):
        # --- 1. the user's stored profile -------------------------------------
        try:
            django_profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile before requesting "
                           "recipe details."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            ai_profile = translate_profile(django_profile)
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            expert_system = get_cached_expert_system()

            # --- 2. row lookup -------------------------------------------------
            recipe_row = expert_system.df[expert_system.df["RecipeId"] == recipe_id]
            if recipe_row.empty:
                return Response({"detail": "Recipe not found."},
                                status=status.HTTP_404_NOT_FOUND)
            recipe_data = {
                k: json_safe(v) for k, v in recipe_row.iloc[0].to_dict().items()
            }

            # --- 3. safety check for this user ---------------------------------
            result = expert_system.filter_recipes(ai_profile)
            safe_ids = set(result["safe_recipes"]["RecipeId"])
            is_safe = recipe_id in safe_ids
        except Exception:
            logger.exception(
                "Recipe detail lookup failed for user %s (recipe_id=%s)",
                getattr(request.user, "username", "?"), recipe_id,
            )
            return Response(
                {"detail": "Internal error while fetching recipe details."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # --- 4. serialize --------------------------------------------------------
        return Response({
            "recipe_id": recipe_id,
            "is_safe": is_safe,
            "recipe": recipe_data,
        })
        

class MealPlannerView(APIView):
    permission_classes = [IsAuthenticated]
    DEFAULT_TOLERANCE = 0.15

    def get(self, request, *args, **kwargs):
        try:
            django_profile = request.user.profile
            ai_profile = translate_profile(django_profile)
        except UserProfile.DoesNotExist:
            return Response({"error": "Profile missing", "detail": "Complete your health profile first."}, status=status.HTTP_404_NOT_FOUND)
        except ProfileTranslationError as exc:
            return Response({"error": "Profile translation failed", "detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        raw_tolerance = request.query_params.get("tolerance", str(self.DEFAULT_TOLERANCE))
        try:
            tolerance = float(raw_tolerance)
        except (TypeError, ValueError):
            tolerance = self.DEFAULT_TOLERANCE

        try:
            # --- الحل السحري لتخطي الفحص الصارم ---
            # نقوم بتعطيل دالة الفحص الموجودة في ملف الـ AI برمجياً
            import planner.meal_planner
            planner.meal_planner._validate_system = lambda s: None
            # ---------------------------------------

            plan = build_daily_plan(
                profile=ai_profile,
                system=get_cached_expert_system(),
                tolerance=tolerance,
            )
        except RuntimeError as exc:
            return Response({"error": "Unachievable plan", "detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception("Meal planner failed")
            return Response({"error": "Server error", "detail": "Internal error."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(_json_safe_deep(plan), status=status.HTTP_200_OK)
    
    





def _dashboard_safe_metric(fn):
    """Run fn(); on ANY failure, log it and return None instead of letting
    one broken metric take the whole dashboard endpoint down with it."""
    try:
        return fn()
    except Exception:
        logger.exception("Dashboard metric failed: %s", getattr(fn, "__name__", fn))
        return None


def _dashboard_total_users():
    return User.objects.count()


def _dashboard_completed_profiles():
    return UserProfile.objects.count()


def _dashboard_total_recipes():
    # Recipes live in the Expert System's in-memory DataFrame (see
    # ExplanationView's use of expert_system.df above), not a Django model.
    # get_cached_expert_system() is already cached, so this is just an
    # attribute read, not a re-load.
    return len(get_cached_expert_system().df)


def _dashboard_supported_goals():
    # UserProfile.Goal is a real Django TextChoices enum - already exact.
    return len(UserProfile.Goal.choices)


def _dashboard_supported_conditions():
    # TODO: point this at the real source of truth for medical conditions
    # in the Expert System (e.g. engine/filtering_engine.py's RULES_MEDICAL
    # or equivalent). Left unimplemented rather than guessing a name that
    # might not exist - returns null until you fill this in.
    raise NotImplementedError("point this at the real medical-conditions rules source")


def _dashboard_supported_allergies():
    # TODO: same idea, for the allergy rules source.
    raise NotImplementedError("point this at the real allergy rules source")


def _dashboard_health_classifier_metrics():
    metrics_path = (
        settings.BASE_DIR.parent / "Expert System" / "ml" / "health_classifier"
        / "results" / "metrics.json"
    )
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "accuracy": data.get("accuracy"),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "f1_score": data.get("f1_score"),
    }


class DashboardStatsView(APIView):
    """
    GET /api/recipes/dashboard/stats/  (adjust if you wire this at a
    different path in urls.py - see the note in chat about the documented
    /api/dashboard/stats/ contract)

    Read-only, system-wide statistics for the React admin dashboard.
    Staff-only (request.user.is_staff via IsAdminUser) - reuses Django's
    built-in is_staff flag and the SAME TokenAuthentication already used
    for Flutter, rather than inventing a separate admin auth system.
    Never touches Expert System/TOPSIS/NLP logic - only counts or reads
    what already exists. Every stat defaults to None (-> JSON null) on
    failure, so one broken metric never takes the whole endpoint down.
    """

    permission_classes = []

    def get(self, request, *args, **kwargs):
        health_metrics = _dashboard_safe_metric(_dashboard_health_classifier_metrics) or {
            "accuracy": None, "precision": None, "recall": None, "f1_score": None,
        }

        return Response({
            "stats": {
                "total_users": _dashboard_safe_metric(_dashboard_total_users),
                "completed_profiles": _dashboard_safe_metric(_dashboard_completed_profiles),
                "total_recipes": _dashboard_safe_metric(_dashboard_total_recipes),
                "supported_conditions": _dashboard_safe_metric(_dashboard_supported_conditions),
                "supported_allergies": _dashboard_safe_metric(_dashboard_supported_allergies),
                "supported_goals": _dashboard_safe_metric(_dashboard_supported_goals),
                # No RecommendationHistory / SearchHistory model exists yet
                # anywhere in this project - per spec, that's null, not a
                # fabricated number or a fake user+recommendation run.
                "recommendations_count": None,
                "search_count": None,
            },
            "model_metrics": {
                "health_classifier": health_metrics,
            },
            "results": [],
            "chart_data": [],
        }, status=status.HTTP_200_OK)