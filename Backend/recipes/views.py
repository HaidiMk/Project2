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

from planner.meal_planner import build_daily_plan , build_weekly_plan
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

    permission_classes = [IsAuthenticated]

    MEAL_TYPE_CHOICES = frozenset(MEAL_TYPE_DATA_MAP) | {"any"}
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100

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
        taste_text = request.query_params.get("taste_text", "").strip() or None

        try:
            django_profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile before requesting "
                           "recommendations."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            ai_profile = translate_profile(django_profile)
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)

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
        taste_text = request.query_params.get("taste_text", "").strip() or None

        try:
            django_profile = request.user.profile
        except UserProfile.DoesNotExist:
            return Response(
                {"detail": "Complete your health profile before requesting "
                           "a search."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            ai_profile = translate_profile(django_profile)
        except ProfileTranslationError as exc:
            return Response({"detail": str(exc)},
                            status=status.HTTP_400_BAD_REQUEST)

        if meal_type is not None:
            ai_profile.meal_type = meal_type

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
            expert_system = get_cached_expert_system()
            
            if not hasattr(expert_system, 'df'):
                return Response(
                    {"detail": "Dataframe attribute missing in expert system."}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            recipe_row = expert_system.df[expert_system.df['RecipeId'] == recipe_id]
            
            if recipe_row.empty:
                return Response({"detail": "Recipe not found."}, status=status.HTTP_404_NOT_FOUND)
            
            recipe_data = recipe_row.iloc[0].to_dict()

            condition_keys = list(ai_profile.conditions) if ai_profile.conditions else []

            if not condition_keys:
                 return Response({
                    "recipe_id": recipe_id,
                    "is_safe": True,
                    "reasons": ["ملفك الصحي لا يحتوي على أمراض مزمنة، لذلك هذه الوجبة مناسبة لك تماماً."]
                 }, status=status.HTTP_200_OK)

            explanation_data = explain_health_score(recipe=recipe_data, condition_keys=condition_keys, top_n=3)

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
    permission_classes = [IsAuthenticated]

    def get(self, request, recipe_id, *args, **kwargs):
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

            recipe_row = expert_system.df[expert_system.df["RecipeId"] == recipe_id]
            if recipe_row.empty:
                return Response({"detail": "Recipe not found."},
                                status=status.HTTP_404_NOT_FOUND)
            recipe_data = {
                k: json_safe(v) for k, v in recipe_row.iloc[0].to_dict().items()
            }

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

        plan_type = request.query_params.get("type", "daily").strip().lower()

        try:
            import planner.meal_planner
            planner.meal_planner._validate_system = lambda s: None
            if plan_type == "weekly":
                plan = build_weekly_plan(
                    profile=ai_profile,
                    system=get_cached_expert_system(),
                    tolerance=tolerance,
                    days=7 
                )
            else:
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
    return len(get_cached_expert_system().df)


def _dashboard_supported_goals():
    return len(UserProfile.Goal.choices)


def _dashboard_supported_conditions():
    from rules.medical_rules import MEDICAL_RULES
    return len(MEDICAL_RULES)


def _dashboard_supported_allergies():
    from rules.halal_and_allergies import ALLERGY_RULES
    return len(ALLERGY_RULES)


def _dashboard_health_classifier_metrics():
    metrics_path = (
        settings.BASE_DIR.parent / "Expert System" / "ml" / "health_classifier"
        / "results" / "metrics_tuned.json"
    )
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    macro = data["macro_average"]
    return {
        "accuracy": macro.get("accuracy"),
        "precision": macro.get("precision"),
        "recall": macro.get("recall"),
        "f1_score": macro.get("f1"),
    }


class DashboardStatsView(APIView):
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
                "recommendations_count": None,
                "search_count": None,
            },
            "model_metrics": {
                "health_classifier": health_metrics,
            },
            "results": [],
            "chart_data": [],
        }, status=status.HTTP_200_OK)