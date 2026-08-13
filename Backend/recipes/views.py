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
    5. get_expert_system().filter_recipes() — the shared Step-3 singleton
       (never constructs a new DietaryExpertSystem).
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

import logging

import numpy as np
import pandas as pd
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# MEAL_TYPE_DATA_MAP keys are the per-request meal vocabulary the engine
# actually understands (breakfast -> Breakfast, lunch/dinner -> MainDish);
# "any" is the engine's no-filter sentinel. Reusing the engine's map instead
# of re-declaring the vocabulary keeps the two in lockstep.
from engine.filtering_engine import MEAL_TYPE_DATA_MAP
from profiles.models import UserProfile
from recipes.services.ai_runtime import get_expert_system
from recipes.services.profile_translator import (
    ProfileTranslationError,
    translate_profile,
)
from topsis_model import rank_with_topsis

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Convert a numpy/pandas scalar to a native Python type; NaN/None -> None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        "RecipeId", "Name", "Calories", "ProteinContent",
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
            result = get_expert_system().filter_recipes(ai_profile)
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
            {k: _json_safe(v) for k, v in row.items()}
            for row in top.to_dict("records")
        ]

        return Response({
            "count": len(results),
            "total_safe": result.get("total_safe", len(result["safe_recipes"])),
            "total_original": result.get("total_original", 0),
            "meal_type": meal_type,
            "results": results,
        })