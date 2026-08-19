from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from typing import List, Optional

_EXPERT_SYSTEM_DIR = str(Path(__file__).resolve().parents[1] / "Expert System")
if _EXPERT_SYSTEM_DIR not in sys.path:
    sys.path.insert(0, _EXPERT_SYSTEM_DIR)

from core.user_profile import UserProfile 
from engine.filtering_engine import DietaryExpertSystem 

try:
    from Nlp.query_parser import parse_query
except ModuleNotFoundError:  
    from query_parser import parse_query


_NUMERIC_FILTER_COLS = {
    "protein_min":  ("ProteinContent", ">="),
    "sugar_max":    ("SugarContent",   "<="),
    "sodium_max":   ("SodiumContent",  "<="),
    "fiber_min":    ("FiberContent",   ">="),
    "calories_max": ("Calories",       "<="),
}

_INGREDIENT_COL_CANDIDATES = (
    "IngredientsList",
    "RecipeIngredientParts",
    "Ingredients",
)

_DISPLAY_COLS = [
    "Name", "Calories", "ProteinContent", "SugarContent", "SodiumContent",
    "FiberContent", "Rating", "MealType", "_reason",
]


def _detect_ingredient_column(df) -> Optional[str]:
    for candidate in _INGREDIENT_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _overlay_profile(base_profile: UserProfile, filters: dict) -> UserProfile:
    conditions = list(base_profile.conditions)
    if filters.get("condition") and filters["condition"] not in conditions:
        conditions.append(filters["condition"])

    allergies = list(base_profile.allergies)
    for allergy in filters.get("allergy") or []:
        if allergy not in allergies:
            allergies.append(allergy)

    preferences = list(base_profile.preferences)
    if filters.get("diet_preference") and filters["diet_preference"] not in preferences:
        preferences.append(filters["diet_preference"])

    meal_type = filters.get("meal_type") or base_profile.meal_type

    return dataclasses.replace(
        base_profile,
        conditions=conditions,
        allergies=allergies,
        preferences=preferences,
        meal_type=meal_type,
    )


def _apply_query_refinement_filters(df, filters: dict, ing_col: Optional[str]):

    refined = df

    for key, (col, op) in _NUMERIC_FILTER_COLS.items():
        value = filters.get(key)
        if value is None or col not in refined.columns:
            continue
        refined = refined[refined[col] >= value] if op == ">=" else \
                  refined[refined[col] <= value]

    ingredient = filters.get("main_ingredient")
    if ingredient:
        match_col = ing_col if (ing_col and ing_col in refined.columns) else \
                    ("Name" if "Name" in refined.columns else None)
        if match_col:
            pattern = re.compile(
                r"\b" + re.escape(ingredient.lower()) + r"s?\b"
            )
            mask = (
                refined[match_col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(pattern, regex=True, na=False)
            )
            refined = refined[mask]

    return refined.reset_index(drop=True)


def search_recipes(
    query_text: str,
    base_profile: UserProfile,
    expert_system: DietaryExpertSystem,
    top_n: int = 20,
) -> dict:

    filters = parse_query(query_text)
    merged_profile = _overlay_profile(base_profile, filters)

    result = expert_system.filter_recipes(merged_profile)

    safe = result["safe_recipes"]
    ing_col = _detect_ingredient_column(safe)
    refined = _apply_query_refinement_filters(safe, filters, ing_col)

    top = refined.head(top_n)
    present_cols = [c for c in _DISPLAY_COLS if c in top.columns]

    return {
        "query":               query_text,
        "filters":             filters,
        "merged_profile":      merged_profile,
        "expert_result":       result,
        "safe_recipes":        top,
        "top_recipes":         top[present_cols].to_dict(orient="records"),
        "total_safe":          len(refined),
        "total_after_expert":  int(result["total_safe"]),
        "total_original":      int(result["total_original"]),
        "meal_type":           result["meal_type"],
        "ai_health_scoring":   "_ai_health_score" in result["safe_recipes"].columns,
        "warnings":            result["warnings"],
    }
