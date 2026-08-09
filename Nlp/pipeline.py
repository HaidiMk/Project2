"""
pipeline.py — the bridge between the Nlp/ query parser and the Expert System.

Given a free-text search query plus a caller-supplied UserProfile, this
module:

    1. parses the query into structured filters (Nlp/query_parser.py),
    2. overlays the query-specified fields onto a COPY of the base profile,
    3. calls the real, existing DietaryExpertSystem.filter_recipes() — the
       ONLY place medical safety is decided (NFR-010), and
    4. applies the extra query constraints that UserProfile has no fields
       for (protein/sugar/sodium/fiber/calories limits, main ingredient)
       as a clearly-separated post-filter that can only narrow the Expert
       System's results, never override or bypass them.

No filtering / medical-rule logic is duplicated here; the Expert System
classes are imported and called as-is.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from typing import List, Optional

# --- Expert System is a sibling folder; its modules use bare imports like
# --- `from core.user_profile import ...`, so its directory must be on
# --- sys.path for them to resolve. This is the integration layer between
# --- the standalone Nlp/ package and Expert System/, so that dependency is
# --- intentional here (unlike query_parser.py, which stays dependency-free).
_EXPERT_SYSTEM_DIR = str(Path(__file__).resolve().parents[1] / "Expert System")
if _EXPERT_SYSTEM_DIR not in sys.path:
    sys.path.insert(0, _EXPERT_SYSTEM_DIR)

from core.user_profile import UserProfile  # noqa: E402
from engine.filtering_engine import DietaryExpertSystem  # noqa: E402

try:
    from Nlp.query_parser import parse_query
except ModuleNotFoundError:  # executed standalone with Nlp/ on sys.path
    from query_parser import parse_query


# Numeric query constraints -> real dataset columns (from core/constants.py
# NUTRIENT_COLS). Direction is fixed by the schema: mins for protein/fiber,
# maxes for sugar/sodium/calories.
_NUMERIC_FILTER_COLS = {
    "protein_min":  ("ProteinContent", ">="),
    "sugar_max":    ("SugarContent",   "<="),
    "sodium_max":   ("SodiumContent",  "<="),
    "fiber_min":    ("FiberContent",   ">="),
    "calories_max": ("Calories",       "<="),
}

# Column candidates for main-ingredient matching (same order the engine uses).
_INGREDIENT_COL_CANDIDATES = (
    "IngredientsList",
    "RecipeIngredientParts",
    "Ingredients",
)

# Columns kept in the convenience `top_recipes` records.
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
    """
    Build a NEW UserProfile from base_profile plus only the fields the
    query actually specified. base_profile is never mutated.

    - condition        -> appended to .conditions (no dupes)
    - meal_type        -> overrides .meal_type when present
    - allergy          -> merged into .allergies (union, no dupes)
    - diet_preference  -> merged into .preferences (no dupes)
    Everything else (age/height/weight/gender/goal/...) comes straight
    from base_profile.
    """
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
    """
    Query-refinement filters, applied AFTER the Expert System's medical
    filtering. These handle the parsed constraints UserProfile has no field
    for (numeric limits + main ingredient). They only narrow the Expert
    System's safe list further — they can never restore or bypass anything
    the Expert System excluded for medical reasons (NFR-010).
    """
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
    """
    Turn a free-text query into a medically-safe, ranked recipe list.

    Args:
        query_text:     free-text English search query.
        base_profile:   already-populated UserProfile of the logged-in user
                        (age/height/weight/gender come from here — they are
                        never guessed from the query text).
        expert_system:  already-constructed DietaryExpertSystem instance
                        (built once by the caller; NOT constructed here).
        top_n:          how many of the top-ranked recipes to return.

    Returns:
        dict with the parsed filters, the merged profile, the full Expert
        System result, and the refined `safe_recipes` DataFrame.
    """
    filters = parse_query(query_text)
    merged_profile = _overlay_profile(base_profile, filters)

    # The real medical filtering + ranking — the only medical-safety step.
    result = expert_system.filter_recipes(merged_profile)

    # Extra constraints that UserProfile cannot express — narrows only.
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
        # NCF ranking was removed from the engine (replaced by the health
        # classifier); today ML scoring means the AI health score the
        # engine always attaches to safe_recipes — reported live from the
        # actual result instead of a stale flag.
        "ai_health_scoring":   "_ai_health_score" in result["safe_recipes"].columns,
        "warnings":            result["warnings"],
    }
