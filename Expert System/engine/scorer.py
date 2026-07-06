
import re
import math
from typing import List, Optional
import pandas as pd

from rules.goals_and_preferences import GOAL_VECTORS


MEAL_TYPE_DATA_MAP = {
    "breakfast": "Breakfast",
    "lunch":     "MainDish",
    "dinner":    "MainDish",
}


_ISO_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?")


def _is_missing(val) -> bool:
    """يعادل: val is None or (isinstance(val, float) and math.isnan(val))."""
    return val is None or (isinstance(val, float) and math.isnan(val))


def _safe_float(val, default: float = 0.0) -> float:
    return default if _is_missing(val) else float(val)


def _parse_minutes(raw) -> Optional[float]:
    
    text = str(raw) if raw is not None else None
    match = _ISO_DURATION.match(text) if text is not None else None

    def _from_iso(m: "re.Match") -> float:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        return hours * 60 + minutes

    try:
        return (
            None if text is None
            else _from_iso(match) if match
            else float(text)
        )
    except (ValueError, AttributeError, TypeError):
        return None




_CALORIE_DIFF_BONUS = [
    (100, 1.5),
    (200, 0.8),
    (300, 0.2),
    (400, 0.0),          
    (float("inf"), None),  
]

_REVIEWS_BONUS = [
    (9, 0.0),
    (49, 0.1),
    (99, 0.3),
    (float("inf"), 0.5),
]

_TIME_BONUS = [
    (20, 1.0),
    (40, 0.5),
    (60, 0.2),
    (120, 0.0),
    (float("inf"), -0.5),
]

_TIME_EXPLAIN_TIER = [
    (20, "quick"),
    (40, "moderate"),
    (float("inf"), None),
]


def _tier_value(value: float, table: List[tuple]) -> Optional[float]:
    """
    يبحث بجدول (حد_أعلى, قيمة) عن أول صف يكون value أصغر أو يساويه،
    ويرجع قيمته — بدون أي if/elif، فقط next() على generator.
    """
    return next(v for limit, v in table if value <= limit)


def _calorie_score(cal_value, target: float) -> float:
    """
    نسخة دقيقة من المنطق الأصلي للسعرات:
        diff<=100→+1.5 | <=200→+0.8 | <=300→+0.2 | >400→عقوبة متناسبة | غير ذلك→0
    """
    diff = abs(float(cal_value) - target) if not _is_missing(cal_value) else None
    tiered = _tier_value(diff, _CALORIE_DIFF_BONUS) if diff is not None else None
    penalty = -(diff - 400) * 0.01 if diff is not None else 0.0
   
    return (
        0.0 if diff is None
        else tiered if tiered is not None
        else (penalty if diff > 400 else 0.0)
    )


def _reviews_score(num_reviews) -> float:
    return (
        0.0 if _is_missing(num_reviews)
        else _tier_value(float(num_reviews), _REVIEWS_BONUS)
    )


def _time_score(raw_time) -> float:
    minutes = _parse_minutes(raw_time)
    return 0.0 if minutes is None else _tier_value(minutes, _TIME_BONUS)



_NORMALIZATION_CAPS = {
    "Calories":            1200.0,
    "ProteinContent":      100.0,
    "FatContent":          70.0,
    "CarbohydrateContent": 150.0,
    "FiberContent":        25.0,
    "SugarContent":        80.0,
    "SodiumContent":       2000.0,
    "SaturatedFatContent": 25.0,
    "CholesterolContent":  300.0,
}


def _normalize_value(col: str, value: float) -> float:
   
    cap = _NORMALIZATION_CAPS.get(col, 1.0)
    return value / cap if cap else value


def _weighted_goal_score(row: "pd.Series", weights: dict) -> float:
   
    return sum(
        w * _normalize_value(col, _safe_float(row.get(col)))
        for col, w in weights.items()
        if not _is_missing(row.get(col))
    )


def _rating_score(row: "pd.Series") -> float:
    rating = row.get("Rating")
    base = 0.0 if _is_missing(rating) else 0.2 * float(rating)
    reviews_bonus = 0.0 if _is_missing(rating) else _reviews_score(row.get("NumReviews"))
    return base + reviews_bonus


def _health_score(row: "pd.Series") -> float:
    hs = row.get("HealthScore")
    return 0.0 if _is_missing(hs) else float(hs) * 0.01


def _preferred_ingredients_score(
    row: "pd.Series", preferred: Optional[List[str]], ing_col: Optional[str]
) -> float:
    """يعادل: matches = sum(1 for p in preferred if p in text); score += matches*0.5."""
    raw = row.get(ing_col) if (preferred and ing_col) else None
    text = (" ".join(raw) if isinstance(raw, list) else str(raw)).lower() if raw is not None else ""
    matches = sum(1 for p in (preferred or []) if p.lower() in text)
    return matches * 0.5


def _meal_type_score(row: "pd.Series", meal_type: str) -> float:
   
    real_value = MEAL_TYPE_DATA_MAP.get(meal_type.lower(), meal_type)
    meal_col = row.get("MealType")
    applies = (
        meal_type != "any"
        and not _is_missing(meal_col)
        and real_value.lower() in str(meal_col).lower()
    )
    return 1.0 if applies else 0.0


def _empty_calorie_penalty(row: "pd.Series") -> float:
    
    calories = row.get("Calories")
    protein = row.get("ProteinContent")
    fat = row.get("FatContent")
    fiber = row.get("FiberContent")

    has_data = (
        not _is_missing(calories) and float(calories) > 0
        and not _is_missing(protein) and not _is_missing(fat)
    )

    cal_f = float(calories) if has_data else 1.0
    protein_f = float(protein) if has_data else 0.0
    fat_f = float(fat) if has_data else 0.0
    fiber_f = float(fiber) if (has_data and not _is_missing(fiber)) else 0.0

    macro_pct = (protein_f * 4 + fat_f * 9) / cal_f * 100
    fiber_density = fiber_f / cal_f * 100

    base_severity = max(0.0, 15 - macro_pct) / 15
    fiber_relief = min(fiber_density / 5, 0.5)  
    final_severity = base_severity * (1 - fiber_relief)

    return -3.0 * final_severity if has_data else 0.0



def score_recipe(
    row: "pd.Series",
    goal: Optional[str],
    target_meal_calories: int = 500,
    preferred: Optional[List[str]] = None,
    ing_col: Optional[str] = None,
    meal_type: str = "any",
) -> float:
    
    weights = GOAL_VECTORS.get(goal, {}).get("score_weights", {})

    full_score = sum([
        _weighted_goal_score(row, weights),
        _calorie_score(row.get("Calories"), target_meal_calories),
        _rating_score(row),
        _health_score(row),
        _preferred_ingredients_score(row, preferred, ing_col),
        _time_score(row.get("TotalTime") or row.get("PrepTime")),
        _meal_type_score(row, meal_type),
        _empty_calorie_penalty(row),
    ])

    return round(full_score, 4)




def _explain_rules(row: "pd.Series", target: float) -> List[Optional[str]]:
    """
    كل قاعدة شرح = (شرط منطقي بدون فرع تنفيذي, نص الرسالة).
    القيمة الناتجة None تعني "لا تنطبق" وتُفلتر لاحقاً بـ comprehension.
    """
    cal = row.get("Calories", 0)
    sodium = row.get("SodiumContent", 9999)
    fat = row.get("FatContent", 9999)
    protein = row.get("ProteinContent", 0)
    fiber = row.get("FiberContent", 0)
    sugar = row.get("SugarContent", 9999)

    return [
        "✅ Calories on target" if (not pd.isna(cal) and abs(float(cal) - target) <= 100) else None,
        "✅ Low sodium" if (not pd.isna(sodium) and float(sodium) < 400) else None,
        "✅ Low fat" if (not pd.isna(fat) and float(fat) < 10) else None,
        "✅ High protein" if (not pd.isna(protein) and float(protein) > 25) else None,
        "✅ Good fiber" if (not pd.isna(fiber) and float(fiber) > 5) else None,
        "✅ Low sugar" if (not pd.isna(sugar) and float(sugar) < 10) else None,
    ]


def _time_explain(raw_time) -> Optional[str]:
    minutes = _parse_minutes(raw_time)
    tier = _tier_value(minutes, _TIME_EXPLAIN_TIER) if minutes is not None else None
    return (
        None if minutes is None
        else {
            "quick": f"⚡ Quick ({int(minutes)} min)",
            "moderate": f"🕐 {int(minutes)} min",
        }.get(tier)
    )


def explain_recipe(row: "pd.Series", goal: Optional[str]) -> str:
    """
    أنشئ سبباً مختصراً يشرح لماذا أُوصي بهذه الوصفة — بدون أي
    if/elif/for/while statement، فقط بناء قائمة وفلترة.
    """
    target = 500 + (GOAL_VECTORS.get(goal, {}).get("target_calorie_offset", 0) if goal else 0)

    candidates = _explain_rules(row, target) + [
        _time_explain(row.get("TotalTime") or row.get("PrepTime"))
    ]
    reasons = [r for r in candidates if r is not None]

    return " | ".join(reasons[:3]) if reasons else "⚖️ Balanced nutrition profile"
