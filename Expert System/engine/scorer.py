"""
scorer.py — Smart Dietary Advisor v4.0 — FIXED
===============================================
التغييرات:
    1. مكافأة عدد التقييمات (NumReviews) — مش بس المتوسط
    2. مكافأة وقت التحضير القصير
    3. مكافأة توافق نوع الوجبة
    4. explain_recipe: أضفنا وقت التحضير في الشرح
"""

import math
from typing import List, Optional
import pandas as pd

from rules.goals_and_preferences import GOAL_VECTORS


def score_recipe(
    row: "pd.Series",
    goal: Optional[str],
    target_meal_calories: int = 500,
    preferred: Optional[List[str]] = None,
    ing_col: Optional[str] = None,
    meal_type: str = "any",
) -> float:
    """
    احسب درجة الوصفة بناءً على:
        1. أوزان الهدف الغذائي (GOAL_VECTORS)
        2. قرب السعرات من الهدف
        3. تقييم المستخدمين (Rating) + عدد التقييمات (NumReviews)
        4. HealthScore
        5. وجود مكونات مفضلة
        6. وقت التحضير
        7. توافق نوع الوجبة
    """
    if not goal or goal not in GOAL_VECTORS:
        rating = row.get("Rating", 3.0)
        return float(rating) if not pd.isna(rating) else 3.0

    gv      = GOAL_VECTORS[goal]
    weights = gv.get("score_weights", {})
    score   = 0.0

    # ── 1: أوزان الهدف ───────────────────────────────────
    for col, w in weights.items():
        val = row.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        score += w * float(val)

    # ── 2: قرب السعرات من الهدف ──────────────────────────
    cal = row.get("Calories")
    if cal is not None and not (isinstance(cal, float) and math.isnan(cal)):
        diff = abs(float(cal) - target_meal_calories)
        if diff <= 100:   score += 1.5
        elif diff <= 200: score += 0.8
        elif diff <= 300: score += 0.2
        elif diff > 400:  score -= (diff - 400) * 0.01

    # ── 3: التقييم + عدد التقييمات ← محسّن ───────────────
    rating = row.get("Rating")
    if rating is not None and not (isinstance(rating, float) and math.isnan(rating)):
        score += 0.2 * float(rating)

        # مكافأة إضافية للوصفات الموثوقة (كثير تقييمات)
        num_reviews = row.get("NumReviews")
        if num_reviews is not None and not (isinstance(num_reviews, float) and math.isnan(num_reviews)):
            n = float(num_reviews)
            if n >= 100:  score += 0.5
            elif n >= 50: score += 0.3
            elif n >= 10: score += 0.1

    # ── 4: HealthScore ────────────────────────────────────
    hs = row.get("HealthScore")
    if hs is not None and not (isinstance(hs, float) and math.isnan(hs)):
        score += float(hs) * 0.01

    # ── 5: المكونات المفضلة ───────────────────────────────
    if preferred and ing_col:
        raw = row.get(ing_col)
        if raw is not None:
            text = (" ".join(raw) if isinstance(raw, list) else str(raw)).lower()
            matches = sum(1 for p in preferred if p.lower() in text)
            score += matches * 0.5

    # ── 6: وقت التحضير ← جديد ────────────────────────────
    total_time = row.get("TotalTime") or row.get("PrepTime")
    if total_time is not None and not (isinstance(total_time, float) and math.isnan(total_time)):
        try:
            # TotalTime قد يكون نص مثل "PT30M" أو رقم بالدقائق
            t = str(total_time)
            if "PT" in t:
                # ISO 8601 format — PT30M أو PT1H30M
                minutes = 0
                if "H" in t:
                    minutes += int(t.split("H")[0].replace("PT", "")) * 60
                    t = t.split("H")[1]
                if "M" in t:
                    minutes += int(t.replace("M", ""))
            else:
                minutes = float(total_time)

            if minutes <= 20:   score += 1.0   # سريع جداً
            elif minutes <= 40: score += 0.5   # معقول
            elif minutes <= 60: score += 0.2   # مقبول
            elif minutes > 120: score -= 0.5   # طويل جداً
        except (ValueError, AttributeError):
            pass

    # ── 7: توافق نوع الوجبة ← جديد ──────────────────────
    if meal_type != "any":
        meal_col = row.get("MealType")
        if meal_col is not None and not (isinstance(meal_col, float) and math.isnan(meal_col)):
            if meal_type.lower() in str(meal_col).lower():
                score += 1.0   # مكافأة توافق نوع الوجبة

    return round(score, 4)


def explain_recipe(row: "pd.Series", goal: Optional[str]) -> str:
    """
    أنشئ سبباً مختصراً يشرح لماذا أُوصي بهذه الوصفة.
    """
    reasons = []
    target = 500 + (GOAL_VECTORS.get(goal, {}).get("target_calorie_offset", 0) if goal else 0)

    cal = row.get("Calories", 0)
    if not pd.isna(cal) and abs(float(cal) - target) <= 100:
        reasons.append("✅ Calories on target")

    sodium = row.get("SodiumContent", 9999)
    if not pd.isna(sodium) and float(sodium) < 400:
        reasons.append("✅ Low sodium")

    fat = row.get("FatContent", 9999)
    if not pd.isna(fat) and float(fat) < 10:
        reasons.append("✅ Low fat")

    protein = row.get("ProteinContent", 0)
    if not pd.isna(protein) and float(protein) > 25:
        reasons.append("✅ High protein")

    fiber = row.get("FiberContent", 0)
    if not pd.isna(fiber) and float(fiber) > 5:
        reasons.append("✅ Good fiber")

    sugar = row.get("SugarContent", 9999)
    if not pd.isna(sugar) and float(sugar) < 10:
        reasons.append("✅ Low sugar")

    # ── وقت التحضير ← جديد ───────────────────────────────
    total_time = row.get("TotalTime") or row.get("PrepTime")
    if total_time is not None and not (isinstance(total_time, float) and math.isnan(total_time)):
        try:
            t = str(total_time)
            if "PT" in t:
                minutes = 0
                if "H" in t:
                    minutes += int(t.split("H")[0].replace("PT", "")) * 60
                    t = t.split("H")[1]
                if "M" in t:
                    minutes += int(t.replace("M", ""))
            else:
                minutes = float(total_time)

            if minutes <= 20:
                reasons.append(f"⚡ Quick ({int(minutes)} min)")
            elif minutes <= 40:
                reasons.append(f"🕐 {int(minutes)} min")
        except (ValueError, AttributeError):
            pass

    return " | ".join(reasons[:3]) if reasons else "⚖️ Balanced nutrition profile"