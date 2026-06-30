"""
cli_interface.py — Smart Dietary Advisor v4.0 — FIXED
======================================================
التغييرات:
    1. PCOS يظهر للإناث فقط
    2. سؤال الحساسية منفصل (نعم/لا أولاً)
    3. الحساسيات تأتي من ALLERGIES_BY_STAGE وليس DISEASES_BY_STAGE
    4. تنبيه عند اختيار milk + lactose_intolerance معاً
    5. مستوى النشاط البدني
    6. حماية التعارضات المستحيلة طبياً
    7. حذف التكرار في بناء UserProfile
    8. إضافة سؤال نوع الوجبة (فطور/غداء/عشاء)
    9. ملخص غذائي + إحصائيات الوصفات المعروضة
"""

import html
import math
from typing import List, Dict, Optional
import pandas as pd

from core.constants import (
    AGE_RANGE_EN, DISEASES_BY_STAGE, DISEASES_FEMALE_ONLY,
    ALLERGIES_BY_STAGE, DISEASE_EN, MEAL_TYPE_EN,
    ALLERGY_EN, PREFERENCE_EN, GOALS_BY_CONDITION, GOAL_EN,
)
from core.user_profile import UserProfile
from rules.halal_and_allergies import ALLERGY_RULES
from rules.goals_and_preferences import GOAL_VECTORS


def _divider(title: str = ""):
    W = 60
    if title:
        pad = max(0, (W - len(title) - 2) // 2)
        print("\n" + "-" * pad + f" {title} " + "-" * pad)
    else:
        print("\n" + "-" * W)


def _pick_multi(options: List[str], labels: Dict[str, str], title: str) -> List[str]:
    _divider(title)
    for i, k in enumerate(options, 1):
        print(f"  {i:2}. {labels.get(k, k)}")
    print("  0.  None / Skip")
    raw = input("\n  Select (comma-separated numbers, or 0 to skip): ").strip()
    if not raw or raw == "0":
        return []
    raw = raw.replace("،", ",").replace("و", ",")
    selected = []
    try:
        for x in raw.split(","):
            x = x.strip()
            if not x:
                continue
            idx = int(x) - 1
            if 0 <= idx < len(options):
                selected.append(options[idx])
    except ValueError:
        print("  Warning: Invalid input — skipped.")
    return selected


def _pick_one(options: List[str], labels: Dict[str, str], title: str) -> Optional[str]:
    _divider(title)
    for i, k in enumerate(options, 1):
        print(f"  {i:2}. {labels.get(k, k)}")
    print("  0.  Skip")
    raw = input("\n  Select a number: ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    return None


def get_available_goals(profile: UserProfile) -> List[str]:
    if profile.pregnant:
        return GOALS_BY_CONDITION["pregnancy"]
    return GOALS_BY_CONDITION.get(profile.life_stage, GOALS_BY_CONDITION["adult"])


# ══ التعارضات المستحيلة طبياً ════════════════════════════════
IMPOSSIBLE_COMBOS = [
    (
        {"obesity", "underweight"},
        "Obesity and Underweight cannot exist together."
    ),
    (
        {"hypothyroidism", "hyperthyroidism"},
        "Hypothyroidism and Hyperthyroidism cannot exist together."
    ),
    (
        {"obesity", "underweight", "diabetes"},
        "Obesity + Underweight + Diabetes is impossible."
    ),
]


def _validate_conditions(conditions: List[str], disease_list: List[str]) -> List[str]:
    """تحقق من التعارضات وأعد الاختيار لو فيه مشكلة."""
    while True:
        cond_set = set(conditions)
        conflict = None
        for combo, msg in IMPOSSIBLE_COMBOS:
            if combo.issubset(cond_set):
                conflict = msg
                break
        if not conflict:
            return conditions
        print(f"\n  ❌  Conflict detected: {conflict}")
        print("  Please re-select your conditions.\n")
        conditions = _pick_multi(disease_list, DISEASE_EN, "Medical Conditions")


def build_profile_interactive() -> UserProfile:
    print("\n" + "=" * 60)
    print("  Smart Dietary Advisor -- Unified v4.0")
    print("  User Profile Setup")
    print("=" * 60)
    print("\n  Age Categories:")
    print("    Child   :   1 - 12  years")
    print("    Teen    :  13 - 17  years")
    print("    Adult   :  18 - 64  years")
    print("    Elderly :  65+       years")

    # ── العمر ─────────────────────────────────────────────
    while True:
        try:
            age = int(input("\n  Age (years): ").strip())
            if 1 <= age <= 110:
                break
            print("  Warning: Age must be 1-110.")
        except ValueError:
            print("  Warning: Please enter a valid integer.")

    stage = ("child"   if age <= 12 else
             "teen"    if age <= 17 else
             "elderly" if age >= 65 else "adult")
    print(f"  OK  Age category: {AGE_RANGE_EN[stage]}")

    # ── الجنس ─────────────────────────────────────────────
    _divider("Gender")
    print("  1. Male\n  2. Female")
    gender = "male"
    while True:
        r = input("  Select (1 or 2): ").strip()
        if r == "1":   gender = "male";   break
        elif r == "2": gender = "female"; break
        print("  Warning: Select 1 or 2.")

    # ── القياسات ──────────────────────────────────────────
    _divider("Measurements")
    while True:
        try:
            height = float(input("  Height (cm): ").strip())
            weight = float(input("  Weight (kg): ").strip())
            if 50 <= height <= 250 and 2 <= weight <= 400:
                break
            print("  Warning: Unrealistic values.")
        except ValueError:
            print("  Warning: Please enter valid numbers.")

    h = height / 100
    bmi_tmp = round(weight / (h * h), 1)
    print(f"\n  BMI: {bmi_tmp}  |  Ideal range: {round(18.5*h*h,1)}-{round(24.9*h*h,1)} kg")
    if stage == "child":
        print("  Note: BMI interpretation for children differs from adults.")
        print("        Please consult a pediatrician for accurate assessment.")

    # ── مستوى النشاط البدني ───────────────────────────────
    activity_level = "light"
    if age >= 13:
        _divider("Physical Activity Level")
        print("  1. Sedentary   — desk job, little or no exercise")
        print("  2. Light       — light exercise 1-3 days/week")
        print("  3. Moderate    — moderate exercise 3-5 days/week")
        print("  4. Active      — hard exercise 6-7 days/week")
        print("  5. Very Active — athlete or physical labor job")

        activity_map = {
            "1": "sedentary",
            "2": "light",
            "3": "moderate",
            "4": "active",
            "5": "very_active",
        }
        while True:
            r = input("\n  Select (1-5): ").strip()
            if r in activity_map:
                activity_level = activity_map[r]
                print(f"  OK  Activity level: {activity_level}")
                break
            print("  Warning: Select 1-5.")

    # ── الحمل (للإناث فقط) ────────────────────────────────
    pregnant = False
    if gender == "female" and age >= 17:
        _divider("Pregnancy")
        print("  1. Yes -- currently pregnant\n  2. No")
        pregnant = (input("  Select (1 or 2): ").strip() == "1")

    # ── الأمراض ───────────────────────────────────────────
    disease_list = list(DISEASES_BY_STAGE[stage])
    if gender == "female" and stage in DISEASES_FEMALE_ONLY:
        disease_list = disease_list + DISEASES_FEMALE_ONLY[stage]

    conditions = _pick_multi(disease_list, DISEASE_EN, "Medical Conditions")
    conditions = _validate_conditions(conditions, disease_list)

    # ── الحساسيات — سؤال منفصل ────────────────────────────
    _divider("Food Allergies")
    print("  Do you have any food allergies?")
    print("  1. Yes -- I have food allergies")
    print("  2. No  -- I have no food allergies")

    allergies = []
    has_allergy = input("\n  Select (1 or 2): ").strip()
    if has_allergy == "1":
        allergy_options = ALLERGIES_BY_STAGE.get(stage, list(ALLERGY_RULES.keys()))
        allergies = _pick_multi(allergy_options, ALLERGY_EN, "Select Your Allergies")

        if "milk" in allergies and "lactose_intolerance" in conditions:
            print("\n  [INFO] You selected both 'Milk Allergy' and 'Lactose Intolerance'.")
            print("         These are different conditions but have similar dietary restrictions.")
            print("         The system will apply the stricter milk allergy rules.")
    else:
        print("  OK  No allergies recorded.")

    # ── التفضيلات ─────────────────────────────────────────
    preferences = _pick_multi(list(PREFERENCE_EN.keys()), PREFERENCE_EN, "Food Preferences")

    # ── بناء الملف الشخصي ─────────────────────────────────
    profile_tmp = UserProfile(
        age=age, height=height, weight=weight, gender=gender,
        pregnant=pregnant, conditions=conditions,
        allergies=allergies, preferences=preferences,
        activity_level=activity_level,
    )

    # ── الهدف الغذائي ─────────────────────────────────────
    goal = _pick_one(get_available_goals(profile_tmp), GOAL_EN, "Nutritional Goal")
    profile_tmp.goal = goal

    # ── نوع الوجبة ────────────────────────────────────────
    _divider("Meal Type")
    print("  1. Breakfast")
    print("  2. Lunch")
    print("  3. Dinner")
    print("  4. No preference — show all")

    meal_map = {
        "1": "breakfast",
        "2": "lunch",
        "3": "dinner",
        "4": "any",
    }
    meal_type = "any"
    while True:
        r = input("\n  Select (1-4): ").strip()
        if r in meal_map:
            meal_type = meal_map[r]
            print(f"  OK  Meal type: {MEAL_TYPE_EN[meal_type]}")
            break
        print("  Warning: Select 1-4.")

    profile_tmp.meal_type = meal_type
    return profile_tmp


def create_profile(
    age: int, height: float, weight: float, gender: str,
    conditions: List[str] = None, allergies: List[str] = None,
    preferences: List[str] = None, goal: str = None,
    pregnant: bool = False,
    activity_level: str = "light",
    meal_type: str = "any",
) -> UserProfile:
    p = UserProfile(
        age=age, height=height, weight=weight, gender=gender,
        pregnant=pregnant,
        conditions=conditions or [],
        allergies=allergies or [],
        preferences=preferences or [],
        goal=goal,
        activity_level=activity_level,
    )
    p.meal_type = meal_type
    return p


def print_results(result: dict, top_n: int = 15):
    SEP  = "=" * 120
    sep2 = "-" * 120
    ps   = result["profile_summary"]

    print(f"\n{SEP}")
    print("  Smart Dietary Advisor -- Unified v4.0 -- Results".center(118))
    print(SEP)

    print(f"\n  Age:              {ps['age']} years  ({ps['life_stage']})")
    g_str = ps["gender"] + (" -- Pregnant" if ps["pregnant"] else "")
    print(f"  Gender:           {g_str}")
    print(f"  Height / Weight:  {ps['height']} cm / {ps['weight']} kg")
    print(f"  BMI:              {ps['bmi']}  ->  {ps['bmi_category']}")
    print(f"  Ideal Weight:     {ps['ideal_weight']}")
    print(f"  Activity Level:   {ps.get('activity_level', 'light')}")
    print(f"  Meal Type:        {result.get('meal_type', 'any')}")

    # ── ملخص غذائي ← جديد ────────────────────────────────
    daily  = ps["daily_calories"]
    meal   = result.get("target_meal_calories", ps["per_meal_kcal"])
    remain = daily - meal
    print(f"\n  Nutritional Summary:")
    print(f"     Daily Target  : {daily:,} kcal")
    print(f"     This Meal     : {meal:,} kcal  ({round(meal/daily*100)}% of daily)")
    print(f"     Remaining     : {remain:,} kcal for other meals")

    if result.get("healthy_style"):
        hs = result["healthy_style"]
        print(f"\n  -- {hs['name']} --")
        for line in hs["description"].split("\n"):
            print(f"  {line}")
        print(f"  Source: {hs['source']}")

    print(f"\n  Safe Recipes: {result['total_safe']:,} / {result['total_original']:,}"
          f"  ({result['filter_rate']}%)")

    if result["warnings"]:
        print("\n  Warnings & Conflicts:")
        for w in result["warnings"]:
            print(f"     * {w}")

    if result["notes"]:
        shown_notes = result["notes"][:10]
        print(f"\n  Applied Clinical Rules ({len(result['notes'])} total, showing {len(shown_notes)}):")
        for n in shown_notes:
            print(f"     * {n}")

    print(f"\n  Top {top_n} Recommended Recipes:")
    print("  " + sep2)

    df = result["safe_recipes"]
    if df.empty:
        print("  No recipes match all restrictions.")
        print(SEP)
        return

    name_col = "Name" if "Name" in df.columns else df.columns[0]
    hdr = (f"  {'#':<3} {'Recipe Name':<50} "
           f"{'Cal':>6} {'Prot':>5} {'Carb':>5} {'Fat':>5} "
           f"{'Sod':>6} {'Fib':>5} {'Sug':>5} {'Rate':>5}")
    print(hdr)
    print("  " + sep2)

    def _v(row, col):
        v = row.get(col)
        return int(float(v)) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0

    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        # 🛠️ إصلاح: ~4.2% من أسماء الوصفات (16,245 من 384,541) تحتوي
        # بقايا HTML entities من استخراج البيانات الأصلي (&amp; بدل &،
        # &quot; بدل "، &eacute; بدل é...) — لا تؤثر على دقة الفلترة
        # الطبية إطلاقاً، لكنها تظهر بشكل غير احترافي بالنتيجة النهائية
        # المعروضة للمستخدم (مثلاً "Horse &amp; Buggy" بدل "Horse &
        # Buggy"). html.unescape() يحوّلها لشكلها الصحيح المقروء.
        raw  = html.unescape(str(row.get(name_col, "Unknown")))
        name = (raw[:46] + "..") if len(raw) > 48 else raw.ljust(48)
        cal  = _v(row, "Calories")
        prot = _v(row, "ProteinContent")
        carb = _v(row, "CarbohydrateContent")
        fat  = _v(row, "FatContent")
        sod  = _v(row, "SodiumContent")
        fib  = _v(row, "FiberContent")
        sug  = _v(row, "SugarContent")
        rating = row.get("Rating")
        r_str = f"{float(rating):.1f}" if (
            rating is not None and not (isinstance(rating, float) and math.isnan(rating))
        ) else "  - "

        print(f"  {i:<3} {name:<50} "
              f"{cal:>6} {prot:>5} {carb:>5} {fat:>5} "
              f"{sod:>6} {fib:>5} {sug:>5} {r_str:>5}")

        if "_reason" in row:
            print(f"       -> {row['_reason']}")
        if i < min(top_n, len(df)):
            print("")

    # ── إحصائيات الوصفات المعروضة ← جديد ─────────────────
    shown_df = df.head(top_n)
    avg_cal  = shown_df["Calories"].mean()       if "Calories"       in shown_df.columns else 0
    avg_prot = shown_df["ProteinContent"].mean() if "ProteinContent" in shown_df.columns else 0
    avg_sod  = shown_df["SodiumContent"].mean()  if "SodiumContent"  in shown_df.columns else 0

    print(f"\n  Average of top {top_n} recipes shown:")
    print(f"     Calories : {avg_cal:>6.0f} kcal")
    print(f"     Protein  : {avg_prot:>6.0f} g")
    print(f"     Sodium   : {avg_sod:>6.0f} mg")

    print(SEP)