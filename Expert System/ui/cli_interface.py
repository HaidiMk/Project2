"""
cli_interface.py — Smart Dietary Advisor v4.0 — NO-CONTROL-FLOW EDITION
========================================================================
خالٍ تماماً من: if / elif / else / for / while / map / filter / reduce
ومن التعابير الثلاثية ومن الـ comprehensions.

البدائل:
    - while True للتحقق من المدخلات → دالة قراءة عودية (تستدعي نفسها عند الخطأ)
    - for العرض/الترقيم             → التكرار العودي _fold / _foreach
    - سلاسل if/elif                  → numpy.select / dict-dispatch
    - if بشرط واحد                   → _pick / _do
    - try/except للتحويل             → _attempt

التوقيعات محفوظة: build_profile_interactive / create_profile / print_results
                  get_available_goals (يستدعيها main.py وفحوص أخرى).
"""

import math
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import pandas as pd

from core.constants import (
    AGE_RANGE_EN, DISEASES_BY_STAGE, DISEASES_FEMALE_ONLY,
    ALLERGIES_BY_STAGE, DISEASE_EN, MEAL_TYPE_EN,
    ALLERGY_EN, PREFERENCE_EN, GOALS_BY_CONDITION, GOAL_EN,
)
from core.user_profile import UserProfile
from rules.halal_and_allergies import ALLERGY_RULES
from rules.goals_and_preferences import GOAL_VECTORS


# ══════════════════════════════════════════════════════════════
# أدوات بديلة عن بنى التحكم
# ══════════════════════════════════════════════════════════════

def _pick(cond, when_true, when_false):
    """بديل التعبير الثلاثي."""
    return {True: when_true, False: when_false}[bool(cond)]


def _do(cond, do_true: Callable, do_false: Callable):
    """بديل if/else تنفيذي."""
    return {True: do_true, False: do_false}[bool(cond)]()


def _attempt(fn: Callable, on_error: Callable, exc_types=ValueError):
    """
    بديل try/except. يلتقط ValueError فقط افتراضياً — مطابقةً للأصل
    (الذي كان يستخدم 'except ValueError'). الاستثناءات الأخرى (مثل EOFError
    من Ctrl+D أو KeyboardInterrupt) تنتشر كما في النسخة الأصلية تماماً.
    """
    try:
        return fn()
    except exc_types as exc:                        # noqa: BLE001
        return on_error(exc)


def _fold(seq, acc, fn: Callable[[Any, Any], Any]):
    """تكرار عودي يُراكم قيمة — بديل for/while/reduce."""
    items = list(seq)

    def step(rest, accumulator):
        empty = (len(rest) == 0)
        return {
            True:  (lambda: accumulator),
            False: (lambda: step(rest[1:], fn(accumulator, rest[0]))),
        }[empty]()

    return step(items, acc)


def _foreach(seq, fn: Callable[[Any], None]):
    """تنفيذ fn على كل عنصر لأثره الجانبي (طباعة) — بلا قيمة متراكمة."""
    def apply(_, item):
        fn(item)
        return None

    _fold(seq, None, apply)
    return None


def _read_valid(prompt: str, parse: Callable, on_fail: str, on_parse_fail=None):
    """
    اقرأ مدخلاً وحقّقه عودياً — بديل حلقة while True.
    parse(raw) يُرجع قيمة صالحة أو None (خارج المدى) أو يرمي ValueError (غير صالح).
    on_fail        → رسالة الخطأ عند القيمة خارج المدى (None).
    on_parse_fail  → رسالة الخطأ عند ValueError (افتراضياً = on_fail).
    """
    parse_msg = _pick(on_parse_fail is not None, on_parse_fail, on_fail)
    raw = input(prompt).strip()

    def attempt():
        value = parse(raw)
        valid = value is not None
        return {
            True:  (lambda: value),
            False: (lambda: _retry(on_fail)),
        }[valid]()

    def _retry(msg):
        print(msg)
        return _read_valid(prompt, parse, on_fail, on_parse_fail)

    def on_err(_exc):
        return _retry(parse_msg)

    return _attempt(attempt, on_err)


# ══════════════════════════════════════════════════════════════
# عناصر العرض
# ══════════════════════════════════════════════════════════════

def _divider(title: str = ""):
    W = 60
    def titled():
        pad = max(0, (W - len(title) - 2) // 2)
        print("\n" + "-" * pad + f" {title} " + "-" * pad)
        return None
    def plain():
        print("\n" + "-" * W)
        return None
    _do(bool(title), titled, plain)


def _print_numbered(options: List[str], labels: Dict[str, str]):
    """اطبع قائمة مرقّمة (1..n) بلا for."""
    idx = list(range(len(options)))

    def line(i):
        k = options[i]
        print(f"  {i + 1:2}. {labels.get(k, k)}")

    _foreach(idx, line)


def _pick_multi(options: List[str], labels: Dict[str, str], title: str) -> List[str]:
    _divider(title)
    _print_numbered(options, labels)
    print("  0.  None / Skip")
    raw = input("\n  Select (comma-separated numbers, or 0 to skip): ").strip()

    skip = (not raw) or (raw == "0")

    def parse_selection():
        cleaned = raw.replace("،", ",").replace("و", ",")
        parts = cleaned.split(",")

        def add(acc, x):
            xs = x.strip()
            empty = (len(xs) == 0)

            def handle():
                idx = int(xs) - 1
                ok = (0 <= idx) and (idx < len(options))
                return _pick(ok, acc + [options[idx]], acc)

            return _do(empty, lambda: acc, handle)

        def on_err(_exc):
            print("  Warning: Invalid input — skipped.")
            return []

        return _attempt(lambda: _fold(parts, [], add), on_err)

    return _do(skip, lambda: [], parse_selection)


def _pick_one(options: List[str], labels: Dict[str, str], title: str) -> Optional[str]:
    _divider(title)
    _print_numbered(options, labels)
    print("  0.  Skip")
    raw = input("\n  Select a number: ").strip()

    def parse():
        idx = int(raw) - 1
        ok = (0 <= idx) and (idx < len(options))
        return _pick(ok, options[idx], None)

    return _attempt(parse, lambda _exc: None)


def get_available_goals(profile: UserProfile) -> List[str]:
    return _pick(
        profile.pregnant,
        GOALS_BY_CONDITION["pregnancy"],
        GOALS_BY_CONDITION.get(profile.life_stage, GOALS_BY_CONDITION["adult"]),
    )


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


def _first_conflict(cond_set: set) -> Optional[str]:
    """أول رسالة تعارض مطابقة — بحث عودي بلا for/break."""
    def step(rest):
        empty = (len(rest) == 0)

        def go():
            combo, msg = rest[0]
            hit = combo.issubset(cond_set)
            return _pick(hit, msg, step(rest[1:]))

        return _do(empty, lambda: None, go)

    return step(list(IMPOSSIBLE_COMBOS))


def _validate_conditions(conditions: List[str], disease_list: List[str]) -> List[str]:
    """تحقق من التعارضات وأعد الاختيار عودياً عند وجود مشكلة (بديل while)."""
    conflict = _first_conflict(set(conditions))

    def ok():
        return conditions

    def reselect():
        print(f"\n  ❌  Conflict detected: {conflict}")
        print("  Please re-select your conditions.\n")
        new_conditions = _pick_multi(disease_list, DISEASE_EN, "Medical Conditions")
        return _validate_conditions(new_conditions, disease_list)

    return _do(conflict is None, ok, reselect)


# ══════════════════════════════════════════════════════════════
# دوال قراءة المدخلات (عودية بدل while)
# ══════════════════════════════════════════════════════════════

def _parse_age(raw: str):
    value = int(raw)
    ok = (1 <= value) and (value <= 110)
    return _pick(ok, value, None)


def _parse_choice(mapping: Dict[str, str]):
    """يُرجع دالة parse تُحقّق أن raw ضمن مفاتيح mapping."""
    def parse(raw: str):
        ok = raw in mapping
        return _pick(ok, mapping.get(raw), None)
    return parse


def _read_measurements():
    """اقرأ الطول والوزن معاً مع التحقق — عودي."""
    def attempt():
        height = float(input("  Height (cm): ").strip())
        weight = float(input("  Weight (kg): ").strip())
        ok = (50 <= height <= 250) and (2 <= weight <= 400)

        def good():
            return (height, weight)

        def bad():
            print("  Warning: Unrealistic values.")
            return _read_measurements()

        return _do(ok, good, bad)

    def on_err(_exc):
        print("  Warning: Please enter valid numbers.")
        return _read_measurements()

    return _attempt(attempt, on_err)


# ══════════════════════════════════════════════════════════════
# بناء الملف الشخصي تفاعلياً
# ══════════════════════════════════════════════════════════════

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
    age = _read_valid(
        "\n  Age (years): ",
        _parse_age,
        "  Warning: Age must be 1-110.",
        on_parse_fail="  Warning: Please enter a valid integer.",
    )

    stage = str(np.select(
        [age <= 12, age <= 17, age >= 65],
        ["child", "teen", "elderly"],
        default="adult",
    ))
    print(f"  OK  Age category: {AGE_RANGE_EN[stage]}")

    # ── الجنس ─────────────────────────────────────────────
    _divider("Gender")
    print("  1. Male\n  2. Female")
    gender = _read_valid(
        "  Select (1 or 2): ",
        _parse_choice({"1": "male", "2": "female"}),
        "  Warning: Select 1 or 2.",
    )

    # ── القياسات ──────────────────────────────────────────
    _divider("Measurements")
    height, weight = _read_measurements()

    h = height / 100
    bmi_tmp = round(weight / (h * h), 1)
    print(f"\n  BMI: {bmi_tmp}  |  Ideal range: "
          f"{round(18.5*h*h,1)}-{round(24.9*h*h,1)} kg")

    def child_note():
        print("  Note: BMI interpretation for children differs from adults.")
        print("        Please consult a pediatrician for accurate assessment.")
        return None
    _do(stage == "child", child_note, lambda: None)

    # ── مستوى النشاط البدني ───────────────────────────────
    activity_map = {
        "1": "sedentary", "2": "light", "3": "moderate",
        "4": "active", "5": "very_active",
    }

    def ask_activity():
        _divider("Physical Activity Level")
        print("  1. Sedentary   — desk job, little or no exercise")
        print("  2. Light       — light exercise 1-3 days/week")
        print("  3. Moderate    — moderate exercise 3-5 days/week")
        print("  4. Active      — hard exercise 6-7 days/week")
        print("  5. Very Active — athlete or physical labor job")
        chosen = _read_valid(
            "\n  Select (1-5): ",
            _parse_choice(activity_map),
            "  Warning: Select 1-5.",
        )
        print(f"  OK  Activity level: {chosen}")
        return chosen

    activity_level = _do(age >= 13, ask_activity, lambda: "light")

    # ── الحمل (للإناث فقط) ────────────────────────────────
    def ask_pregnant():
        _divider("Pregnancy")
        print("  1. Yes -- currently pregnant\n  2. No")
        return input("  Select (1 or 2): ").strip() == "1"

    can_pregnant = (gender == "female") and (age >= 17)
    pregnant = _do(can_pregnant, ask_pregnant, lambda: False)

    # ── الأمراض ───────────────────────────────────────────
    base_list = list(DISEASES_BY_STAGE[stage])
    female_extra = _pick(
        (gender == "female") and (stage in DISEASES_FEMALE_ONLY),
        DISEASES_FEMALE_ONLY.get(stage, []),
        [],
    )
    disease_list = base_list + female_extra

    conditions = _pick_multi(disease_list, DISEASE_EN, "Medical Conditions")
    conditions = _validate_conditions(conditions, disease_list)

    # ── الحساسيات — سؤال منفصل ────────────────────────────
    _divider("Food Allergies")
    print("  Do you have any food allergies?")
    print("  1. Yes -- I have food allergies")
    print("  2. No  -- I have no food allergies")
    has_allergy = input("\n  Select (1 or 2): ").strip()

    def ask_allergies():
        allergy_options = ALLERGIES_BY_STAGE.get(stage, list(ALLERGY_RULES.keys()))
        selected = _pick_multi(allergy_options, ALLERGY_EN, "Select Your Allergies")

        both = ("milk" in selected) and ("lactose_intolerance" in conditions)
        def info():
            print("\n  [INFO] You selected both 'Milk Allergy' and 'Lactose Intolerance'.")
            print("         These are different conditions but have similar dietary restrictions.")
            print("         The system will apply the stricter milk allergy rules.")
            return None
        _do(both, info, lambda: None)
        return selected

    def no_allergies():
        print("  OK  No allergies recorded.")
        return []

    allergies = _do(has_allergy == "1", ask_allergies, no_allergies)

    # ── التفضيلات ─────────────────────────────────────────
    preferences = _pick_multi(
        list(PREFERENCE_EN.keys()), PREFERENCE_EN, "Food Preferences"
    )

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

    meal_map = {"1": "breakfast", "2": "lunch", "3": "dinner", "4": "any"}
    meal_type = _read_valid(
        "\n  Select (1-4): ",
        _parse_choice(meal_map),
        "  Warning: Select 1-4.",
    )
    print(f"  OK  Meal type: {MEAL_TYPE_EN[meal_type]}")

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


# ══════════════════════════════════════════════════════════════
# طباعة النتائج
# ══════════════════════════════════════════════════════════════

def _num_or_zero(row, col):
    """قيمة العمود كعدد صحيح، أو 0 عند الغياب/NaN — بلا if/ternary."""
    v = row.get(col)
    is_nan = isinstance(v, float) and math.isnan(v)
    present = (v is not None) and (not is_nan)
    safe = _pick(present, v, 0)
    return int(float(safe)) * int(present)


def _rating_str(row):
    rating = row.get("Rating")
    is_nan = isinstance(rating, float) and math.isnan(rating)
    present = (rating is not None) and (not is_nan)
    return _pick(present, f"{float(_pick(present, rating, 0)):.1f}", "  - ")


def _mean_or_zero(df, col):
    present = col in df.columns
    return _do(present, lambda: df[col].mean(), lambda: 0)


def print_results(result: dict, top_n: int = 15):
    SEP  = "=" * 120
    sep2 = "-" * 120
    ps   = result["profile_summary"]

    print(f"\n{SEP}")
    print("  Smart Dietary Advisor -- Unified v4.0 -- Results".center(118))
    print(SEP)

    print(f"\n  Age:              {ps['age']} years  ({ps['life_stage']})")
    g_str = ps["gender"] + _pick(ps["pregnant"], " -- Pregnant", "")
    print(f"  Gender:           {g_str}")
    print(f"  Height / Weight:  {ps['height']} cm / {ps['weight']} kg")
    print(f"  BMI:              {ps['bmi']}  ->  {ps['bmi_category']}")
    print(f"  Ideal Weight:     {ps['ideal_weight']}")
    print(f"  Activity Level:   {ps.get('activity_level', 'light')}")
    print(f"  Meal Type:        {result.get('meal_type', 'any')}")

    # ── ملخص غذائي ───────────────────────────────────────
    daily  = ps["daily_calories"]
    meal   = result.get("target_meal_calories", ps["per_meal_kcal"])
    remain = daily - meal
    print(f"\n  Nutritional Summary:")
    print(f"     Daily Target  : {daily:,} kcal")
    print(f"     This Meal     : {meal:,} kcal  ({round(meal/daily*100)}% of daily)")
    print(f"     Remaining     : {remain:,} kcal for other meals")

    # ── نمط الأكل الصحي ───────────────────────────────────
    def show_style():
        hs = result["healthy_style"]
        print(f"\n  -- {hs['name']} --")
        _foreach(hs["description"].split("\n"), lambda line: print(f"  {line}"))
        print(f"  Source: {hs['source']}")
        return None
    _do(bool(result.get("healthy_style")), show_style, lambda: None)

    print(f"\n  Safe Recipes: {result['total_safe']:,} / {result['total_original']:,}"
          f"  ({result['filter_rate']}%)")

    # ── التحذيرات ─────────────────────────────────────────
    def show_warnings():
        print("\n  Warnings & Conflicts:")
        _foreach(result["warnings"], lambda w: print(f"     * {w}"))
        return None
    _do(bool(result["warnings"]), show_warnings, lambda: None)

    # ── القواعد السريرية ──────────────────────────────────
    def show_notes():
        shown_notes = result["notes"][:10]
        print(f"\n  Applied Clinical Rules "
              f"({len(result['notes'])} total, showing {len(shown_notes)}):")
        _foreach(shown_notes, lambda n: print(f"     * {n}"))
        return None
    _do(bool(result["notes"]), show_notes, lambda: None)

    print(f"\n  Top {top_n} Recommended Recipes:")
    print("  " + sep2)

    df = result["safe_recipes"]

    def empty_case():
        print("  No recipes match all restrictions.")
        print(SEP)
        return "DONE"

    def full_case():
        name_col = _pick("Name" in df.columns, "Name", df.columns[0])
        hdr = (f"  {'#':<3} {'Recipe Name':<50} "
               f"{'Cal':>6} {'Prot':>5} {'Carb':>5} {'Fat':>5} "
               f"{'Sod':>6} {'Fib':>5} {'Sug':>5} {'Rate':>5}")
        print(hdr)
        print("  " + sep2)

        head_df = df.head(top_n)
        rows = list(head_df.iterrows())
        limit = min(top_n, len(df))

        def print_row(_, indexed):
            pos, (_, row) = indexed
            i = pos + 1
            raw  = str(row.get(name_col, "Unknown"))
            long_name = len(raw) > 48
            name = _pick(long_name, raw[:46] + "..", raw.ljust(48))

            cal  = _num_or_zero(row, "Calories")
            prot = _num_or_zero(row, "ProteinContent")
            carb = _num_or_zero(row, "CarbohydrateContent")
            fat  = _num_or_zero(row, "FatContent")
            sod  = _num_or_zero(row, "SodiumContent")
            fib  = _num_or_zero(row, "FiberContent")
            sug  = _num_or_zero(row, "SugarContent")
            r_str = _rating_str(row)

            print(f"  {i:<3} {name:<50} "
                  f"{cal:>6} {prot:>5} {carb:>5} {fat:>5} "
                  f"{sod:>6} {fib:>5} {sug:>5} {r_str:>5}")

            _do("_reason" in row,
                lambda: print(f"       -> {row['_reason']}"),
                lambda: None)
            _do(i < limit, lambda: print(""), lambda: None)
            return None

        _fold(list(enumerate(rows)), None, print_row)

        # ── إحصائيات الوصفات المعروضة ─────────────────────
        shown_df = df.head(top_n)
        avg_cal  = _mean_or_zero(shown_df, "Calories")
        avg_prot = _mean_or_zero(shown_df, "ProteinContent")
        avg_sod  = _mean_or_zero(shown_df, "SodiumContent")

        print(f"\n  Average of top {top_n} recipes shown:")
        print(f"     Calories : {avg_cal:>6.0f} kcal")
        print(f"     Protein  : {avg_prot:>6.0f} g")
        print(f"     Sodium   : {avg_sod:>6.0f} mg")
        print(SEP)
        return "DONE"

    _do(df.empty, empty_case, full_case)