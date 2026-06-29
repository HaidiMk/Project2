"""
scorer.py — Smart Dietary Advisor v4.0 — DECLARATIVE EDITION
================================================================
نسخة معاد كتابتها بالكامل بأسلوب Declarative — بدون أي:
    if / elif / else / for / while  (statements)

الاستبدالات المستخدمة:
    for loop على weights / preferred  → sum() على generator
    if/elif متدرّج (سلالم شروط)        → دالة _tier() + bisect أو
                                           تعبير شرطي واحد محسوب رياضياً
    parsing "PT1H30M" بـ if "H" in t   → regex واحد يلتقط الكل دفعة واحدة
    try/except على الوقت               → دالة تعيد None بأمان (نُبقي
                                           try/except لأنه ليس if/for/while،
                                           هو معالجة استثناء لا فرع منطقي)
    تجميع reasons في explain_recipe    → list comprehension + فلترة None
"""

import re
import math
from typing import List, Optional
import pandas as pd

from rules.goals_and_preferences import GOAL_VECTORS

# ════════════════════════════════════════════════════════════════
# خريطة ترجمة خيار الواجهة → القيمة الحقيقية بعمود MealType
# ════════════════════════════════════════════════════════════════
# 🛠️ مطابقة لنفس الخريطة بـ filtering_engine.py — ضرورية هنا لأن
# _meal_type_score كانت تبحث عن "lunch"/"dinner" حرفياً بعمود
# MealType الذي لا يحوي هذه القيم أصلاً (فقط MainDish/Other/Drink/
# Breakfast)، فلا تُمنح أي وصفة نقطة meal_type أبداً عند هذين
# الخيارين تحديداً، رغم نجاح الفلتر (بعد إصلاحه) بإيجاد MainDish
# المطابقة. عدم التوحيد هنا كان يعني فلترة صحيحة لكن تقييم لا يميّز
# المطابقة الصحيحة بنقطة إضافية.
MEAL_TYPE_DATA_MAP = {
    "breakfast": "Breakfast",
    "lunch":     "MainDish",
    "dinner":    "MainDish",
}

# ════════════════════════════════════════════════════════════════
# أدوات عامة بدون أي if/for/while
# ════════════════════════════════════════════════════════════════

_ISO_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?")


def _is_missing(val) -> bool:
    """يعادل: val is None or (isinstance(val, float) and math.isnan(val))."""
    return val is None or (isinstance(val, float) and math.isnan(val))


def _safe_float(val, default: float = 0.0) -> float:
    return default if _is_missing(val) else float(val)


def _parse_minutes(raw) -> Optional[float]:
    """
    يحوّل TotalTime/PrepTime لدقائق — يدعم:
        - صيغة ISO 8601 ("PT1H30M", "PT45M", "PT2H")
        - رقم مباشر (45, "90")
    بدون أي if/elif statement — اعتماداً على ternary + regex + try/except
    (try/except هو معالجة استثناء، لا فرع منطقي شرطي).
    يرجع None عند الفشل أو عند raw=None.
    """
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


# ════════════════════════════════════════════════════════════════
# جداول التدرّج (Tiers) — تستبدل سلالم if/elif بالكامل بجدول بيانات
# ════════════════════════════════════════════════════════════════
# كل جدول: قائمة من (الحد الأعلى للفئة, القيمة المقابلة)، مرتبة تصاعدياً.
# آخر عنصر هو "الافتراضي" (شرط دائماً صحيح: float('inf')).

_CALORIE_DIFF_BONUS = [
    (100, 1.5),
    (200, 0.8),
    (300, 0.2),
    (400, 0.0),          # بين 300 و400: لا تغيير (الأصل لم يحدد حالة هنا)
    (float("inf"), None),  # يُعالَج بشكل خاص أدناه (عقوبة متناسبة لا قيمة ثابتة)
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
    # tiered is None فقط في حالة "أكبر من 400" (آخر صف) → نطبّق العقوبة
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


# ════════════════════════════════════════════════════════════════
# جدول تطبيع القيم الغذائية — يحل bug تفاوت المقاييس
# ════════════════════════════════════════════════════════════════
# 🛠️ مكتشَف بالاستخدام الفعلي: GOAL_VECTORS تحمل أوزاناً صغيرة
# (مثل ProteinContent: 0.9) مصمَّمة على فرض أنها تُضرَب بقيمة
# مُطبَّعة (0-1 تقريباً) — لكن _weighted_goal_score كانت تضربها
# مباشرة بالقيمة الخام (مثلاً Calories=999)، فتنتج
# 0.9 × 999 ≈ 899 نقطة من عنصر واحد فقط، تطغى بالكامل على باقي
# معايير full_score (تقييم، صحة، عقوبة) ذات المقياس الصغير (0-5
# تقريباً). النتيجة العملية: وصفات "مربى/جلي" بسعرات مرتفعة صدفوية
# كانت تتصدّر "Weight Gain" رغم بروتين=0 تماماً، لأن وزن Calories
# الموجب الكبير (0.9) يُخفي عقوبة صفر التغذية (-2.0) بسهولة.
#
# الحل: نُطبّع كل عمود لمقياس 0-1 تقريباً (بقسمته على حد أعلى واقعي
# طبياً لوجبة واحدة) قبل ضربه بالوزن — هذا يحافظ على المعنى النسبي
# لكل وزن (الأهم يبقى الأهم) دون أن يطغى عمود واحد على الباقي.
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
    """يطبّع قيمة غذائية خام لمقياس 0-1 تقريباً حسب حدها الأعلى الواقعي."""
    cap = _NORMALIZATION_CAPS.get(col, 1.0)
    return value / cap if cap else value


def _weighted_goal_score(row: "pd.Series", weights: dict) -> float:
    """
    يعادل حلقة for col,w in weights.items(): score += w*normalized_val
    — بدون for. القيمة تُطبَّع أولاً (انظر _normalize_value) قبل
    ضربها بالوزن، لمنع طغيان أي عمود واحد على باقي full_score.
    """
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
    """
    🛠️ إصلاح: meal_type القادم من الواجهة ("lunch"/"dinner") لا يطابق
    أي قيمة بعمود MealType الحقيقي (MainDish/Other/Drink/Breakfast).
    نترجمه أولاً عبر MEAL_TYPE_DATA_MAP قبل البحث، لتبقى نقطة المكافأة
    متّسقة مع نتائج الفلتر بـ filtering_engine.py.
    """
    real_value = MEAL_TYPE_DATA_MAP.get(meal_type.lower(), meal_type)
    meal_col = row.get("MealType")
    applies = (
        meal_type != "any"
        and not _is_missing(meal_col)
        and real_value.lower() in str(meal_col).lower()
    )
    return 1.0 if applies else 0.0


def _empty_calorie_penalty(row: "pd.Series") -> float:
    """
    عقوبة متدرّجة للوصفات "سعرات فارغة" (حلويات/حشوات/سكر مصنّع) التي
    تنافس وجبات حقيقية بالدرجة فقط بسبب قرب السعرات من الهدف.

    🛠️ تطوّر العقوبة عبر ثلاث محاولات (كل واحدة مكتشَفة بالاستخدام
    الفعلي على بيانات حقيقية):

      محاولة 1 (ثنائية: protein<1 AND fat<1): أوقفت "Mountain Dew
      Jelly" لكن فوّتت "Pineapple Cake Filling" (بروتين=3).

      محاولة 2 (macro_pct<15 AND fiber_density<1.5، استثناء كامل):
      أصلحت Cake Filling الأولى، لكن فوّتت "Raspberry Cake Filling"
      (بروتين=2، دهون=0، لكن ألياف=16 من بذور التوت المهروس) — ألياف
      الفاكهة المهروسة بالحشوة رفعت fiber_density فوق العتبة فألغت
      العقوبة بالكامل رغم أن macro_pct=1.3% (سعرات فارغة فعلياً).

      محاولة 3 (هذه): macro_pct هو المعيار الأساسي والأقوى (نسبة
      السعرات البنّاءة من بروتين+دهون). الألياف تُستخدم فقط كـ"تخفيف
      جزئي محدود" (حتى 50% كحد أقصى)، لا كاستثناء كامل ثنائي — هذا
      يحمي الفاكهة الطبيعية الخفيفة من عقوبة قصوى ظالمة، بينما يبقي
      عقوبة حقيقية متناسبة على أي حشوة/حلوى macro_pct منخفضة جداً
      بغض النظر عن وجود ألياف فاكهة مهروسة فيها.
    """
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
    fiber_relief = min(fiber_density / 5, 0.5)  # تخفيف جزئي، أقصاه 50%
    final_severity = base_severity * (1 - fiber_relief)

    return -3.0 * final_severity if has_data else 0.0


# ════════════════════════════════════════════════════════════════
# الدالة الرئيسية — score_recipe
# ════════════════════════════════════════════════════════════════

def score_recipe(
    row: "pd.Series",
    goal: Optional[str],
    target_meal_calories: int = 500,
    preferred: Optional[List[str]] = None,
    ing_col: Optional[str] = None,
    meal_type: str = "any",
) -> float:
    """
    احسب درجة الوصفة — بدون أي if/for/while statement.

    🛠️ إصلاح 1 (goal=None): عند عدم اختيار المستخدم هدفاً غذائياً،
    كانت النسخة الأصلية تُرجع Rating فقط (1-5)، متجاهلةً تماماً قرب
    السعرات والصحة والوقت. الآن full_score (كل المعايير) تُحتسب دائماً.

    🛠️ إصلاح 2 (عقوبة السعرات الفارغة، نسخة محسَّنة): وصفات "سعرات
    فارغة" (سكر بودرة، حشوة كيك، حلويات مصنّعة...) كانت تنافس وجبات
    حقيقية بالدرجة فقط بسبب قرب السعرات الصدفوي. _empty_calorie_penalty
    تجمع نسبة الماكرو البنّاء (بروتين+دهون) مع كثافة الألياف لمعاقبتها
    بتدرّج دقيق، دون معاقبة الفاكهة الطبيعية الخفيفة (ألياف عالية رغم
    بروتين منخفض) أو الوجبات النباتية الحقيقية.
    """
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


# ════════════════════════════════════════════════════════════════
# explain_recipe — نفس الأسلوب: قائمة قواعد بيانات بدل سلسلة if
# ════════════════════════════════════════════════════════════════

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
