"""
build_labels.py — توليد مجموعة بيانات مصنّفة (Multi-Label) للتدريب
===================================================================
يقرأ cleaned_recipes.csv ويولّد لكل وصفة 24 ملصقاً — واحد لكل حالة طبية
من MEDICAL_RULES (medical_rules.py):

    1. rule_label ∈ {0, 1}:
       - نطبّق حدود الحالة الرقمية (numeric_rules) حرفياً كما في النظام الخبير
       - مع الحد العام العالمي: ProteinContent <= 80 (مطبّق على كل الحالات)
       - أي قيمة مفقودة (NaN) تُعامَل كـ "غير آمنة" — نفس سلوك
         filtering_engine._apply_numeric_rule

    2. دمج التقييم كهدف ناعم (soft target):
       normalized_rating = (Rating - 1) / 4        # مقياس 1..5 → 0..1
       final_label = rule_label * (0.5 + 0.5 * normalized_rating)
       final_label = rule_label                    # عند غياب Rating

       لماذا هذه الصيغة؟
       - rule_label بوابة صلبة: وصفة "غير آمنة" (0) تبقى 0 مهما كان تقييمها.
       - الوصفة الآمنة (1) تترواح بين 0.5 (Rating=1) و 1.0 (Rating=5) —
         الهدف الناعم يضيف إشارة جودة/شعبية للترتيب داخل الحالات الآمنة
         دون كسر معنى الأمان الطبي الصارم.
       - الناتج ∈ {0} ∪ [0.5, 1.0]

ملاحظة: الأهداف الثمانية (weight_loss, muscle_gain ...) ليست ضمن هذا
التمرير — تستخدم score_weights لا حدوداً صارمة، وتُعالَج لاحقاً بشكل منفصل.

الحالات المختارة (24):
    كل مفاتيح MEDICAL_RULES ما عدا المفاتيح الأربعة المخصّصة فقط لدعم
    ALLERGY_RULES (nut_allergy, egg_allergy, seafood_allergy, sesame_allergy)
    — ليست حالات طبية قابلة للاختيار في constants (DISEASE_EN).
    لاحظ: lactose_intolerance و gluten_intolerance بلا numeric_rules
    (فلترة نصية فقط) → rule_label الخاص بها = 1 لكل الوصفات.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]          # Expert System/
sys.path.insert(0, str(BASE_DIR))

from rules.medical_rules import MEDICAL_RULES            # noqa: E402

DATA_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = OUT_DIR / "labeled_recipes.csv"

NUTRITION_COLS = [
    "Calories", "ProteinContent", "CarbohydrateContent", "SugarContent",
    "SodiumContent", "FatContent", "SaturatedFatContent",
    "CholesterolContent", "FiberContent",
]

PROTEIN_CAP = 80.0          # الحد العام العالمي — يُطبَّق على كل الحالات
RATING_MIN, RATING_MAX = 1.0, 5.0

# مفاتيح موجودة في MEDICAL_RULES فقط لدعم ALLERGY_RULES — ليست حالات طبية
ALLERGY_ONLY_KEYS = {"nut_allergy", "egg_allergy", "seafood_allergy", "sesame_allergy"}

CONDITIONS = {k: v for k, v in MEDICAL_RULES.items() if k not in ALLERGY_ONLY_KEYS}


def apply_numeric_rule(series: pd.Series, rule) -> pd.Series:
    """طبّق قاعدة رقمية واحدة (أو between) وتُرجع قناعاً منطقياً.
    NaN → False (غير آمن) — نفس سلوك محرك الفلترة."""
    op = rule[0]
    if op == "between":
        return series.between(rule[1], rule[2])
    val = rule[1]
    if op == "<=":
        return series <= val
    if op == ">=":
        return series >= val
    if op == "<":
        return series < val
    if op == ">":
        return series > val
    raise ValueError(f"Unsupported operator in rule: {rule}")


def condition_rule_mask(df: pd.DataFrame, rule_dict: dict) -> pd.Series:
    """قناع الأمان لحالة واحدة = تقاطع كل حدودها الرقمية."""
    mask = pd.Series(True, index=df.index)
    for col, rule in rule_dict.items():
        if col in df.columns:
            mask = mask & apply_numeric_rule(df[col], rule)
    return mask


def main():
    print("Loading recipes...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} recipes x {len(df.columns)} columns")

    for col in NUTRITION_COLS + ["Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    assert len(CONDITIONS) == 24, f"Expected 24 conditions, got {len(CONDITIONS)}"
    excluded = sorted(set(MEDICAL_RULES) - set(CONDITIONS))
    print(f"Conditions included: {len(CONDITIONS)}  |  excluded (allergy-rule support only): {excluded}")

    # ── التقييم: تطبيع 1..5 → 0..1 ومعامل المزج ────────────────────
    rating_available = df["Rating"].notna()
    normalized_rating = ((df["Rating"] - RATING_MIN) / (RATING_MAX - RATING_MIN)).clip(0.0, 1.0)
    blend_factor = 0.5 + 0.5 * normalized_rating

    # الحد العام العالمي — يُطبَّق على كل الحالات
    global_protein_ok = df["ProteinContent"] <= PROTEIN_CAP

    labels = pd.DataFrame(index=df.index)
    stats = []
    for key, rule in CONDITIONS.items():
        numeric = rule.get("numeric_rules", {})
        if numeric:
            numeric_ok = condition_rule_mask(df, numeric)
        else:
            # لا حدود رقمية لهذه الحالة (فلترة نصية فقط) → الكل آمن رقمياً
            numeric_ok = pd.Series(True, index=df.index)

        rule_label = (numeric_ok & global_protein_ok).astype(float)

        # المزج الناعم: آمن*تقييم حين يوجد تقييم، وإلا يبقى rule_label كما هو
        soft = rule_label * blend_factor
        final_label = soft.where(rating_available, rule_label)

        labels[f"label_{key}"] = final_label

        n_safe = int((rule_label == 1).sum())
        n_unsafe = len(df) - n_safe
        n_rated = int((rating_available & (rule_label == 1)).sum())
        stats.append((key, n_safe, n_unsafe, n_rated))

    # ── الحفظ: RecipeId + القيم الغذائية + ملصق لكل حالة ────────────
    out = pd.concat([df[["RecipeId"]], df[NUTRITION_COLS], labels], axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved labeled dataset ({out.shape[0]:,} rows x {out.shape[1]} columns) -> {OUT_PATH}")

    # ── إحصائيات التوزيع ────────────────────────────────────────────
    print(f"\nRecipes with Rating available to blend: {int(rating_available.sum()):,} of {len(df):,}")
    print("\n" + "=" * 92)
    print("Label distribution per condition")
    print("=" * 92)
    print(f"{'condition':<28} {'safe (1)':>11} {'unsafe (0)':>11} {'safe+rated':>11} {'rated%':>8}")
    print("-" * 92)
    for key, n_safe, n_unsafe, n_rated in stats:
        pct = n_rated / n_safe * 100 if n_safe else 0.0
        print(f"{key:<28} {n_safe:>11,} {n_unsafe:>11,} {n_rated:>11,} {pct:>7.1f}%")
    print("=" * 92)

    # ── عيّنة من الصفوف ─────────────────────────────────────────────
    print("\nSample rows (first 5 recipes, subset of columns):")
    subset = ["RecipeId", "Calories", "ProteinContent", "CarbohydrateContent",
              "SugarContent", "SodiumContent", "FatContent", "FiberContent"]
    sample_cols = subset + [f"label_{k}" for k in list(CONDITIONS)[:5]]
    print(out[sample_cols].head(5).to_string(index=False))

    print("\nFull label vector of the first recipe:")
    row0 = out.iloc[0]
    for k in CONDITIONS:
        print(f"  label_{k:<26} = {row0[f'label_{k}']:.4f}")


if __name__ == "__main__":
    main()
