"""
eda_report.py — Exploratory Data Analysis
==========================================
تحليل شامل لقاعدة بيانات الوصفات
شغّل بـ: python eda_report.py
"""

import pandas as pd
import numpy as np
import os

os.makedirs("output", exist_ok=True)

print("\n" + "=" * 60)
print("  Smart Dietary Advisor — EDA Report")
print("=" * 60)

# ── تحميل البيانات ─────────────────────────────────────────
print("\n  Loading data...")
df = pd.read_csv("data/cleaned_recipes.csv", low_memory=False)
print(f"  Total recipes: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")

# ── 1. معلومات أساسية ──────────────────────────────────────
print("\n" + "-" * 60)
print("  1. Basic Information")
print("-" * 60)
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ── 2. القيم المفقودة ──────────────────────────────────────
print("\n" + "-" * 60)
print("  2. Missing Values")
print("-" * 60)
nutrient_cols = [
    "Calories", "ProteinContent", "FatContent",
    "CarbohydrateContent", "FiberContent", "SugarContent",
    "SodiumContent", "SaturatedFatContent", "CholesterolContent",
]
for col in nutrient_cols:
    if col in df.columns:
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        print(f"  {col:<25}: {missing:>7,} missing ({pct:.1f}%)")

# ── 3. إحصائيات القيم الغذائية ─────────────────────────────
print("\n" + "-" * 60)
print("  3. Nutritional Statistics")
print("-" * 60)
print(f"  {'Column':<25} {'Min':>8} {'Mean':>8} {'Median':>8} {'Max':>8}")
print("  " + "-" * 57)
for col in nutrient_cols:
    if col in df.columns:
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        print(f"  {col:<25} {data.min():>8.1f} {data.mean():>8.1f} "
              f"{data.median():>8.1f} {data.max():>8.1f}")

# ── 4. توزيع التقييمات ─────────────────────────────────────
print("\n" + "-" * 60)
print("  4. Rating Distribution")
print("-" * 60)
if "Rating" in df.columns:
    ratings = pd.to_numeric(df["Rating"], errors="coerce").dropna()
    print(f"  Total rated recipes : {len(ratings):,}")
    print(f"  Average rating      : {ratings.mean():.2f}")
    print(f"  5-star recipes      : {(ratings == 5).sum():,} ({(ratings == 5).mean()*100:.1f}%)")
    print(f"  4-star recipes      : {(ratings >= 4).sum():,} ({(ratings >= 4).mean()*100:.1f}%)")
    print(f"  Below 3-star        : {(ratings < 3).sum():,} ({(ratings < 3).mean()*100:.1f}%)")

# ── 5. فئات الوصفات ────────────────────────────────────────
print("\n" + "-" * 60)
print("  5. Top Recipe Categories")
print("-" * 60)
if "RecipeCategory" in df.columns:
    top_cats = df["RecipeCategory"].value_counts().head(10)
    for cat, count in top_cats.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {str(cat):<30} {count:>7,} ({pct:>5.1f}%) {bar}")

# ── 6. توزيع السعرات ───────────────────────────────────────
print("\n" + "-" * 60)
print("  6. Calorie Distribution")
print("-" * 60)
if "Calories" in df.columns:
    cals = pd.to_numeric(df["Calories"], errors="coerce").dropna()
    ranges = [
        ("< 200 kcal (very low)",  cals < 200),
        ("200-400 kcal (low)",     (cals >= 200) & (cals < 400)),
        ("400-600 kcal (medium)",  (cals >= 400) & (cals < 600)),
        ("600-800 kcal (high)",    (cals >= 600) & (cals < 800)),
        ("> 800 kcal (very high)", cals >= 800),
    ]
    for label, mask in ranges:
        count = mask.sum()
        pct = count / len(cals) * 100
        bar = "█" * int(pct / 3)
        print(f"  {label:<30} {count:>7,} ({pct:>5.1f}%) {bar}")

# ── 7. الحساسيات في البيانات ───────────────────────────────
print("\n" + "-" * 60)
print("  7. Allergen Presence in Database")
print("-" * 60)
allergy_cols = {
    "HasLactose": "Contains Lactose",
    "HasGluten":  "Contains Gluten",
    "HasNuts":    "Contains Nuts",
    "HasSoy":     "Contains Soy",
    "HasSeafood": "Contains Seafood",
}
for col, label in allergy_cols.items():
    if col in df.columns:
        count = df[col].sum()
        pct = count / len(df) * 100
        print(f"  {label:<25}: {count:>7,} ({pct:.1f}%)")

# ── 8. إحصائيات نظام الخبير ────────────────────────────────
print("\n" + "-" * 60)
print("  8. Expert System Coverage")
print("-" * 60)

diabetic_safe = df[
    (pd.to_numeric(df.get("SugarContent", pd.Series()), errors="coerce") <= 15) &
    (pd.to_numeric(df.get("CarbohydrateContent", pd.Series()), errors="coerce") <= 60)
]
print(f"  Safe for Diabetes     : {len(diabetic_safe):>7,} ({len(diabetic_safe)/len(df)*100:.1f}%)")

hyper_safe = df[
    pd.to_numeric(df.get("SodiumContent", pd.Series()), errors="coerce") <= 600
]
print(f"  Safe for Hypertension : {len(hyper_safe):>7,} ({len(hyper_safe)/len(df)*100:.1f}%)")

heart_safe = df[
    (pd.to_numeric(df.get("SodiumContent", pd.Series()), errors="coerce") <= 500) &
    (pd.to_numeric(df.get("SaturatedFatContent", pd.Series()), errors="coerce") <= 4)
]
print(f"  Safe for Heart Disease: {len(heart_safe):>7,} ({len(heart_safe)/len(df)*100:.1f}%)")

both_safe = df[
    (pd.to_numeric(df.get("SugarContent", pd.Series()), errors="coerce") <= 15) &
    (pd.to_numeric(df.get("SodiumContent", pd.Series()), errors="coerce") <= 500)
]
print(f"  Safe for Diab+Hypert  : {len(both_safe):>7,} ({len(both_safe)/len(df)*100:.1f}%)")

# ── حفظ التقرير ────────────────────────────────────────────
report_lines = []
report_lines.append("Smart Dietary Advisor — EDA Report")
report_lines.append(f"Total Recipes: {len(df):,}")
report_lines.append(f"Total Columns: {len(df.columns)}")

with open("output/eda_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n" + "=" * 60)
print("  EDA Complete — report saved to output/eda_report.txt")
print("=" * 60)