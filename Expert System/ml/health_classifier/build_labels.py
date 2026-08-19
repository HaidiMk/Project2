import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]       
sys.path.insert(0, str(BASE_DIR))

from rules.medical_rules import MEDICAL_RULES            

DATA_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"   
OUT_PATH = OUT_DIR / "labeled_recipes.csv"

NUTRITION_COLS = [
    "Calories", "ProteinContent", "CarbohydrateContent", "SugarContent",
    "SodiumContent", "FatContent", "SaturatedFatContent",
    "CholesterolContent", "FiberContent",
]

PROTEIN_CAP = 80.0        
RATING_MIN, RATING_MAX = 1.0, 5.0

ALLERGY_ONLY_KEYS = {"nut_allergy", "egg_allergy", "seafood_allergy", "sesame_allergy"}

CONDITIONS = {k: v for k, v in MEDICAL_RULES.items() if k not in ALLERGY_ONLY_KEYS}


def apply_numeric_rule(series: pd.Series, rule) -> pd.Series:
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

    rating_available = df["Rating"].notna()
    normalized_rating = ((df["Rating"] - RATING_MIN) / (RATING_MAX - RATING_MIN)).clip(0.0, 1.0)
    blend_factor = 0.5 + 0.5 * normalized_rating

    global_protein_ok = df["ProteinContent"] <= PROTEIN_CAP

    labels = pd.DataFrame(index=df.index)
    stats = []
    for key, rule in CONDITIONS.items():
        numeric = rule.get("numeric_rules", {})
        if numeric:
            numeric_ok = condition_rule_mask(df, numeric)
        else:
            numeric_ok = pd.Series(True, index=df.index)

        rule_label = (numeric_ok & global_protein_ok).astype(float)

        soft = rule_label * blend_factor
        final_label = soft.where(rating_available, rule_label)

        labels[f"label_{key}"] = final_label

        n_safe = int((rule_label == 1).sum())
        n_unsafe = len(df) - n_safe
        n_rated = int((rating_available & (rule_label == 1)).sum())
        stats.append((key, n_safe, n_unsafe, n_rated))

    out = pd.concat([df[["RecipeId"]], df[NUTRITION_COLS], labels], axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved labeled dataset ({out.shape[0]:,} rows x {out.shape[1]} columns) -> {OUT_PATH}")

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
