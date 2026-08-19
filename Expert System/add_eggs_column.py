from pathlib import Path
import pandas as pd

EGGS = {"egg", "eggs", "egg white", "egg yolk", "mayonnaise", "meringue", "albumin"}


def has_egg_trigger(raw_text) -> bool:
    is_missing = raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text))
    text = "" if is_missing else (
        " ".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
    )
    text_lower = text.lower()
    return any(term in text_lower for term in EGGS)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    input_path = data_dir / "cleaned_recipes.csv"
    output_path = data_dir / "cleaned_recipes_with_eggs.csv"

    print(f"  Reading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df):,} recipes")

    if "IngredientsList" not in df.columns:
        print("  Column 'IngredientsList' not found — aborting.")
        return

    print("  Computing HasEggs column...")
    df["HasEggs"] = df["IngredientsList"].apply(has_egg_trigger)

    egg_count = int(df["HasEggs"].sum())
    pct = round(egg_count / len(df) * 100, 1)
    print(f"   Found eggs in {egg_count:,} recipes ({pct}%)")

    print(f"  Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("  Done.")
    print()
    print("الخطوة التالية اليدوية:")
    print(f"  1. تحقق من الملف الجديد: {output_path}")
    print(f"  2. لو كل شي تمام، استبدل الملف القديم:")
    print(f"       احذف: {input_path}")
    print(f"       أعد تسمية: {output_path} → cleaned_recipes.csv")


if __name__ == "__main__":
    main()
