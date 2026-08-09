import ast
import pandas as pd

print("جاري تحميل البيانات...")
df = pd.read_csv("Expert System/data/cleaned_recipes.csv", low_memory=False)
print("تم تحميل", len(df), "وصفة")


def normalize_ingredients(value) -> str:
    """تحويل المكونات إلى نص موحد وآمن للبحث."""
    if pd.isna(value):
        return ""
    if isinstance(value, list):
        return " | ".join(map(str, value)).lower()
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return " | ".join(map(str, parsed)).lower()
    except (ValueError, SyntaxError):
        pass
    return text.lower()


def inspect_recipe(df, recipe_name, ingredient="chicken"):
    """فحص وصفة محددة والتأكد من سبب نجاح فلتر المكونات."""
    required_columns = {"Name", "IngredientsList"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    matches = df[
        df["Name"].astype(str).str.strip().str.casefold()
        == recipe_name.strip().casefold()
    ].copy()

    if matches.empty:
        print(f"Recipe not found: {recipe_name}")
        return

    matches["normalized_ingredients"] = matches["IngredientsList"].apply(normalize_ingredients)
    matches["ingredient_found"] = matches["normalized_ingredients"].str.contains(
        ingredient.casefold(), regex=False, na=False
    )

    columns_to_show = [
        column for column in [
            "Name", "RecipeServings", "Calories", "ProteinContent",
            "SodiumContent", "IngredientsList", "ingredient_found",
        ]
        if column in matches.columns
    ]
    print(matches[columns_to_show].to_string(index=False))


def inspect_nutrition_semantics(df):
    nutrition_columns = [
        "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
        "SodiumContent", "CarbohydrateContent", "FiberContent",
        "SugarContent", "ProteinContent",
    ]
    available_nutrition = [c for c in nutrition_columns if c in df.columns]

    print("Available serving columns:")
    for column in ["RecipeServings", "RecipeYield"]:
        print(f"- {column}: {column in df.columns}")

    print("\nAvailable nutrition columns:")
    for column in available_nutrition:
        print(f"- {column}")

    if "RecipeServings" not in df.columns:
        print("\nRecipeServings is missing, so serving conversion cannot be performed directly.")
        return

    sample_columns = [
        c for c in ["Name", "RecipeServings", "Calories", "ProteinContent", "SodiumContent"]
        if c in df.columns
    ]
    sample = df[sample_columns].copy()
    sample["RecipeServings"] = pd.to_numeric(sample["RecipeServings"], errors="coerce")

    for column in available_nutrition:
        if column in sample.columns:
            sample[column] = pd.to_numeric(sample[column], errors="coerce")

    sample = sample[sample["RecipeServings"].notna() & (sample["RecipeServings"] > 0)].head(20)

    if "SodiumContent" in sample.columns:
        sample["sodium_if_total_recipe"] = sample["SodiumContent"] / sample["RecipeServings"]
    if "Calories" in sample.columns:
        sample["calories_if_total_recipe"] = sample["Calories"] / sample["RecipeServings"]

    print("\nSample for manual inspection:")
    print(sample.to_string(index=False))


print("\n" + "=" * 70)
print("الفحص الأول: مكونات Black Bean Soup")
print("=" * 70)
inspect_recipe(df, recipe_name="Easy, Low-Fat Black Bean Soup", ingredient="chicken")

print("\n" + "=" * 70)
print("الفحص الثاني: القيم الغذائية - للوصفة كاملة أم للحصة؟")
print("=" * 70)
inspect_nutrition_semantics(df)