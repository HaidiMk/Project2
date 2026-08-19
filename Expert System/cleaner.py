import pandas as pd
import numpy as np
import re
import json

print(" Loading dataset...")

df = pd.read_csv(
    "data/recipes.csv",
    low_memory=False,
    on_bad_lines="skip"
)

print(f"   Original recipes: {len(df):,}")

NUTRITION_COLS = [
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
]

print(" Converting nutrition values to numeric...")

for col in NUTRITION_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(" Fixing FiberContent (outlier removal)...")

if "FiberContent" in df.columns:
    df.loc[df["FiberContent"] > 50, "FiberContent"] = np.nan
    median_fiber = df["FiberContent"].median()
    df["FiberContent"] = df["FiberContent"].fillna(median_fiber)
    print(f"   Fiber fixed → mean: {df['FiberContent'].mean():.2f}, max: {df['FiberContent'].max():.2f}")

print(" FIX #1: Capping CholesterolContent outliers...")

if "CholesterolContent" in df.columns:
    outlier_count = (df["CholesterolContent"] > 600).sum()
    df.loc[df["CholesterolContent"] > 600, "CholesterolContent"] = np.nan
    median_chol = df["CholesterolContent"].median()
    df["CholesterolContent"] = df["CholesterolContent"].fillna(median_chol)
    print(f"   Capped {outlier_count:,} rows > 600 mg → median fill {median_chol:.1f}")
    print(f"   Cholesterol fixed → mean: {df['CholesterolContent'].mean():.1f}, max: {df['CholesterolContent'].max():.1f}")

print(" FIX #2: Capping SugarContent outliers...")

if "SugarContent" in df.columns:
    outlier_count = (df["SugarContent"] > 120).sum()
    df.loc[df["SugarContent"] > 120, "SugarContent"] = np.nan
    median_sugar = df["SugarContent"].median()
    df["SugarContent"] = df["SugarContent"].fillna(median_sugar)
    print(f"   Capped {outlier_count:,} rows > 120 g → median fill {median_sugar:.1f}")
    print(f"   Sugar fixed → mean: {df['SugarContent'].mean():.1f}, max: {df['SugarContent'].max():.1f}")

print(" Removing outliers...")

before = len(df)

df = df[
    (df["Calories"].fillna(0) > 0) &
    (df["Calories"].fillna(0) <= 8000)
]

for col in NUTRITION_COLS:
    if col in df.columns:
        df = df[df[col].fillna(0) >= 0]

after = len(df)
print(f"   Removed: {before - after:,} recipes → remaining: {after:,}")

print(" Smart normalization to PER SERVING values...")

servings_col = None
for col in ["RecipeServings", "Servings", "Yield", "NumberOfServings"]:
    if col in df.columns:
        servings_col = col
        break

if servings_col:
    print(f"   Using servings column: '{servings_col}'")
    df["Servings"] = pd.to_numeric(df[servings_col], errors="coerce")
else:
    print("   No servings column found - estimating from calories")
    df["Servings"] = (df["Calories"] / 500).round().clip(1, 12)

df["Servings"] = df["Servings"].fillna(4).replace(0, 4)
df["Servings"] = df["Servings"].clip(1, 24)

df["NeedsNormalization"] = df["Calories"] > 1500

print(f"   Recipes needing normalization: {df['NeedsNormalization'].sum():,}")

for col in NUTRITION_COLS:
    if col in df.columns:
        df.loc[df["NeedsNormalization"], col] = (
            df.loc[df["NeedsNormalization"], col] / df.loc[df["NeedsNormalization"], "Servings"]
        ).round(2)

print(f"   Nutrition values normalized intelligently")

print(" Creating validation flags...")

fiber_threshold = 50

df["IsValidServing"] = (
    (df["Calories"] >= 50) &
    (df["Calories"] <= 1200) &
    (df["ProteinContent"] <= 100) &
    (df["FiberContent"] <= fiber_threshold) &
    (df["SodiumContent"] <= 2000)
)

suspicious = (~df["IsValidServing"]).sum()
print(f"   Flagged {suspicious:,} recipes as suspicious serving size")

print("🧹 Cleaning ingredients...")

def parse_r_list(raw):
    if pd.isna(raw) or str(raw).strip() in ("character(0)", ""):
        return []
    return re.findall(r'"([^"]*)"', str(raw))

def parse_r_list_lower(raw):
    return [x.lower() for x in parse_r_list(raw)]

if "RecipeIngredientParts" in df.columns:
    df["IngredientsList"] = df["RecipeIngredientParts"].apply(parse_r_list_lower)
else:
    df["IngredientsList"] = [[] for _ in range(len(df))]

if "RecipeInstructions" in df.columns:
    df["InstructionsList"] = df["RecipeInstructions"].apply(parse_r_list)
else:
    df["InstructionsList"] = [[] for _ in range(len(df))]

if "Keywords" in df.columns:
    df["KeywordsList"] = df["Keywords"].apply(parse_r_list_lower)
else:
    df["KeywordsList"] = [[] for _ in range(len(df))]

print(" Classifying meal types...")

NOT_MAIN_DISH = {
    "dessert", "cake", "cookie", "pie", "ice cream", "pudding", "custard",
    "drink", "beverage", "juice", "smoothie", "cocktail", "tea", "coffee",
    "dip", "sauce", "dressing", "marinade", "spice", "seasoning", "rub",
    "appetizer", "snack", "candy", "brownie", "muffin", "pastry",
    "bread", "roll", "biscuit", "cracker",
    "jam", "jelly", "preserve", "butter", "spread",
    "syrup", "topping", "frosting", "glaze"
}

BREAKFAST_WORDS = {
    "oat", "oatmeal", "cereal", "parfait", "granola",
    "porridge", "muesli", "pancake", "waffle", "toast",
    "overnight oat", "breakfast bowl"
}

MEAL_TYPES = {
    "breakfast", "lunch", "dinner", "meal", "main dish", "entree",
    "stew", "soup", "curry", "casserole", "roast", "bake",
    "stir fry", "saute", "grill", "chili"
}

DRINK_WORDS = {
    "juice", "smoothie", "shake", "drink", "beverage", "lemonade",
    "cocktail", "mocktail", "tea", "coffee", "latte", "frappe",
    "milkshake", "punch", "agua", "horchata"
}

def classify_meal_type(row):
    name     = str(row.get("Name", "")).lower()
    category = str(row.get("RecipeCategory", "")).lower()
    keywords = row.get("KeywordsList", [])
    ingredients = row.get("IngredientsList", [])

    all_text = f"{name} {category} {' '.join(keywords[:5])} {' '.join(ingredients[:3])}"

    for word in DRINK_WORDS:
        if word in all_text:
            return "Drink"

    for word in BREAKFAST_WORDS:
        if word in all_text:
            return "Breakfast"

    for word in NOT_MAIN_DISH:
        if word in all_text:
            return "Other"

    for word in MEAL_TYPES:
        if word in all_text:
            return "MainDish"

    has_protein = any(p in all_text for p in ["chicken", "beef", "fish", "tofu", "lentil", "bean", "pork", "turkey"])
    cal = row.get("Calories", 0)

    if has_protein and 200 <= cal <= 900:
        return "MainDish"

    return "Other"

df["MealType"] = df.apply(classify_meal_type, axis=1)

print(f"   MainDish:  {(df['MealType'] == 'MainDish').sum():,}")
print(f"   Drink:     {(df['MealType'] == 'Drink').sum():,}")
print(f"   Breakfast: {(df['MealType'] == 'Breakfast').sum():,}")
print(f"   Other:     {(df['MealType'] == 'Other').sum():,}")

print(" Cleaning images...")

def extract_image_urls(raw):
    if pd.isna(raw) or str(raw).strip() in ("character(0)", ""):
        return []
    return re.findall(r'"(https?://[^"]+)"', str(raw))

if "Images" in df.columns:
    all_images = df["Images"].apply(extract_image_urls)
else:
    all_images = [[] for _ in range(len(df))]

df["ImageUrl"]    = all_images.apply(lambda x: x[0] if x else "")
df["ImagesCount"] = all_images.apply(len)
df["ImagesJson"]  = all_images.apply(lambda x: json.dumps(x) if x else "[]")

for col in ["Name", "AuthorName", "Description", "RecipeCategory"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

print(" Creating MainCategory...")

def map_main_category(row):
    cat         = str(row.get("RecipeCategory", "")).lower()
    name        = str(row.get("Name", "")).lower()
    meal_type   = row.get("MealType", "Other")
    ingredients = row.get("IngredientsList", [])
    combined    = f"{cat} {name} {' '.join(ingredients[:5])}"

    if meal_type == "Drink":
        return "Drink"

    if meal_type == "Breakfast":
        return "Breakfast"

    if meal_type == "MainDish":
        return "Main Dish"

    main_ingredients = [
        "chicken", "beef", "pork", "lamb", "fish", "salmon", "tuna", "shrimp",
        "tofu", "rice", "pasta", "potato", "steak", "roast", "lentil", "bean",
        "turkey", "duck", "veal", "bacon", "sausage", "ham", "egg", "eggs"
    ]
    if any(ing in combined for ing in main_ingredients):
        return "Main Dish"

    drink_keywords = ["drink", "juice", "smoothie", "cocktail", "beverage", "tea", "coffee", "shake", "latte"]
    if any(x in combined for x in drink_keywords):
        return "Drink"

    dessert_keywords = ["dessert", "cake", "cookie", "pie", "ice cream", "pudding", "custard", "brownie", "muffin", "pastry", "candy"]
    if any(x in combined for x in dessert_keywords):
        return "Dessert"

    bakery_keywords = ["bread", "baking", "pastry", "roll", "biscuit", "cracker", "bagel", "croissant"]
    if any(x in combined for x in bakery_keywords):
        return "Bakery"

    healthy_keywords = ["salad", "vegetable", "vegan", "healthy", "bowl"]
    if any(x in combined for x in healthy_keywords):
        return "Healthy"

    snack_keywords = ["snack", "appetizer", "dip", "finger food"]
    if any(x in combined for x in snack_keywords):
        return "Snack"

    return "Other"

df["MainCategory"] = df.apply(map_main_category, axis=1)

print(" FIX #4: Syncing MealType with MainCategory (eliminating mismatch)...")

df["MealType_original"] = df["MealType"]

MAINCATEGORY_TO_MEALTYPE = {
    "Main Dish": "MainDish",
    "Drink":     "Drink",
    "Breakfast": "Breakfast",
    "Dessert":   "Other",
    "Bakery":    "Other",
    "Healthy":   "MainDish",
    "Snack":     "Other",
    "Other":     "Other",
}

df["MealType"] = df["MainCategory"].map(MAINCATEGORY_TO_MEALTYPE).fillna("Other")

mismatch_remaining = (
    (df["MealType"] == "Other") & (df["MainCategory"] == "Main Dish")
).sum()
print(f"   MealType/MainCategory mismatch after sync: {mismatch_remaining} (was ~130K)")

print(" Creating DietType...")

MEAT    = {"chicken", "beef", "pork", "lamb", "turkey", "meat", "bacon", "ham", "sausage"}
SEAFOOD = {"fish", "salmon", "tuna", "shrimp", "crab", "seafood", "prawn", "lobster"}
DAIRY   = {"milk", "cheese", "cream", "butter", "yogurt"}
EGGS    = {"egg", "eggs"}

def classify_diet(ingredients):
    text = " ".join(ingredients)
    if any(x in text for x in MEAT):
        return "Meat"
    elif any(x in text for x in SEAFOOD):
        return "Seafood"
    elif any(x in text for x in DAIRY) or any(x in text for x in EGGS):
        return "Vegetarian"
    else:
        return "Vegan"

df["DietType"] = df["IngredientsList"].apply(classify_diet)

print(" Calculating allergy flags...")

LACTOSE          = {"milk", "cheese", "cream", "butter", "yogurt", "dairy", "whey", "casein"}
GLUTEN           = {"wheat", "barley", "oat", "rye", "flour", "bread", "pasta", "noodle", "spaghetti"}
NUTS             = {"peanut", "almond", "walnut", "cashew", "pistachio", "hazelnut", "pecan"}
SOY              = {"soy", "soya", "tofu", "tempeh", "miso", "edamame"}
SEAFOOD_ALLERGEN = {"fish", "salmon", "tuna", "shrimp", "crab", "lobster", "seafood", "prawn"}

def has_trigger(lst, triggers):
    return any(any(t in i for t in triggers) for i in lst)

df["HasLactose"] = df["IngredientsList"].apply(lambda x: has_trigger(x, LACTOSE))
df["HasGluten"]  = df["IngredientsList"].apply(lambda x: has_trigger(x, GLUTEN))
df["HasNuts"]    = df["IngredientsList"].apply(lambda x: has_trigger(x, NUTS))
df["HasSoy"]     = df["IngredientsList"].apply(lambda x: has_trigger(x, SOY))
df["HasSeafood"] = df["IngredientsList"].apply(lambda x: has_trigger(x, SEAFOOD_ALLERGEN))

print(" Creating HealthScore...")

def health_score(row):
    score = 70

    sugar = row.get("SugarContent", 0)
    if sugar > 25:   score -= 20
    elif sugar > 15: score -= 10

    sodium = row.get("SodiumContent", 0)
    if sodium > 800:   score -= 20
    elif sodium > 500: score -= 10

    sat_fat = row.get("SaturatedFatContent", 0)
    if sat_fat > 10:  score -= 15
    elif sat_fat > 5: score -= 5

    fiber = row.get("FiberContent", 0)
    if fiber > 8:   score += 15
    elif fiber > 5: score += 10
    elif fiber > 2: score += 5

    protein = row.get("ProteinContent", 0)
    if protein > 25:   score += 10
    elif protein > 15: score += 5

    return max(0, min(100, score))

df["HealthScore"] = df.apply(health_score, axis=1)

def risk_level(score):
    if score >= 70: return "LOW"
    if score >= 50: return "MEDIUM"
    return "HIGH"

df["RiskLevel"] = df["HealthScore"].apply(risk_level)

df["DifficultyScore"] = (
    df["IngredientsList"].apply(len) * 0.3 +
    df["InstructionsList"].apply(len) * 0.3
).clip(1, 10).round(1)

print(" FIX #3: Parsing TimeCategory with correct ISO 8601 parser...")

def parse_iso8601_duration_minutes(s):
    if not s or pd.isna(s):
        return None
    s = str(s).strip().upper()
    if s in ("", "PT", "P"):
        return None

    total_minutes = 0

    day_match = re.search(r'(\d+)D', s)
    if day_match:
        total_minutes += int(day_match.group(1)) * 1440

    hour_match = re.search(r'(\d+)H', s)
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60

    min_match = re.search(r'(\d+)M', s)
    if min_match:
        t_pos = s.find('T')
        m_pos = s.find('M', t_pos if t_pos >= 0 else 0)
        if m_pos >= 0:
            total_minutes += int(min_match.group(1))

    return total_minutes if total_minutes > 0 else None

def time_category(row):
    for col in ["TotalTime", "CookTime", "PrepTime"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ("", "PT", "P"):
            minutes = parse_iso8601_duration_minutes(str(val))
            if minutes is not None:
                if minutes <= 30:  return "Quick"
                if minutes <= 60:  return "Medium"
                return "Long"
    return "Unknown"

df["TimeCategory"] = df.apply(time_category, axis=1)

print(f"   TimeCategory distribution:")
print(df["TimeCategory"].value_counts().to_string())

print(" Removing duplicates...")

before = len(df)
df     = df.drop_duplicates(subset=["Name"])
after  = len(df)
print(f"   Removed {before - after:,} duplicates")

print(" Filtering for expert system ready recipes (ALL categories)...")

df_expert_ready = df[
    df["IsValidServing"] &
    (df["Calories"] >= 30) &
    (df["Calories"] <= 1000)
].copy()

print(f"   Expert system ready recipes: {len(df_expert_ready):,}")
print("\n By category:")
print(df_expert_ready["MainCategory"].value_counts())

NUTRITION_COLS_FINAL = [
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
]

KEEP = [
    "RecipeId", "Name", "AuthorId", "AuthorName",
    "CookTime", "PrepTime", "TotalTime", "DatePublished",
    "Description", "RecipeCategory", "MainCategory", "DietType", "MealType",
    "AggregatedRating", "ReviewCount",
    "Servings", "IsValidServing",
    *NUTRITION_COLS_FINAL,
    "IngredientsList", "InstructionsList", "KeywordsList",
    "ImageUrl", "ImagesCount", "ImagesJson",
    "HasLactose", "HasGluten", "HasNuts", "HasSoy", "HasSeafood",
    "HealthScore", "RiskLevel", "DifficultyScore", "TimeCategory"
]

KEEP     = [c for c in KEEP if c in df_expert_ready.columns]
df_clean = df_expert_ready[KEEP].copy()

df_clean = df_clean.rename(columns={
    "AggregatedRating": "Rating",
    "ReviewCount":      "NumReviews"
})

print(" Saving files...")

df_clean.to_csv("data/cleaned_recipes.csv", index=False)
df.to_csv("data/cleaned_recipes_full.csv", index=False)

print("    data/cleaned_recipes.csv     (all categories — expert system ready)")
print("    data/cleaned_recipes_full.csv (raw full dataset)")

print("\n" + "═" * 60)
print(" CLEANING COMPLETED (v3)")
print("═" * 60)
print(f"Expert system ready recipes: {len(df_clean):,}")
print(f"Columns: {len(df_clean.columns)}")

print("\n Category breakdown:")
print(df_clean["MainCategory"].value_counts())

print("\n DietType distribution:")
print(df_clean["DietType"].value_counts())

print("\n TimeCategory distribution (fixed):")
print(df_clean["TimeCategory"].value_counts())

print("\n Nutrition per serving statistics:")
print("-" * 50)
for col in ["Calories", "ProteinContent", "FiberContent", "SodiumContent", "SugarContent", "CholesterolContent"]:
    if col in df_clean.columns:
        print(f"{col:<22}: {df_clean[col].mean():.1f} ± {df_clean[col].std():.1f}  (max: {df_clean[col].max():.1f})")

print("\n Fixes applied in v3:")
print("   FIX #1: CholesterolContent capped at 600 mg (was max 2,544)")
print("   FIX #2: SugarContent capped at 120 g (was max 253)")
print("   FIX #3: TimeCategory uses correct ISO 8601 hour/minute parser")
print("   FIX #4: MealType now derived from MainCategory (0 mismatch rows)")

sample = df_clean.sample(1).iloc[0]
print("\n SAMPLE RECIPE")
for col in ["Name", "MealType", "MainCategory", "DietType", "Calories",
            "ProteinContent", "HealthScore", "Rating", "TimeCategory"]:
    if col in sample.index:
        value = sample[col]
        if isinstance(value, float):
            value = f"{value:.1f}"
        print(f"{col:<22} : {value}")