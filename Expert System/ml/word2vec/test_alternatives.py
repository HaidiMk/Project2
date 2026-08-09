import sys
import json

sys.path.insert(0, r'C:\Smart-Dietary-Advisor\Expert System')

import pandas as pd
from main import load_data
from engine.filtering_engine import DietaryExpertSystem
from core.user_profile import UserProfile
from ml.word2vec.alternatives import suggest_alternatives


print("Loading database...")
df = load_data()
system = DietaryExpertSystem(df)

# Create a profile for a user with a peanut allergy
profile = UserProfile(
    age=30,
    height=165,
    weight=60,
    gender='female',
    allergies=['peanuts']
)

result = system.filter_recipes(profile)
safe = result['safe_recipes']

print(f"Number of safe recipes for this user: {len(safe):,}\n")


def try_alternative(recipe_id):
    blocked_row = df.loc[df['RecipeId'] == recipe_id]

    if blocked_row.empty:
        print(f"RecipeId {recipe_id} does not exist")
        return

    blocked = blocked_row.iloc[0]

    print(f"{'=' * 70}")
    print(f"Blocked recipe: {blocked['Name']} (ID: {recipe_id})")
    print(f"{'=' * 70}")

    result = suggest_alternatives(
        blocked,
        safe,
        top_n=3
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


# Test a recipe containing peanuts
try_alternative(508281)   # Creamy Peanut Dessert

# Test another peanut-containing recipe
try_alternative(220974)   # Super Easy Peanut Noodles
# الحالة الصعبة — وصفة فقيرة بالمكونات (مكوّنين بس)
try_alternative(218)