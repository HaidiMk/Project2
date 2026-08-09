import sys, json
sys.path.insert(0, r'C:\Smart-Dietary-Advisor\Expert System')

import pandas as pd
from ml.health_classifier.explain import explain_health_score
from ml.health_classifier.health_classifier import NUTRITION_COLS

df = pd.read_csv(r'C:\Smart-Dietary-Advisor\Expert System\data\cleaned_recipes.csv', low_memory=False)

def explain_recipe(recipe_id, conditions):
    row = df.loc[df['RecipeId'] == recipe_id].iloc[0][NUTRITION_COLS]
    name = df.loc[df['RecipeId'] == recipe_id].iloc[0]['Name']
    print(f"\n{'='*70}")
    print(f"الوصفة: {name} (ID: {recipe_id})")
    print(f"{'='*70}")
    result = explain_health_score(row, conditions)
    print(json.dumps(result, indent=2, ensure_ascii=False))

# جرب وصفة بروتين عالي وسكر منخفض لمريض سكري
explain_recipe(49, ["diabetes"])

# جرب نفس الوصفة لأكتر من حالة مع بعض
explain_recipe(49, ["diabetes", "hypertension"])

# جرب وصفة حلوة (كيكة جزر) — لازم السكر يطلع "hurt"
explain_recipe(54, ["diabetes"])