import sys, re
sys.path.insert(0, ".")
import pandas as pd

df = pd.read_csv("data/cleaned_recipes.csv", low_memory=False)

halal_words = ["pork", "bacon", "ham", "wine", "beer", "alcohol",
               "lard", "pepperoni", "salami", "prosciutto"]

for word in halal_words:
    pattern = r"\b" + re.escape(word) + r"\b"
    count = df["Name"].fillna("").str.lower().str.contains(pattern, regex=True).sum()
    if count > 0:
        samples = df[df["Name"].fillna("").str.lower().str.contains(
            pattern, regex=True)]["Name"].head(3).tolist()
        print(f"{word}: {count} — {samples}")