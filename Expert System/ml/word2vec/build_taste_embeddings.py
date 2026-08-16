import ast
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

BASE_DIR = Path(__file__).resolve().parents[2]        
DATA_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"
MODEL_PATH = BASE_DIR / "data" / "word2vec_ingredients.model"
EMBED_PATH = BASE_DIR / "data" / "recipe_taste_embeddings.pkl"

VECTOR_SIZE = 150
WINDOW = 6
MIN_COUNT = 5
EPOCHS = 15
SG = 1

SANITY_INGREDIENTS = [
    "chicken_breast",
    "olive_oil",
    "garlic",
    "soy_sauce",
    "chocolate",
    "basmati_rice",
]


def clean_ingredient_list(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        s = str(item).strip().lower()
        s = re.sub(r"\([^)]*\)", "", s)        
        s = re.sub(r",", " ", s)           
        s = re.sub(r"\s+", " ", s).strip()   
        s = re.sub(r" ", "_", s)             
        if s:
            out.append(s)
    return out


def main():
    t_total0 = time.time()

    print("Loading recipes...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    print("\nCleaning IngredientsList...")
    t0 = time.time()
    cleaned = df["IngredientsList"].apply(clean_ingredient_list)
    print(f"Cleaned in {time.time()-t0:.1f}s")

    n_empty = int((cleaned.str.len() == 0).sum())
    print(f"\nRecipes with EMPTY ingredient list (dropped from training): {n_empty:,}")
    dropped_ids = df.loc[cleaned.str.len() == 0, "RecipeId"].tolist()
    print(f"  (dropped RecipeIds count: {len(dropped_ids):,} — not saved, just reported)")

    train_mask = cleaned.str.len() > 0
    sentences = cleaned[train_mask].tolist()
    n_train = len(sentences)
    print(f"Training corpus: {n_train:,} recipes (kept)")

    n_tokens = int(sum(len(s) for s in sentences))
    print(f"Total cleaned tokens in corpus: {n_tokens:,}")

    print("\nTraining Word2Vec "
          f"(vector_size={VECTOR_SIZE}, window={WINDOW}, min_count={MIN_COUNT}, "
          f"epochs={EPOCHS}, sg={SG}, workers={os.cpu_count()})...")
    t0 = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=os.cpu_count(),
        epochs=EPOCHS,
        sg=SG,
        seed=42,
    )
    train_time = time.time() - t0
    print(f"Word2Vec training done in {train_time:.0f}s ({train_time/60:.1f} min)")

    vocab_size = len(model.wv)
    print(f"\nFinal vocabulary size: {vocab_size:,}")

    print("\n=== Sanity check: top-5 most_similar ===")
    for w in SANITY_INGREDIENTS:
        if w in model.wv:
            sims = model.wv.most_similar(w, topn=5)
            print(f"\n  {w}:")
            for other, score in sims:
                print(f"    {other:<30} {score:.4f}")
        else:
            print(f"\n  {w}: NOT IN VOCAB (below min_count=5)")

    print("\nSaving Word2Vec model...")
    model.save(str(MODEL_PATH))
    print(f"Saved -> {MODEL_PATH}")

    print("\nComputing recipe-level embeddings (mean of ingredient vectors)...")
    wv = model.wv
    t0 = time.time()
    embeddings = {}
    n_saved = 0
    n_no_vocab = 0
    no_vocab_ids = []
    for idx, tokens in zip(df.loc[train_mask, "RecipeId"], sentences):
        vecs = [wv[t] for t in tokens if t in wv]
        if len(vecs) > 0:
            embeddings[int(idx)] = np.vstack(vecs).mean(axis=0).astype(np.float32)
            n_saved += 1
        else:
            n_no_vocab += 1
            no_vocab_ids.append(int(idx))
    print(f"Computed {n_saved:,} recipe embeddings in {time.time()-t0:.1f}s")
    print(f"Recipes with NO in-vocab ingredient (all tokens below min_count): {n_no_vocab:,}")
    if no_vocab_ids:
        print("  example RecipeIds:", no_vocab_ids[:10])

    print("\nSaving recipe embeddings...")
    with open(EMBED_PATH, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {EMBED_PATH}")

    print(f"\n=== SUMMARY ===")
    print(f"Total wall time: {time.time()-t_total0:.0f}s")
    print(f"Training time: {train_time:.0f}s")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Recipes with embeddings saved: {n_saved:,}")
    print(f"Expected: {len(df) - n_empty:,} (384,541 - {n_empty:,} empty)")
    print(f"Difference (all-tokens-below-min_count): {n_no_vocab:,}")
    if n_saved != len(df) - n_empty:
        print("NOTE: saved-embedding count < expected because some recipes have no "
              "in-vocab ingredient tokens — this is expected and intentional.")
    else:
        print("OK: saved-embedding count matches expected.")

    for p in (MODEL_PATH, EMBED_PATH):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

