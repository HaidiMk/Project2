# Data Artifacts — What's in Git vs. Regenerated Locally

Short guide for anyone who clones this repo fresh (teammates, reviewers).
Large/generated data files are **excluded from Git**; everything below is
reproducible locally from `recipes.csv` + the code in this repo.

## Artifact inventory

| Artifact | Size | In Git? | Regenerate via | Runtime | Depends on |
|---|---|---|---|---|---|
| `data/recipes.csv` | ~672 MB | **No** (`.gitignore`) | **Cannot regenerate** — raw dataset | — | External source: [Kaggle — Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) (download `RAW_recipes.csv` and save as `Expert System/data/recipes.csv`) |
| `data/cleaned_recipes.csv` | ~496 MB | **No** (`.gitignore`) | `python cleaner.py` (run from `Expert System/`) | not measured (large CSV) | `data/recipes.csv` |
| `data/cleaned_recipes_full.csv` | ~994 MB | **No** (`.gitignore`) | same as above (byproduct of `cleaner.py`) | not measured | `data/recipes.csv` |
| `ml/data/labeled_recipes.csv` | ~55 MB | **No** (ignored via blanket `*.csv` rule) | `python "Expert System/ml/health_classifier/build_labels.py"` (repo root) | not measured | `data/cleaned_recipes.csv` + medical rules (`rules/medical_rules.py`) |
| `data/recipe_taste_embeddings.pkl` | ~230 MB | **No** (explicit `.gitignore` entry — too large for GitHub without LFS) | `python "Expert System/ml/word2vec/build_taste_embeddings.py"` (repo root) | **~61 s** (cleaning 14 s + Word2Vec training 26 s + embedding loop 15 s) | `data/cleaned_recipes.csv` + `gensim>=4` installed |
| `data/word2vec_ingredients.model` | ~5.3 MB | **Yes** (committed normally — small) | Same script as above (byproduct of `build_taste_embeddings.py`) | included in the ~61 s | `gensim>=4` |
| `data/health_classifier.pt` | small | **Yes** (committed) | `python "Expert System/ml/health_classifier/health_classifier.py"` (see `ml/health_classifier/MODEL_DOCUMENTATION.md`) | not measured | `ml/data/labeled_recipes.csv`, PyTorch |
| `data/health_classifier_labels.json` | small | **Yes** (committed) | byproduct of classifier training | — | — |
| `data/health_classifier_thresholds.json` | small | **Yes** (committed) | byproduct of `ml/health_classifier/tune_thresholds.py` | — | classifier outputs |
| `data/health_scaler.pkl` | small | **Yes** (committed) | byproduct of classifier training | — | scikit-learn |

## Quick setup for a fresh clone (correct order)

From the repo root:

```powershell
# 1) Raw dataset — external download (cannot be regenerated from Git)
#    Save Kaggle's RAW_recipes.csv as:  Expert System/data/recipes.csv

# 2) Cleaned datasets (cleaner.py uses relative paths → run from Expert System/)
cd "Expert System"
python cleaner.py
cd ..

# 3) Labeled training data for the health classifier
python "Expert System/ml/health_classifier/build_labels.py"

# 4) Taste embeddings + Word2Vec model (required by the TOPSIS taste ranking)
pip install gensim
python "Expert System/ml/word2vec/build_taste_embeddings.py"
```

After steps 1–4 the pipeline runs fully: `python "Expert System/main.py"`,
`python TOPSIS/main.py`, `python TOPSIS/try_profile.py`, `python TOPSIS/sanity_check.py`.

## Git LFS status

- `.gitattributes` contains LFS rules for `*.csv` and `data/*.csv` (root-level `data/` folder only — they do **not** match `Expert System/data/`), but **no CSV is actually tracked** (`git ls-files "*.csv"` is empty) and **no LFS objects exist** (`git lfs ls-files` is empty).
- Current practice for large data is: **gitignore, don't commit** (neither plain nor LFS). This doc's ignore entry for `recipe_taste_embeddings.pkl` follows the same convention.

## Verification

```powershell
git status                          # recipe_taste_embeddings.pkl must NOT appear
git check-ignore -v "Expert System/data/recipe_taste_embeddings.pkl"   # shows matching rule
```
