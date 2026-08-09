"""
alternatives.py — taste-based "safe alternative recipes" (standalone, opt-in)
================================================================================

Given a BLOCKED recipe (allergen / medical-rule violation) and a user's own
already-computed SAFE recipe set (e.g. the output of
DietaryExpertSystem.filter_recipes()['safe_recipes']), ranks the safe set by
taste similarity to the blocked recipe using the pre-trained Word2Vec recipe
embeddings (Expert System/data/recipe_taste_embeddings.pkl). Candidates are
drawn ONLY from the caller-supplied safe set, so medical safety is inherited
for free — this module adds no safety logic of its own.

Pure additive module: nothing in the scoring/ranking pipeline imports this
file; a backend (e.g. Django) calls it explicitly as a separate endpoint, the
same integration pattern as explain_health_score().

Method (validated in exploration PoC):
    cosine similarity between the blocked recipe's L2-normalized taste
    embedding and each candidate's embedding, vectorized against one
    process-wide aligned matrix of all 383,293 available embeddings.
    Well-populated recipes (>= min_vocab_tokens in-vocab ingredient tokens)
    yield coherent, sensible neighbors via this path. Recipes with very few
    in-vocab tokens (e.g. a 2-ingredient list) produce flat, near-tied,
    nutritionally nonsensical rankings, so they automatically take a
    NUTRITION-ONLY fallback instead (z-scored Euclidean distance over the same
    9 NUTRITION_COLS used by health_classifier, corpus mean/std cached from
    cleaned_recipes.csv).

Input validation discipline (same as explain.py): missing required columns or
non-numeric nutrition values raise ValueError — nothing is silently imputed.

==============  API CONTRACT (backend consumption, no ML knowledge needed)  ==============

suggest_alternatives(blocked_recipe, safe_recipes_df, top_n=3,
                     min_vocab_tokens=4) -> dict

    blocked_recipe:  pandas Series / one-row DataFrame, or plain dict,
                     supplying RecipeId (int), Name, IngredientsList, and the 9
                     nutrition columns (Calories, ProteinContent,
                     CarbohydrateContent, SugarContent, SodiumContent,
                     FatContent, SaturatedFatContent, CholesterolContent,
                     FiberContent). Missing column or NaN value -> ValueError.
    safe_recipes_df: pandas DataFrame — the caller's already-computed safe set
                     (must contain RecipeId, Name, and the same 9 nutrition
                     columns). Medical safety is the caller's responsibility:
                     only recipes in this frame are ever returned.
    top_n:           how many alternatives to return (default 3).
    min_vocab_tokens: minimum number of in-vocab ingredient tokens the blocked
                     recipe must have for TASTE similarity to be trustworthy
                     (default 4). Below this (or no embedding at all) the
                     NUTRITION fallback is used.

    Return (all values JSON-serializable, numpy types converted to native)::

        {
          "method": "taste_similarity",   # or "nutrition_fallback"
          "blocked_recipe": {"RecipeId": 218, "Name": "..."},
          "requested_top_n": 3,
          "returned_count": 3,            # may be < top_n (fewer candidates)
          "reason": "",                   # non-empty when returned_count < top_n
          "alternatives": [
            {"RecipeId": 123, "Name": "...", "score": 0.9612,
             "Calories": 465.0, "ProteinContent": 5.6},   # top_n of these
            ...
          ]
        }

    "method" tells the caller which path was taken:
      - "taste_similarity": score = cosine similarity of taste embeddings
        (higher = more similar; ties within 1e-4 are broken by ascending
        z-scored Euclidean nutrition distance, so the order is deterministic).
      - "nutrition_fallback": score = 1 / (1 + z-scored Euclidean nutrition
        distance) (higher = more similar). Used when the blocked recipe has no
        embedding, has fewer than min_vocab_tokens in-vocab tokens, or the
        taste path finds zero eligible candidates.

    The blocked recipe's own RecipeId is always excluded from its alternatives.
    If safe_recipes_df yields no candidates at all (empty frame, or all rows
    excluded), "alternatives" is [] with a clear "reason" field — never an
    exception.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]          # Expert System/
sys.path.insert(0, str(BASE_DIR))

from ml.health_classifier.health_classifier import NUTRITION_COLS     # noqa: E402
from ml.word2vec.build_taste_embeddings import clean_ingredient_list  # noqa: E402
from ml.word2vec.taste_inference import _load as _load_taste          # noqa: E402

CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"

TIE_EPSILON = 1e-4
FALLBACK_SCORE_K = 1.0      # score = 1 / (1 + K-adjacent z-distance) -> (0, 1]

# ---------------------------------------------------------------------------
# module-level lazy caches (mirrored after taste_inference.py's singleton)
#   _aligned_ids:     np.ndarray (N,) int64 — every RecipeId WITH an embedding
#   _aligned_matrix:  np.ndarray (N, 150) float32 — L2-normalized vectors for
#                     _aligned_ids, in the same row order
#   _aligned_index:   dict RecipeId(int) -> row index into the two arrays
#   _nutr_stats:      (mean, std) np.ndarray (9,) float64 over the full corpus
# ---------------------------------------------------------------------------
_aligned_ids = None
_aligned_matrix = None
_aligned_index = None
_nutr_stats = None


def _ensure_aligned():
    """Build the process-wide aligned embedding matrix once (lazy singleton)."""
    global _aligned_ids, _aligned_matrix, _aligned_index
    if _aligned_index is not None:
        return _aligned_ids, _aligned_matrix, _aligned_index

    _, embeddings = _load_taste()          # production singleton loader
    all_ids = np.fromiter(embeddings.keys(), dtype=np.int64, count=len(embeddings))
    matrix = np.stack(list(embeddings.values()))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = (matrix / norms).astype(np.float32)
    index = {int(rid): i for i, rid in enumerate(all_ids)}

    _aligned_ids, _aligned_matrix, _aligned_index = all_ids, matrix, index
    return _aligned_ids, _aligned_matrix, _aligned_index


def _ensure_nutr_stats():
    """Corpus mean/std (per NUTRITION_COL) over cleaned_recipes.csv, cached.

    Loaded the same way build_taste_embeddings.py reads the corpus
    (pd.read_csv low_memory=False) but only the 9 needed columns, and only on
    first actual use of the nutrition fallback.
    """
    global _nutr_stats
    if _nutr_stats is not None:
        return _nutr_stats

    df = pd.read_csv(CLEANED_CSV_PATH, usecols=NUTRITION_COLS, low_memory=False)
    mean = df.mean().to_numpy(dtype=np.float64)
    std = df.std().to_numpy(dtype=np.float64)
    std = np.where(std > 0, std, 1.0)       # degenerate constant column -> no-op z
    _nutr_stats = (mean, std)
    return _nutr_stats


def _in_vocab_token_count(recipe_row) -> int:
    """Tokenize the recipe's IngredientsList with the REAL build-time cleaning
    logic and count how many tokens are in the Word2Vec vocabulary. This is the
    trust metric deciding between taste similarity and the nutrition fallback:
    the PoC showed recipes with very few in-vocab tokens produce flat,
    nutritionally nonsensical taste rankings."""
    raw = recipe_row["IngredientsList"]
    tokens = clean_ingredient_list(raw)     # ast.literal_eval + punctuation rules
    if not tokens:
        return 0
    wv = _load_taste()[0]                   # cached Word2Vec KeyedVectors
    vocab = wv.key_to_index
    return sum(1 for t in tokens if t in vocab)


def _extract_recipe_row(recipe) -> pd.Series:
    """Normalize dict / Series / one-row DataFrame to a validated Series with
    RecipeId, Name, IngredientsList and all 9 nutrition columns present and
    finite (no silent imputation, same discipline as explain.py)."""
    if isinstance(recipe, pd.DataFrame):
        if len(recipe) != 1:
            raise ValueError(
                f"blocked_recipe must be a single recipe; got a DataFrame with "
                f"{len(recipe)} rows")
        row = recipe.iloc[0]
    elif isinstance(recipe, pd.Series):
        row = recipe
    elif isinstance(recipe, dict):
        row = pd.Series(recipe)
    else:
        raise TypeError(
            f"blocked_recipe must be a dict, pandas Series, or one-row "
            f"DataFrame; got {type(recipe)}")

    required = ["RecipeId", "Name", "IngredientsList"] + list(NUTRITION_COLS)
    missing = [c for c in required if c not in row.index]
    if missing:
        raise ValueError(
            f"blocked_recipe is missing required columns: {missing}; expected "
            f"{required}")

    out = row.copy()
    for col in NUTRITION_COLS:
        val = pd.to_numeric(out[col], errors="coerce")
        if pd.isna(val):
            raise ValueError(
                f"blocked_recipe has non-numeric/NaN value for column "
                f"{col!r} (refusing to impute — alternatives must reflect the "
                f"actual recipe)")
        out[col] = float(val)
    return out


def _validate_safe_df(safe_recipes_df) -> None:
    if not isinstance(safe_recipes_df, pd.DataFrame):
        raise TypeError(
            f"safe_recipes_df must be a pandas DataFrame; got "
            f"{type(safe_recipes_df)}")
    required = ["RecipeId", "Name"] + list(NUTRITION_COLS)
    missing = [c for c in required if c not in safe_recipes_df.columns]
    if missing:
        raise ValueError(
            f"safe_recipes_df is missing required columns: {missing}; expected "
            f"{required}")


def _zscore_table(nutr_df: pd.DataFrame):
    """Return (N, 9) float64 z-scored table. Missing/NaN nutrition cells are
    treated as the corpus average (z = 0), so such candidates simply do not
    contribute in those dimensions instead of poisoning the distance."""
    mean, std = _ensure_nutr_stats()
    vals = nutr_df[NUTRITION_COLS].to_numpy(dtype=np.float64)
    vals = np.where(np.isnan(vals), mean, vals)
    return (vals - mean) / std


def _nutrition_distance(blocked_z, cand_z):
    """Euclidean distance between one z-scored blocked row and each candidate
    row, equal weighting across NUTRITION_COLS (corpus-z-scored)."""
    diff = cand_z - blocked_z
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _neutr_row_from_blocked(row: pd.Series) -> np.ndarray:
    mean, std = _ensure_nutr_stats()
    vals = np.asarray(
        [pd.to_numeric(row[c], errors="coerce") for c in NUTRITION_COLS],
        dtype=np.float64)
    return (vals - mean) / std


def _json_native(obj):
    """Recursively convert numpy scalars/arrays to native Python types so the
    result is always json.dumps()-serializable without a custom encoder."""
    if isinstance(obj, dict):
        return {k: _json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _tie_grouped_order(scores, distances):
    """Deterministic ranking: descending score; candidates whose scores are
    within TIE_EPSILON are sub-ordered by ascending nutrition distance, so the
    output order never depends on dict/array iteration order. Vectorized — no
    per-element Python loop."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    d = distances[order]
    new_group = np.concatenate(([True], s[1:] < s[:-1] - TIE_EPSILON))
    gid = np.cumsum(new_group) - 1
    idx = np.lexsort((d, gid))          # primary: group id, secondary: distance
    return order[idx]


def suggest_alternatives(blocked_recipe, safe_recipes_df: pd.DataFrame,
                         top_n: int = 3, min_vocab_tokens: int = 4) -> dict:
    """Top-N taste-similar (or nutrition-fallback) safe alternatives for a
    blocked recipe. Full contract and output shape in the module docstring.
    Returns a plain dict of native Python types."""
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive int, got {top_n!r}")
    if not isinstance(min_vocab_tokens, int) or min_vocab_tokens < 1:
        raise ValueError(
            f"min_vocab_tokens must be a positive int, got {min_vocab_tokens!r}")

    row = _extract_recipe_row(blocked_recipe)
    blocked_id = int(row["RecipeId"])
    _validate_safe_df(safe_recipes_df)

    safe_ids = pd.to_numeric(
        safe_recipes_df["RecipeId"], errors="coerce").astype("Int64")
    cand_df = safe_recipes_df.copy()
    cand_df["_rid"] = safe_ids
    cand_df = cand_df[cand_df["_rid"].notna()]          # drop unparseable ids

    if (cand_df["_rid"] == blocked_id).any():           # Concern 3: never self-recommend
        cand_df = cand_df[cand_df["_rid"] != blocked_id]

    reason = ""

    def _fallback(empty_reason):
        nonlocal reason
        reason = empty_reason
        if len(cand_df) == 0:
            return _finish("nutrition_fallback", [], reason)
        return _finish("nutrition_fallback", _nutrition_top(), reason)

    def _nutrition_top():
        blocked_z = _neutr_row_from_blocked(row)
        cand_z = _zscore_table(cand_df)
        dists = _nutrition_distance(blocked_z, cand_z)
        order = np.argsort(dists, kind="stable")
        rows = []
        for i in order[:top_n]:
            r = cand_df.iloc[int(i)]
            rows.append({
                "RecipeId": int(r["_rid"]),
                "Name": str(r["Name"]),
                "score": FALLBACK_SCORE_K / (FALLBACK_SCORE_K + float(dists[int(i)])),
                "Calories": float(pd.to_numeric(r["Calories"], errors="coerce")),
                "ProteinContent": float(
                    pd.to_numeric(r["ProteinContent"], errors="coerce")),
            })
        return rows

    def _finish(method, rows, note=""):
        nonlocal reason
        return {
            "method": method,
            "blocked_recipe": {"RecipeId": blocked_id, "Name": str(row["Name"])},
            "requested_top_n": top_n,
            "returned_count": len(rows),
            "reason": note,
            "alternatives": rows,
        }

    if len(cand_df) == 0:
        return _finish("nutrition_fallback", [],
                       "safe_recipes_df is empty (or contained only the blocked "
                       "recipe itself)")

    # ---- method decision ---------------------------------------------------
    _, _, aligned_index = _ensure_aligned()
    _, embeddings = _load_taste()
    blocked_vec = embeddings.get(blocked_id)
    n_tokens = _in_vocab_token_count(row)
    use_taste = blocked_vec is not None and n_tokens >= min_vocab_tokens

    if use_taste:
        cand_rids = cand_df["_rid"].to_numpy(dtype=np.int64)
        idx_mask = np.isin(cand_rids, list(aligned_index.keys()))
        rows_mask = np.where(idx_mask)[0]
        if len(rows_mask) == 0:
            return _fallback(
                "taste path had zero candidates with embeddings; fell back to "
                "nutrition similarity")
        rows_np = cand_rids[rows_mask]

        _, matrix, _ = _ensure_aligned()
        row_idxs = np.asarray([aligned_index[int(r)] for r in rows_np])
        cand_rows = matrix[row_idxs]
        scores = cand_rows @ (blocked_vec / np.linalg.norm(blocked_vec))

        blocked_z = _neutr_row_from_blocked(row)
        cand_z = _zscore_table(cand_df.iloc[rows_mask])
        dists = _nutrition_distance(blocked_z, cand_z)

        order = _tie_grouped_order(scores, dists)[:top_n]   # Concern 2: ties
        rows = []
        for k in order:
            pos = int(rows_mask[k])
            r = cand_df.iloc[pos]
            rows.append({
                "RecipeId": int(r["_rid"]),
                "Name": str(r["Name"]),
                "score": float(scores[k]),
                "Calories": float(pd.to_numeric(r["Calories"], errors="coerce")),
                "ProteinContent": float(
                    pd.to_numeric(r["ProteinContent"], errors="coerce")),
            })
        note = ""
        if len(rows) < top_n:
            note = (f"only {len(rows)} candidates available after exclusions "
                    f"(requested {top_n})")
        return _json_native(_finish("taste_similarity", rows, note))

    # ---- nutrition fallback ------------------------------------------------
    return _fallback(
        f"blocked recipe has no taste embedding or only {n_tokens} "
        f"in-vocab token(s) (min_vocab_tokens={min_vocab_tokens}); "
        "used nutrition-based similarity instead")