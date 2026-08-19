import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]       
sys.path.insert(0, str(BASE_DIR))

from ml.health_classifier.health_classifier import NUTRITION_COLS     
from ml.word2vec.build_taste_embeddings import clean_ingredient_list  
from ml.word2vec.taste_inference import _load as _load_taste          

CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"

TIE_EPSILON = 1e-4
FALLBACK_SCORE_K = 1.0      

_aligned_ids = None
_aligned_matrix = None
_aligned_index = None
_nutr_stats = None


def _ensure_aligned():
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
    global _nutr_stats
    if _nutr_stats is not None:
        return _nutr_stats

    df = pd.read_csv(CLEANED_CSV_PATH, usecols=NUTRITION_COLS, low_memory=False)
    mean = df.mean().to_numpy(dtype=np.float64)
    std = df.std().to_numpy(dtype=np.float64)
    std = np.where(std > 0, std, 1.0)      
    _nutr_stats = (mean, std)
    return _nutr_stats


def _in_vocab_token_count(recipe_row) -> int:
    raw = recipe_row["IngredientsList"]
    tokens = clean_ingredient_list(raw)   
    if not tokens:
        return 0
    wv = _load_taste()[0]                 
    vocab = wv.key_to_index
    return sum(1 for t in tokens if t in vocab)


def _extract_recipe_row(recipe) -> pd.Series:
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
    mean, std = _ensure_nutr_stats()
    vals = nutr_df[NUTRITION_COLS].to_numpy(dtype=np.float64)
    vals = np.where(np.isnan(vals), mean, vals)
    return (vals - mean) / std


def _nutrition_distance(blocked_z, cand_z):
    diff = cand_z - blocked_z
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _neutr_row_from_blocked(row: pd.Series) -> np.ndarray:
    mean, std = _ensure_nutr_stats()
    vals = np.asarray(
        [pd.to_numeric(row[c], errors="coerce") for c in NUTRITION_COLS],
        dtype=np.float64)
    return (vals - mean) / std


def _json_native(obj):
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
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    d = distances[order]
    new_group = np.concatenate(([True], s[1:] < s[:-1] - TIE_EPSILON))
    gid = np.cumsum(new_group) - 1
    idx = np.lexsort((d, gid))        
    return order[idx]


def suggest_alternatives(blocked_recipe, safe_recipes_df: pd.DataFrame,
                         top_n: int = 3, min_vocab_tokens: int = 4) -> dict:
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
    cand_df = cand_df[cand_df["_rid"].notna()]       

    if (cand_df["_rid"] == blocked_id).any():          
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

        order = _tie_grouped_order(scores, dists)[:top_n]   
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

    return _fallback(
        f"blocked recipe has no taste embedding or only {n_tokens} "
        f"in-vocab token(s) (min_vocab_tokens={min_vocab_tokens}); "
        "used nutrition-based similarity instead")