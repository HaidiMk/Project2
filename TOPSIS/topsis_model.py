import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

EXPERT_SYSTEM_DIR = Path(__file__).resolve().parents[1] / "Expert System"
if str(EXPERT_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERT_SYSTEM_DIR))


CRITERIA: List[str] = [
    "Calories", "ProteinContent", "FatContent",
    "CarbohydrateContent", "FiberContent",
    "SugarContent", "SodiumContent",
]

GOAL_CRITERIA: Dict[str, Dict[str, Tuple[float, str]]] = {
    "weight_loss": {
        "Calories":            (0.25, "cost"),
        "ProteinContent":      (0.25, "benefit"),
        "FatContent":          (0.10, "cost"),
        "CarbohydrateContent": (0.10, "cost"),
        "FiberContent":        (0.15, "benefit"),
        "SugarContent":        (0.10, "cost"),
        "SodiumContent":       (0.05, "cost"),
    },
    "weight_gain": {
        "Calories":            (0.30, "benefit"),
        "ProteinContent":      (0.25, "benefit"),
        "FatContent":          (0.15, "benefit"),
        "CarbohydrateContent": (0.15, "benefit"),
        "FiberContent":        (0.05, "cost"),
        "SugarContent":        (0.05, "benefit"),
        "SodiumContent":       (0.05, "cost"),
    },
    "muscle_gain": {
        "Calories":            (0.15, "benefit"),
        "ProteinContent":      (0.40, "benefit"),
        "FatContent":          (0.10, "cost"),
        "CarbohydrateContent": (0.15, "benefit"),
        "FiberContent":        (0.05, "benefit"),
        "SugarContent":        (0.10, "cost"),
        "SodiumContent":       (0.05, "cost"),
    },
    "maintenance": {
        "Calories":            (0.15, "cost"),
        "ProteinContent":      (0.20, "benefit"),
        "FatContent":          (0.10, "cost"),
        "CarbohydrateContent": (0.15, "benefit"),
        "FiberContent":        (0.20, "benefit"),
        "SugarContent":        (0.10, "cost"),
        "SodiumContent":       (0.10, "cost"),
    },
    "heart_health": {
        "Calories":            (0.10, "cost"),
        "ProteinContent":      (0.15, "benefit"),
        "FatContent":          (0.20, "cost"),
        "CarbohydrateContent": (0.05, "cost"),
        "FiberContent":        (0.20, "benefit"),
        "SugarContent":        (0.10, "cost"),
        "SodiumContent":       (0.20, "cost"),
    },
}

DEFAULT_CRITERIA: Dict[str, Tuple[float, str]] = {
    "Calories":            (0.15, "cost"),
    "ProteinContent":      (0.20, "benefit"),
    "FatContent":          (0.15, "cost"),
    "CarbohydrateContent": (0.10, "cost"),
    "FiberContent":        (0.15, "benefit"),
    "SugarContent":        (0.15, "cost"),
    "SodiumContent":       (0.10, "cost"),
}


def topsis_score(df: pd.DataFrame, goal: Optional[str] = None) -> np.ndarray:

    available = [c for c in CRITERIA if c in df.columns]
    if not available or len(df) == 0:
        return np.zeros(len(df))

    matrix = df[available].fillna(0).to_numpy(dtype=float)

    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom[denom == 0] = 1.0  
    normalized = matrix / denom

    criteria_map = GOAL_CRITERIA.get(goal, DEFAULT_CRITERIA)
    weights = np.array([criteria_map.get(c, (0.0, "cost"))[0] for c in available])
    weighted = normalized * weights

    ideal_best = np.zeros(len(available))
    ideal_worst = np.zeros(len(available))
    for i, col in enumerate(available):
        col_values = weighted[:, i]
        col_type = criteria_map.get(col, (0.0, "cost"))[1]
        if col_type == "benefit":
            ideal_best[i] = col_values.max()
            ideal_worst[i] = col_values.min()
        else:  # cost
            ideal_best[i] = col_values.min()
            ideal_worst[i] = col_values.max()

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    total_dist = dist_best + dist_worst
    total_dist[total_dist == 0] = 1.0  
    closeness = dist_worst / total_dist

    return closeness


BLEND_WEIGHTS: Dict[str, float] = {
    "_topsis_score":     0.4,
    "_ai_health_score":  0.4,
    "_expert_score":     0.2,
}
TASTE_BLEND_WEIGHTS: Dict[str, float] = {
    "_topsis_score":     0.35,
    "_ai_health_score":  0.35,
    "_expert_score":     0.15,
    "_taste_score":      0.15,
}


def _normalize_minmax(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    flat = hi <= lo + 1e-12
    return np.full_like(v, 0.5) if flat else (v - lo) / (hi - lo)


def rank_with_topsis(
    df: pd.DataFrame,
    goal: Optional[str] = None,
    user_taste_text: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    df["_topsis_score"] = topsis_score(df, goal=goal)

    if len(df) == 0:
        return df

    blend = BLEND_WEIGHTS
    if user_taste_text:
        from ml.word2vec.taste_inference import taste_score
        df["_taste_score"] = taste_score(df, user_taste_text)
        blend = TASTE_BLEND_WEIGHTS

    used = [c for c in blend if c in df.columns]
    weights = np.array([blend[c] for c in used], dtype=float)
    weights = weights / weights.sum()

    parts = []
    for col, w in zip(used, weights):
        vals = df[col].astype(float).to_numpy()
        if col == "_expert_score":
            vals = _normalize_minmax(vals)
        parts.append(w * vals)

    df["final_score"] = sum(parts)
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)