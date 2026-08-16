import json  
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parents[2]        
sys.path.insert(0, str(BASE_DIR))

from ml.health_classifier.inference import _load           
from ml.health_classifier.health_classifier import NUTRITION_COLS

import shap

BACKGROUND_N = 150
BACKGROUND_SEED = 42
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"

_background = None        
_explainers = {}    


class _SingleOutput(nn.Module):

    def __init__(self, model, idx: int):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, self.idx:self.idx + 1]


class _ConditionAdapter:

    def __init__(self, model, idx: int):
        self._wrap = _SingleOutput(model, idx)

    def __call__(self, x):
        with torch.no_grad():
            out = self._wrap(torch.tensor(np.asarray(x, dtype=np.float32)))
        return out.numpy()


_FEATURE_PHRASES = {
    "Calories":             ("calories",              "kcal"),
    "ProteinContent":       ("protein content",       "g"),
    "CarbohydrateContent":  ("carbohydrate content",  "g"),
    "SugarContent":         ("sugar content",         "g"),
    "SodiumContent":        ("sodium content",        "mg"),
    "FatContent":           ("fat content",           "g"),
    "SaturatedFatContent":  ("saturated fat content", "g"),
    "CholesterolContent":   ("cholesterol content",   "mg"),
    "FiberContent":         ("fiber content",         "g"),
}

PROTEIN_FEATURE = "ProteinContent"
_PROTEIN_HURT_TEXT = (
    "overall calorie/protein profile reduced this recipe's fit in the model's view "
    "(high protein correlates with calorie-dense recipes in the training data — "
    "protein itself is not medically penalized)"
)
_PROTEIN_HELPED_TEXT = (
    "overall calorie/protein profile improved this recipe's fit in the model's view "
    "(reflects the model's learned protein/calorie-density correlation, not a "
    "medical rule favoring protein)"
)

CALORIES_FEATURE = "Calories"
_CALORIES_HURT_TEXT = (
    "calorie level reduced this recipe's fit in the model's view — the model "
    "associates lower calories with a better fit within compliant recipes"
)
_CALORIES_HELPED_TEXT = (
    "calorie level raised this recipe's fit in the model's view — a pattern the "
    "model learned from its training data rather than a medical claim; the "
    "contribution sign can invert for recipes beyond a condition's calorie limit"
)



def _percentile_adjective(value: float, background_col: np.ndarray) -> str:
    pct = float((background_col < value).mean())
    if pct >= 0.90:
        return "Very high"
    if pct >= 0.70:
        return "High"
    if pct > 0.30:
        return "Moderate"
    if pct > 0.10:
        return "Low"
    return "Very low"


def _reason_text(feature: str, value: float, scaled_value: float,
                 shap_value: float, background_col: np.ndarray) -> str:
    helped = shap_value > 0
    verb = "helped" if helped else "hurt"

    if feature == PROTEIN_FEATURE:
        caveat = _PROTEIN_HELPED_TEXT if helped else _PROTEIN_HURT_TEXT
        return f"This recipe's {caveat}"

    if feature == CALORIES_FEATURE:
        caveat = _CALORIES_HELPED_TEXT if helped else _CALORIES_HURT_TEXT
        return f"This recipe's {caveat}"

    phrase, unit = _FEATURE_PHRASES[feature]
    adj = _percentile_adjective(scaled_value, background_col)
    return f"{adj} {phrase} ({value:.1f}{unit}) {verb} this recipe's fit"


def _ensure_background() -> np.ndarray:
    global _background
    if _background is None:
        _, scaler, _ = _load()
        df = pd.read_csv(CLEANED_CSV_PATH, low_memory=False)
        sample = df.sample(BACKGROUND_N, random_state=BACKGROUND_SEED)[NUTRITION_COLS]
        _background = scaler.transform(sample.astype(np.float32)).astype(np.float32)
    return _background


def _get_explainer(condition_idx: int):
    if condition_idx not in _explainers:
        model, _, _ = _load()
        background = _ensure_background()
        masker = shap.maskers.Independent(background)
        _explainers[condition_idx] = shap.Explainer(
            _ConditionAdapter(model, condition_idx), masker,
            feature_names=list(NUTRITION_COLS))
    return _explainers[condition_idx]


def warm_up() -> None:
    _load()
    background = _ensure_background()
    mean_row = background.mean(axis=0, keepdims=True).astype(np.float32)
    for idx in range(len(_labels_of())):
        _get_explainer(idx)         
        np.asarray(_get_explainer(idx)(mean_row).values)   


def _labels_of():
    return _load()[2]


def _extract_recipe_row(recipe) -> pd.Series:
    if isinstance(recipe, pd.DataFrame):
        if len(recipe) != 1:
            raise ValueError(
                f"recipe must be a single recipe; got a DataFrame with {len(recipe)} rows")
        row = recipe.iloc[0]
    elif isinstance(recipe, pd.Series):
        row = recipe
    elif isinstance(recipe, dict):
        row = pd.Series(recipe)
    else:
        raise TypeError(
            f"recipe must be a dict, pandas Series, or one-row DataFrame; got {type(recipe)}")

    missing = [c for c in NUTRITION_COLS if c not in row.index]
    if missing:
        raise ValueError(
            f"recipe is missing required nutrition columns: {missing}; "
            f"expected exactly {list(NUTRITION_COLS)}")

    out = pd.to_numeric(row[NUTRITION_COLS], errors="coerce").astype(float)
    if out.isna().any():
        bad = [c for c, v in out.items() if pd.isna(v)]
        raise ValueError(f"recipe has non-numeric/NaN values for columns: {bad} "
                         f"(refusing to impute — explanations must reflect the "
                         f"actual recipe)")
    return out


def explain_health_score(recipe, condition_keys, top_n: int = 3) -> dict:
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive int, got {top_n!r}")

    row = _extract_recipe_row(recipe)
    values = row[NUTRITION_COLS].astype(np.float32).to_numpy()

    model, scaler, labels = _load()
    background = _ensure_background()

    scaled_x = scaler.transform(values.reshape(1, -1)).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(scaled_x, dtype=torch.float32)).numpy()[0]

    unmatched = []
    matched_idx = []
    for key in condition_keys:
        label = "label_" + str(key)
        if label in labels:
            matched_idx.append(labels.index(label))
        else:
            unmatched.append(str(key))

    explanations = {}
    for idx in matched_idx:
        condition = labels[idx].removeprefix("label_")
        with torch.no_grad():
            prob = float(torch.sigmoid(torch.tensor(logits[idx], dtype=torch.float64)))

        sv = np.asarray(_get_explainer(idx)(scaled_x).values)[0]
        order = sorted(range(len(NUTRITION_COLS)),
                       key=lambda i: -abs(float(sv[i])))[:min(top_n, len(NUTRITION_COLS))]

        top_reasons = []
        for i in order:
            feature = NUTRITION_COLS[i]
            value = float(values[i])
            sval = float(sv[i])
            top_reasons.append({
                "feature": feature,
                "value": value,
                "shap_value": round(sval, 4),
                "direction": "helped" if sval > 0 else "hurt",
                "text": _reason_text(feature, value, float(scaled_x[0, i]), sval,
                                     background[:, i]),
            })

        explanations[condition] = {
            "probability": round(prob, 6),
            "top_reasons": top_reasons,
        }

    return {
        "condition_explanations": explanations,
        "unmatched_keys": unmatched,
    }


if __name__ == "__main__":
    import pandas as _pd
    _df = _pd.read_csv(CLEANED_CSV_PATH, low_memory=False)
    _recipe = _df.loc[_df["RecipeId"] == 49].iloc[0][NUTRITION_COLS]
    print(json.dumps(explain_health_score(_recipe, ["diabetes"]), indent=2, ensure_ascii=False))