"""
explain.py — SHAP-based explanation layer for health_classifier (standalone, opt-in)
=====================================================================================

Explains, per medical condition, WHY the trained multi-label model gave a recipe its
suitability score. Pure additive module: nothing in the scoring/ranking pipeline imports
this file; a backend (e.g. Django) calls it explicitly as a separate endpoint.

Method (validated in prior investigations):
    shap.Explainer over a single-condition output wrapper, which auto-selects the
    ExactExplainer for this small feedforward net — deterministic, near-additive, and
    ~20 ms per condition per recipe after a one-time warmup. DeepExplainer was tested
    and rejected (failed its own additivity check on this model). The background
    baseline is 150 rows sampled from cleaned_recipes.csv (random_state=42), scaled
    with the same saved health_scaler.pkl used by production inference.

PERFORMANCE NOTE FOR BACKEND:
    The FIRST explanation call in a process pays a one-time ~8 s cost (numba JIT
    compilation of SHAP's masker machinery). Call warm_up() once at server startup
    (or issue one throwaway explanation) so real user requests stay in the tens-of-ms
    range.

==============  API CONTRACT (backend consumption, no ML knowledge needed)  ==============

explain_health_score(recipe, condition_keys, top_n=3) -> dict

    recipe:          pandas Series / one-row DataFrame, or plain dict, supplying the
                     9 nutrition columns (Calories, ProteinContent, CarbohydrateContent,
                     SugarContent, SodiumContent, FatContent, SaturatedFatContent,
                     CholesterolContent, FiberContent). Missing column or NaN value
                     -> ValueError (nothing is silently imputed).
    condition_keys:  list of bare condition keys as used elsewhere in the project
                     (see core.constants.DISEASE_EN / rules.medical_rules), e.g.
                     ["diabetes", "hypertension"]. The "label_" prefix is added
                     internally exactly like inference.ai_health_score does. Keys that
                     are not among the 22 trained conditions are NOT silently mapped to
                     a fallback mean: they are reported in "unmatched_keys" and
                     skipped (explaining an all-22 average would be misleading).
    top_n:           how many top contributing features per condition (by |SHAP|).

    Return (all values JSON-serializable, numpy types converted to native)::

        {
          "condition_explanations": {
            "diabetes": {
              "probability": 0.0,         # model sigmoid for this condition, 0..1
              "top_reasons": [
                {"feature": "FiberContent", "value": 0.6, "shap_value": -196.6,
                 "direction": "hurt",      # "helped" | "hurt" (sign of SHAP on the
                                           # condition logit: helped = pushed the
                                           # model's fit score up)
                 "text": "Very low fiber content (0.6g) hurt this recipe's fit"}
              ]
            }
          },
          "unmatched_keys": []
        }

    SHAP values are in raw-logit space and can be large (the model's decision margins
    are extreme); the "text" field is the human-consumable part, "shap_value" is for
    debugging/ranking only. "value" is the recipe's actual (unscaled) feature value.
"""

import json  # used by the __main__ dev smoke below
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parents[2]          # Expert System/
sys.path.insert(0, str(BASE_DIR))

from ml.health_classifier.inference import _load           # production singleton loader
from ml.health_classifier.health_classifier import NUTRITION_COLS

import shap

# ---------------------------------------------------------------------------
# global (module-level) caches — mirrored after inference.py's singleton pattern
# ---------------------------------------------------------------------------
BACKGROUND_N = 150
BACKGROUND_SEED = 42
CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned_recipes.csv"

_background = None          # np.ndarray (150, 9), scaled by the saved scaler
_explainers = {}            # condition index -> cached shap.Explainer (ExactExplainer)


class _SingleOutput(nn.Module):
    """Exposes only one condition logit so the explainer back-propagates one target."""

    def __init__(self, model, idx: int):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, self.idx:self.idx + 1]


class _ConditionAdapter:
    """numpy-in / numpy-out adapter for ONE condition index: SHAP's masked sampling
    passes numpy rows, the torch model needs tensors, results back as numpy."""

    def __init__(self, model, idx: int):
        self._wrap = _SingleOutput(model, idx)

    def __call__(self, x):
        with torch.no_grad():
            out = self._wrap(torch.tensor(np.asarray(x, dtype=np.float32)))
        return out.numpy()


# ---------------------------------------------------------------------------
# human-readable text helpers
# ---------------------------------------------------------------------------
# Feature name -> (natural-language phrase, unit). Framing ("high/low") is derived
# per recipe from the SHAP sign + the value's percentile vs the background, never
# assumed a-priori (a feature can help or hurt depending on the recipe's context).
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

# PROTEINCONTENT HONEST-WORDING SAFEGUARD.
# Prior investigation (feature/ai-health-classifier): the model's diabetes head
# learned a strong negative main effect for ProteinContent which the training LABEL
# does not contain (marginal label correlation is even slightly positive). It is a
# learned shortcut driven by the protein <-> calorie-density collinearity of this
# corpus (high-protein recipes violate the Calories<=600 gate 5.5x more often).
# Therefore ProteinContent's explanation text must NEVER assert protein is medically
# harmful or penalized — only that the recipe's overall calorie/protein profile moved
# the model's score. The feature still appears in top_reasons with its real SHAP value.
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

# CALORIES HONEST-WORDING SAFEGUARD.
# Combined investigation (feature/ai-health-classifier): within the population the
# model actually scores (recipes already compliant with the hard rules), lower
# calories genuinely associate with a HIGHER diabetes-fit score (std-coef -0.917,
# partial corr -0.377) — a real, correct pattern, unlike the protein shortcut. But
# on OFF-distribution recipes (e.g. one that exceeds a condition's calorie limit —
# the hard rules remove those before scoring, yet explain_health_score() can still
# be called on any recipe) the raw SHAP contribution can INVERT (a 627.7-kcal
# recipe showed a positive +34.8 calorie contribution) because the model's calorie
# response is non-monotone across the full value range. Therefore calorie wording
# must NOT claim a simple universal rule ("high calories always penalized / always
# helpful"); it is stated as the model's learned pattern instead.
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

# SUGARCONTENT: intentionally NOT given special wording. The investigations found
# sugar's SHAP direction consistently correct with no off-distribution inversion:
# the two-recipe test showed a negative contribution for 47.9g sugar, and the
# within-safe-population partial correlation of sugar with the diabetes logit is
# -0.265 (higher sugar -> lower fit, as expected). The generic phrasing stays
# accurate, so a safeguard would only add noise. (feature/ai-health-classifier.)


def _percentile_adjective(value: float, background_col: np.ndarray) -> str:
    """Very low / Low / Moderate / High / Very high vs the 150-row background."""
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
    """Human line for one feature. sign = SHAP sign (helped/hurt on the logit).

    value = raw (unscaled) feature value for display; scaled_value = the model-input
    value, used for the background-percentile adjective (both must live in the same
    space as the background column, which is scaled)."""
    helped = shap_value > 0
    verb = "helped" if helped else "hurt"

    if feature == PROTEIN_FEATURE:
        # careful wording — see safeguard comment above; never "protein is bad"
        caveat = _PROTEIN_HELPED_TEXT if helped else _PROTEIN_HURT_TEXT
        return f"This recipe's {caveat}"

    if feature == CALORIES_FEATURE:
        # careful wording — see safeguard comment above; no universal
        # "high calories = bad/good" claim, sign can invert off-distribution
        caveat = _CALORIES_HELPED_TEXT if helped else _CALORIES_HURT_TEXT
        return f"This recipe's {caveat}"

    phrase, unit = _FEATURE_PHRASES[feature]
    adj = _percentile_adjective(scaled_value, background_col)
    return f"{adj} {phrase} ({value:.1f}{unit}) {verb} this recipe's fit"


# ---------------------------------------------------------------------------
# lazy loading: model/scaler/labels from production, background, per-condition explainers
# ---------------------------------------------------------------------------
def _ensure_background() -> np.ndarray:
    """150-row baseline (random_state=42) from cleaned_recipes.csv, scaled by the
    SAME saved health_scaler.pkl — identical setup to the validated investigations."""
    global _background
    if _background is None:
        _, scaler, _ = _load()
        df = pd.read_csv(CLEANED_CSV_PATH, low_memory=False)
        sample = df.sample(BACKGROUND_N, random_state=BACKGROUND_SEED)[NUTRITION_COLS]
        _background = scaler.transform(sample.astype(np.float32)).astype(np.float32)
    return _background


def _get_explainer(condition_idx: int):
    """Per-condition ExactExplainer, built once and cached (numba JIT happens once)."""
    if condition_idx not in _explainers:
        model, _, _ = _load()
        background = _ensure_background()
        masker = shap.maskers.Independent(background)
        _explainers[condition_idx] = shap.Explainer(
            _ConditionAdapter(model, condition_idx), masker,
            feature_names=list(NUTRITION_COLS))
    return _explainers[condition_idx]


def warm_up() -> None:
    """One-time startup cost (approx. 8 s): builds the background and all 22
    per-condition explainers and triggers the numba JIT compile. Recommended at
    server startup so the first real request is not the slow one."""
    _load()
    background = _ensure_background()
    mean_row = background.mean(axis=0, keepdims=True).astype(np.float32)
    for idx in range(len(_labels_of())):
        _get_explainer(idx)          # construct (cheap)…
        np.asarray(_get_explainer(idx)(mean_row).values)   # …and JIT-compile


def _labels_of():
    return _load()[2]


def _extract_recipe_row(recipe) -> pd.Series:
    """Normalize dict / Series / one-row DataFrame to a numeric Series, validating
    that all 9 nutrition columns are present and finite (no silent imputation)."""
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
    """Per-condition SHAP explanation of the model's suitability score for one recipe.

    Full contract, output shape and JSON-serializability pledge in the module
    docstring. Returns a plain dict of native Python types.
    """
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive int, got {top_n!r}")

    row = _extract_recipe_row(recipe)
    values = row[NUTRITION_COLS].astype(np.float32).to_numpy()

    model, scaler, labels = _load()
    background = _ensure_background()

    scaled_x = scaler.transform(values.reshape(1, -1)).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.tensor(scaled_x, dtype=torch.float32)).numpy()[0]

    # map bare keys the same way production inference does ("label_" + key)
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
    # dev-only smoke: explain RecipeId 49 for diabetes on demand
    import pandas as _pd
    _df = _pd.read_csv(CLEANED_CSV_PATH, low_memory=False)
    _recipe = _df.loc[_df["RecipeId"] == 49].iloc[0][NUTRITION_COLS]
    print(json.dumps(explain_health_score(_recipe, ["diabetes"]), indent=2, ensure_ascii=False))