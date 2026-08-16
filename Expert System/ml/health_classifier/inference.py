import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[2]          
sys.path.insert(0, str(BASE_DIR))

from ml.health_classifier.health_classifier import HealthClassifier, NUTRITION_COLS

MODEL_PATH = BASE_DIR / "data" / "health_classifier.pt"
SCALER_PATH = BASE_DIR / "data" / "health_scaler.pkl"
LABELS_PATH = BASE_DIR / "data" / "health_classifier_labels.json"

_model = None
_scaler = None
_labels = None


def _load():
    global _model, _scaler, _labels
    if _model is not None:
        return _model, _scaler, _labels

    with open(LABELS_PATH, encoding="utf-8") as f:
        _labels = json.load(f)

    _scaler = joblib.load(SCALER_PATH)

    model = HealthClassifier(num_conditions=len(_labels))
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    _model = model
    return _model, _scaler, _labels


def ai_health_score(df: pd.DataFrame, condition_keys=None) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    model, scaler, labels = _load()

    X = df[NUTRITION_COLS].astype(np.float32)
    X = X.fillna(X.median())

    scaled = scaler.transform(X)

    with torch.no_grad():
        logits = model(torch.tensor(scaled, dtype=torch.float32))
    probs = torch.sigmoid(logits).numpy()

    wanted = set("label_" + str(k) for k in (condition_keys or []))
    idx = np.array([i for i, lab in enumerate(labels) if lab in wanted])
    if len(idx) == 0:
        idx = np.arange(len(labels))

    return probs[:, idx].mean(axis=1)
