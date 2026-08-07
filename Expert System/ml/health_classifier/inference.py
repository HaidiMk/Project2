"""
inference.py — الاستدلال على درجة الصحة الغذائية (AI Health Score)
=====================================================================
يحمّل مصنّف HealthClassifier المدرب + الـ scaler + مفاتيح الحالات الـ 22
مرة واحدة فقط (singleton) عند أول استدعاء، ويحسب لكل وصفة:

    ai_health_score = mean( sigmoid(logits) ) على الحالات الطبية
    المطبّقة على المستخدم فقط (condition_keys = مثلاً ["diabetes"])
    — الحالات غير المدرّبة (مثل label_lactose_intolerance) تُتخطّى.
    لو لم يُمرَّر أي مفتاح أو لم يطابق أي منها المدرّبين، نستخدم
    متوسط الحالات الـ 22 كلها (fallback محايد).

التكامل:
    engine/filtering_engine.py يستدعي ai_health_score داخل filter_recipes
    ويضيف الناتج عموداً جديداً _ai_health_score (Layer 1.5)، ثم
    TOPSIS/topsis_model.rank_with_topsis يدمجه بالمعادلة:
        final = 0.4*TOPSIS + 0.4*AI + 0.2*expert_normalized

المسارات كلها مطلقة (مرساة على موقع هذا الملف) — حماية من أخطاء
الـ cwd النسبية.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[2]          # Expert System/
sys.path.insert(0, str(BASE_DIR))

from ml.health_classifier.health_classifier import HealthClassifier, NUTRITION_COLS

MODEL_PATH = BASE_DIR / "data" / "health_classifier.pt"
SCALER_PATH = BASE_DIR / "data" / "health_scaler.pkl"
LABELS_PATH = BASE_DIR / "data" / "health_classifier_labels.json"

_model = None
_scaler = None
_labels = None


def _load():
    """حمّل المصنّف + الـ scaler + المفاتيح مرة واحدة فقط (singleton)."""
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
    """
    متوسط احتمال (sigmoid) الحالات المطلوبة لكل وصفة في df.

    المعاملات:
        df: إطار الوصفات (يجب أن يحوي أعمدة NUTRITION_COLS الغذائية)
        condition_keys: قائمة مفاتيح الحالات (مثل "diabetes") أو None
            — كل مفتاح يُرسم إلى "label_<key>"؛ غير المطابق يتخطّى.
    المخرجات:
        np.ndarray بطول len(df) — قيم بين 0 و 1 (كلما ارتفعت كانت
        الوصفة أكثر توافقاً مع الحالة المطلوبة).
    """
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
