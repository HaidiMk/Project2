"""
topsis_model.py — TOPSIS Algorithm (Layer 2)
===============================================
خوارزمية TOPSIS (Technique for Order Preference by Similarity
to Ideal Solution) — ترتيب الوصفات حسب عدة معايير غذائية بنفس
الوقت، بدل الاعتماد على معادلة وزن واحدة بس.

الخطوات:
    1. بناء مصفوفة القرار (Decision Matrix) من القيم الغذائية
    2. تطبيع المصفوفة (Vector Normalization)
    3. تطبيق أوزان المعايير حسب هدف المستخدم
    4. تحديد الحل الأمثل الموجب (Ideal Best) والسالب (Ideal Worst)
    5. حساب المسافة الإقليدية من كل وصفة للحلين
    6. حساب درجة التقارب (Closeness Score) — بين 0 و 1
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple


# ══ المعايير المستخدمة بـ TOPSIS ═════════════════════════════
CRITERIA: List[str] = [
    "Calories", "ProteinContent", "FatContent",
    "CarbohydrateContent", "FiberContent",
    "SugarContent", "SodiumContent",
]

# ══ كل هدف له (وزن المعيار, اتجاهه) ══════════════════════════
# "benefit" = الأعلى أحسن (زي البروتين والفايبر دايماً)
# "cost"    = الأقل أحسن (زي السكر والصوديوم دايماً)
# ⚠️ مهم: الاتجاه نفسه بيتغيّر حسب الهدف — مثلاً Calories
#    "cost" بـ weight_loss (الأقل أحسن) بس "benefit" بـ
#    weight_gain (الأكثر أحسن). هذا الفرق هو سبب التصحيح.
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

# افتراضي لو ما في هدف محدد أو الهدف مش موجود بالقائمة فوق
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
    """
    احسب درجة TOPSIS لكل وصفة بـ df.
    الناتج: مصفوفة قيم بين 0 و 1 — كل ما زادت القيمة كانت
    الوصفة أقرب للحل المثالي (أفضل توازن بين المعايير).
    """
    available = [c for c in CRITERIA if c in df.columns]
    if not available or len(df) == 0:
        return np.zeros(len(df))

    # 1) مصفوفة القرار
    matrix = df[available].fillna(0).to_numpy(dtype=float)

    # 2) التطبيع — نقسم كل عمود على جذر مجموع مربعاته
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom[denom == 0] = 1.0   # تجنب القسمة على صفر
    normalized = matrix / denom

    # 3) تطبيق الأوزان حسب الهدف (الوزن والاتجاه مع بعض)
    criteria_map = GOAL_CRITERIA.get(goal, DEFAULT_CRITERIA)
    weights = np.array([criteria_map.get(c, (0.0, "cost"))[0] for c in available])
    weighted = normalized * weights

    # 4) الحل الأمثل الموجب والسالب لكل معيار
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

    # 5) المسافة الإقليدية عن الحل الأمثل والأسوأ
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # 6) درجة التقارب النهائية
    total_dist = dist_best + dist_worst
    total_dist[total_dist == 0] = 1.0   # تجنب القسمة على صفر
    closeness = dist_worst / total_dist

    return closeness