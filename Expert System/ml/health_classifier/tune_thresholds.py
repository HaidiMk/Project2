import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from health_classifier import (                 
    set_seed, load_data, train_val_test_split,
    HealthClassifier, SCALER_PATH, MODEL_PATH,
)

THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "health_classifier_thresholds.json"
)

GRID = np.arange(0.10, 0.951, 0.05)            


def main():
    set_seed()
    X, Y, label_cols = load_data()
    _, val_idx, _ = train_val_test_split(X, Y)
    print(f"Tuning on validation set only: {len(val_idx):,} recipes")

    scaler = joblib.load(SCALER_PATH)
    X_val = scaler.transform(X.iloc[val_idx])
    y_val = Y.iloc[val_idx].values

    model = HealthClassifier()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32))).numpy()

    y_true_bin = (y_val >= 0.5).astype(int)

    best_thresholds = {}
    print(f"\n{'condition':<28} {'best thr':>9} {'F1@0.5':>8} {'F1@best':>8} {'delta':>8}")
    print("-" * 65)
    for i, col in enumerate(label_cols):
        yt, p = y_true_bin[:, i], probs[:, i]
        f1_by_thr = [
            f1_score(yt, (p >= t).astype(int), zero_division=0) for t in GRID
        ]
        best_idx = int(np.argmax(f1_by_thr))
        best_t = float(GRID[best_idx])
        best_thresholds[col] = best_t
        f1_flat = f1_score(yt, (p >= 0.5).astype(int), zero_division=0)
        print(f"{col:<28} {best_t:>9.2f} {f1_flat:>8.4f} "
              f"{f1_by_thr[best_idx]:>8.4f} {f1_by_thr[best_idx]-f1_flat:>+8.4f}")

    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\nSaved -> {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
