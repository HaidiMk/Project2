import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from health_classifier import (                 
    set_seed, load_data, train_val_test_split,
    HealthClassifier, SCALER_PATH, MODEL_PATH,
)

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS_MD = RESULTS_DIR / "metrics.md"
METRICS_JSON = RESULTS_DIR / "metrics.json"
METRICS_TUNED_MD = RESULTS_DIR / "metrics_tuned.md"
METRICS_TUNED_JSON = RESULTS_DIR / "metrics_tuned.json"
THRESHOLDS_PATH = BASE_DIR / "data" / "health_classifier_thresholds.json"

FLAT_THRESHOLD = 0.5
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "auc")


def fmt(v, width: int) -> str:
    return "   n/a" if (v is None or np.isnan(v)) else f"{v:{width}.4f}"


def compute_metrics(yt: np.ndarray, yp: np.ndarray, p: np.ndarray) -> dict:
    has_both_classes = len(np.unique(yt)) == 2
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "auc": float(roc_auc_score(yt, p)) if has_both_classes else None,
    }


def macro_of(rows) -> dict:
    return {m: float(np.nanmean([r[m] for r in rows])) for m in METRIC_KEYS}


def write_report(path_md: Path, path_json: Path, rows, macro, header_note: str,
                 thresholds: dict = None):
    lines = [
        "# Health Classifier — Test Set Metrics",
        "",
        header_note,
        "",
        "| condition | accuracy | precision | recall | F1 | AUC |"
        + (" threshold |" if thresholds else ""),
        "|---|---|---|---|---|---|" + ("---|" if thresholds else ""),
    ]
    for r in rows:
        auc_str = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
        base = (f"| {r['condition']} | {r['accuracy']:.4f} | {r['precision']:.4f} "
                f"| {r['recall']:.4f} | {r['f1']:.4f} | {auc_str} |")
        if thresholds:
            base += f" {thresholds[r['condition']]:.2f} |"
        lines.append(base)
    auc_macro = "n/a" if np.isnan(macro["auc"]) else f"{macro['auc']:.4f}"
    lines.append(
        f"| **macro-average** | {macro['accuracy']:.4f} | {macro['precision']:.4f} "
        f"| {macro['recall']:.4f} | {macro['f1']:.4f} | {auc_macro} |"
        + (f" |" if thresholds else "")
    )
    lines.append("")

    payload = {
        "macro_average": macro,
        "conditions": rows,
    }
    if thresholds:
        payload["thresholds"] = {r["condition"]: thresholds[r["condition"]]
                                 for r in rows}

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true",
                        help="use per-condition thresholds from "
                             "data/health_classifier_thresholds.json")
    args = parser.parse_args()

    set_seed()

    X, Y, label_cols = load_data()
    _, _, test_idx = train_val_test_split(X, Y)
    print(f"Test set: {len(test_idx):,} recipes x {len(label_cols)} conditions")

    scaler = joblib.load(SCALER_PATH)
    X_test = scaler.transform(X.iloc[test_idx])
    y_test = Y.iloc[test_idx].values

    model = HealthClassifier()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
    probs = torch.sigmoid(logits).numpy()

    y_true_bin = (y_test >= FLAT_THRESHOLD).astype(int)

    rows_flat = [
        compute_metrics(y_true_bin[:, i], (probs[:, i] >= FLAT_THRESHOLD).astype(int), probs[:, i])
        for i in range(len(label_cols))
    ]
    for r, col in zip(rows_flat, label_cols):
        r["condition"] = col.removeprefix("label_")

    macro_flat = macro_of(rows_flat)

    if not args.tuned:
        print_table(rows_flat, macro_flat)
        write_report(
            METRICS_MD, METRICS_JSON, rows_flat, macro_flat,
            f"- Test set: **{len(test_idx):,}** recipes | flat threshold: **0.5** "
            f"| macro = simple mean over 22 conditions",
        )
        print(f"\nSaved -> {METRICS_MD}")
        print(f"Saved -> {METRICS_JSON}")
        return

    with open(THRESHOLDS_PATH, encoding="utf-8") as f:
        thresholds = json.load(f)
    assert set(thresholds) == set(label_cols), "threshold keys != label keys"

    rows_tuned = []
    for i, col in enumerate(label_cols):
        thr = thresholds[col]
        r = compute_metrics(
            y_true_bin[:, i], (probs[:, i] >= thr).astype(int), probs[:, i]
        )
        r["condition"] = col.removeprefix("label_")
        r["threshold"] = float(thr)
        rows_tuned.append(r)

    macro_tuned = macro_of(rows_tuned)

    combined = sorted(
        ((r["condition"], r["f1"], rt["f1"], rt["threshold"], rt["f1"] - r["f1"])
         for r, rt in zip(rows_flat, rows_tuned)),
        key=lambda x: x[4], reverse=True,
    )
    print("\n" + "=" * 78)
    print("Side-by-side on TEST set — sorted by improvement")
    print("=" * 78)
    print(f"{'condition':<26} {'F1@0.5':>9} {'F1@tuned':>9} {'threshold':>10} {'delta':>8}")
    print("-" * 78)
    for cond, f1_flat, f1_tuned, thr, delta in combined:
        print(f"{cond:<26} {f1_flat:>9.4f} {f1_tuned:>9.4f} {thr:>10.2f} {delta:>+8.4f}")
    print("-" * 78)
    print(f"{'macro-average':<26} {macro_flat['f1']:>9.4f} {macro_tuned['f1']:>9.4f} "
          f"{'':>10} {macro_tuned['f1']-macro_flat['f1']:>+8.4f}")

    thr_by_cond = {r["condition"]: r["threshold"] for r in rows_tuned}
    write_report(
        METRICS_TUNED_MD, METRICS_TUNED_JSON, rows_tuned, macro_tuned,
        f"- Test set: **{len(test_idx):,}** recipes | per-condition thresholds "
        f"(tuned on validation only, never test) | macro = simple mean over 22 conditions",
        thresholds=thr_by_cond,
    )
    print(f"\nSaved -> {METRICS_TUNED_MD}")
    print(f"Saved -> {METRICS_TUNED_JSON}")


def print_table(rows, macro):
    hdr = (f"{'condition':<28} {'acc':>8} {'prec':>8} "
           f"{'rec':>8} {'f1':>8} {'auc':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['condition']:<28} "
              f"{fmt(r['accuracy'], 8)} {fmt(r['precision'], 8)} "
              f"{fmt(r['recall'], 8)} {fmt(r['f1'], 8)} {fmt(r['auc'], 8)}")
    print("-" * len(hdr))
    print(f"{'macro-average':<28} "
          f"{fmt(macro['accuracy'], 8)} {fmt(macro['precision'], 8)} "
          f"{fmt(macro['recall'], 8)} {fmt(macro['f1'], 8)} {fmt(macro['auc'], 8)}")


if __name__ == "__main__":
    main()
