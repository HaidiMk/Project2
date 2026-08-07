"""
health_classifier.py — تدريب مصنّف متعدد الملصقات للحالات الطبية
==================================================================
المدخلات:  Expert System/ml/data/labeled_recipes.csv
           (RecipeId + 9 قيم غذائية + 24 ملصقاً ناعماً)

الأهداف (22 فقط): نستبعد label_lactose_intolerance و label_gluten_intolerance
           — قيمتان شبه ثابتتين (~99.8% موجبة) بلا إشارة فعلية؛ تبقى
           معالجة هاتين الحالتين عبر الفلترة النصية في النظام الخبير.

الشبكة (بنمط المكونات الغذائية — Sequential + ReLU + Dropout(0.2)):
    9 ميزات غذائية -> Linear(64)+ReLU+Dropout -> Linear(32)+ReLU+Dropout
                    -> Linear(16)+ReLU -> Linear(22)  (logits خام)

    معيار الخسارة: BCEWithLogitsLoss (سوفت ماكس الداخل: sigmoid) مع
    pos_weight لكل حالة = n_negative / n_positive (من مجموعة التدريب،
    عند عتبة 0.5، بسقف 10 لتفادي عدم الاستقرار).

    الأهداف المستخدمة هي الملصقات الناعمة نفسها (∈ {0} ∪ [0.5, 1.0])
    — BCE تدعم أهدافاً عشرية في [0,1] (تفسير soft-label) فتحتفظ بإشارة
    جودة التقييم التي بنيناها في build_labels.py.

التقسيم 70/15/15:
    الطبقات الحقيقية هنا متعددة الملصقات (22 حالة) — لا توجد دالة
    stratification متعددة الملصقات في sklearn. نستخدم وكيلاً (proxy):
    تقسيم طبقية على label_diabetes المُثنّى (>= 0.5) — الحالة الطبية
    الأكثر تمثيلاً للمخاطر والأكثر حرصاً في بياناتنا، فهي بديل عملي
    يضمن بقاء الحالات النادرة موزّعة بتوازن عبر الأقسام الثلاثة.
    نتحقق لاحقاً بمطبوعة توازن كل حالة عبر الأقسام.

التطبيع: StandardScaler مُلاءَم على التدريب فقط، محفوظ بـ joblib إلى
    Expert System/data/health_scaler.pkl لاستخدامه في الاستدلال.

المخرجات:
    Expert System/data/health_classifier.pt     — أفضل أوزان (early stopping)
    Expert System/data/health_classifier_labels.json — المفاتيح الـ 22 بنفس ترتيب المخرجات
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]              # Expert System/
LABELED_PATH = BASE_DIR / "ml" / "data" / "labeled_recipes.csv"
SCALER_PATH = BASE_DIR / "data" / "health_scaler.pkl"
MODEL_PATH = BASE_DIR / "data" / "health_classifier.pt"
LABELS_PATH = BASE_DIR / "data" / "health_classifier_labels.json"

NUTRITION_COLS = [
    "Calories", "ProteinContent", "CarbohydrateContent", "SugarContent",
    "SodiumContent", "FatContent", "SaturatedFatContent",
    "CholesterolContent", "FiberContent",
]

# ملصقات مستبعدة من التدريب — شبه ثابتة، تبقى ضمن الفلترة النصية للنظام الخبير
EXCLUDED_LABELS = {"label_lactose_intolerance", "label_gluten_intolerance"}

INPUT_DIM = 9
NUM_CONDITIONS = 22
SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 512
MAX_EPOCHS = 40
PATIENCE = 10
POS_WEIGHT_CAP = 10.0


class HealthClassifier(nn.Module):
    """شبكة تغذية أمامية متعددة الملصقات (MLP بسيط)."""

    def __init__(self, input_dim: int = INPUT_DIM, num_conditions: int = NUM_CONDITIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_conditions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)      # logits خام — sigmoid داخل BCEWithLogitsLoss


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    df = pd.read_csv(LABELED_PATH, low_memory=False)
    label_cols = [
        c for c in df.columns
        if c.startswith("label_") and c not in EXCLUDED_LABELS
    ]
    assert len(label_cols) == NUM_CONDITIONS, (
        f"Expected {NUM_CONDITIONS} condition labels, got {len(label_cols)}"
    )
    X = df[NUTRITION_COLS].astype(np.float32)
    Y = df[label_cols].astype(np.float32)
    assert not X.isna().any().any(), "NaN found in nutritional features"
    return X, Y, label_cols


def train_val_test_split(X, Y):
    """تقسيم 70/15/15 مع stratification تقريبي على label_diabetes المُثنّى.

    حجّة الوكيل (proxy): الحالات 22 متعددة الملصقات لا تدعمها
    train_test_split مباشرة؛ اختيار السكري كبُعد طبقية لأنه الحالة
    الطبية الأكثر تمثيلاً (من حيث انتشارها وسلامتها الغذائية) في
    بياناتنا، فيضمن بقاء نسبتها موزّعة بالتساوي عبر الأقسام. باقي
    الحالات تُتحقّق توازنها بالمطبوعة اللاحقة.
    """
    diabetes_bin = (Y["label_diabetes"] >= 0.5).astype(int).values
    idx = np.arange(len(X))

    train_idx, temp_idx = train_test_split(
        idx, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED,
        stratify=diabetes_bin,
    )
    temp_bin = diabetes_bin[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED, stratify=temp_bin,
    )
    return train_idx, val_idx, test_idx


def compute_pos_weight(y_train: np.ndarray, cap: float = POS_WEIGHT_CAP) -> torch.Tensor:
    """n_negative / n_positive لكل حالة من مجموعة التدريب (عتبة 0.5) بسقف cap."""
    n_pos = (y_train >= 0.5).sum(axis=0).astype(np.float64)
    n_neg = y_train.shape[0] - n_pos
    ratios = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), cap)
    return torch.tensor(np.clip(ratios, 1.0, cap), dtype=torch.float32)


def make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
    )


def run_training(model, X_train, y_train, X_val, y_val,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """تدريب مع early stopping على خسارة التحقق؛ يعيد أفضل الأوزان والتاريخ."""
    pos_weight = compute_pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_val, best_state, best_epoch = float("inf"), None, -1
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        train_loss = total / count

        model.eval()
        total_v, count_v = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                loss = criterion(model(xb), yb)
                total_v += loss.item() * xb.size(0)
                count_v += xb.size(0)
        val_loss = total_v / count_v

        history.append((epoch, train_loss, val_loss))
        print(f"  epoch {epoch:>3} | train {train_loss:.4f} | val {val_loss:.4f}"
              f" | patience {epochs_no_improve}/{patience}")

        if val_loss < best_val - 1e-4:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience}).")
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


def print_split_balance(X, Y, train_idx, val_idx, test_idx, label_cols):
    """تحقّق توازن الحالات عبر الأقسام (نسب الموجب عند عتبة 0.5)."""
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    print("\nClass balance (positive % @0.5) per split:")
    print(f"  {'condition':<28} {'train':>7} {'val':>7} {'test':>7}")
    for col in label_cols:
        row = []
        for name, idx in splits.items():
            frac = (Y.loc[idx, col] >= 0.5).mean() * 100
            row.append(frac)
        print(f"  {col.removeprefix('label_'):<28} {row[0]:>6.2f}% {row[1]:>6.2f}% {row[2]:>6.2f}%")


def main():
    set_seed()
    print("Loading labeled data...")
    X, Y, label_cols = load_data()
    print(f"Loaded {len(X):,} recipes x {len(label_cols)} condition labels")

    train_idx, val_idx, test_idx = train_val_test_split(X, Y)
    print(f"Split: train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")

    # scaler على التدريب فقط
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_val = scaler.transform(X.iloc[val_idx])
    X_test = scaler.transform(X.iloc[test_idx])

    y_train = Y.iloc[train_idx].values
    y_val = Y.iloc[val_idx].values
    y_test = Y.iloc[test_idx].values

    print_split_balance(X, Y, train_idx, val_idx, test_idx, label_cols)

    print("\nBuilding model + training (CPU)...")
    model = HealthClassifier()
    t0 = time.time()
    model, history, best_epoch, best_val = run_training(
        model, X_train, y_train, X_val, y_val,
    )
    elapsed = time.time() - t0
    train_at_best = dict((e, tr) for e, tr, _ in history)[best_epoch]

    print(f"\nTraining done in {elapsed:.0f}s. "
          f"Best epoch {best_epoch}: train {train_at_best:.4f} | val {best_val:.4f}")

    # حفظ المخرجات — test لا يُستخدم في هذا التمرير (خطوة لاحقة)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_cols, f, indent=2)

    print(f"Saved model      -> {MODEL_PATH}")
    print(f"Saved scaler     -> {SCALER_PATH}")
    print(f"Saved labels     -> {LABELS_PATH}  ({len(label_cols)} conditions)")


if __name__ == "__main__":
    main()
