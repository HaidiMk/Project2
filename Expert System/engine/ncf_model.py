

import os
from functools import reduce
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import pandas as pd
import numpy as np



def _pick(cond, when_true, when_false):
    """بديل التعبير الثلاثي."""
    return {True: when_true, False: when_false}[bool(cond)]


def _do(cond, do_true: Callable, do_false: Callable):
    """بديل if/else تنفيذي."""
    return {True: do_true, False: do_false}[bool(cond)]()


def _fold(seq, acc, fn: Callable[[Any, Any], Any]):
    """تطبيق تراكمي بلا عودية: functools.reduce بدل استدعاء الدالة لنفسها."""
    return reduce(lambda accumulator, item: fn(accumulator, item), list(seq), acc)


def _attempt(fn: Callable, on_error):
    """بديل try/except: ينفّذ fn ويعيد on_error(الاستثناء) عند الفشل."""
    try:
        return fn()
    except Exception as exc:                       
        return on_error(exc)




class NCFModel(nn.Module):
    """شبكة بسيطة: ميزات غذائية → درجة بين 0 و 5."""

    def __init__(self, input_dim: int = 10):
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
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) * 5.0



FEATURE_COLS = [
    "Calories", "ProteinContent", "FatContent",
    "CarbohydrateContent", "FiberContent", "SugarContent",
    "SodiumContent", "SaturatedFatContent",
    "CholesterolContent", "Rating",
]

GOAL_WEIGHTS = {
    "weight_loss":    [0.3, 0.8, 0.1, 0.2, 0.7, 0.1, 0.2, 0.1, 0.1, 0.5],
    "weight_gain":    [0.9, 0.9, 0.7, 0.8, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5],
    "muscle_gain":    [0.6, 1.0, 0.4, 0.6, 0.5, 0.2, 0.2, 0.2, 0.2, 0.5],
    "maintenance":    [0.5, 0.6, 0.4, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.5],
    "heart_health":   [0.4, 0.6, 0.2, 0.4, 0.8, 0.2, 0.1, 0.1, 0.1, 0.5],
    "healthy_eating": [0.5, 0.6, 0.3, 0.4, 0.7, 0.2, 0.2, 0.2, 0.2, 0.5],
    "pregnancy_diet": [0.7, 0.9, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3, 0.3, 0.5],
    "elderly_diet":   [0.5, 0.8, 0.3, 0.4, 0.6, 0.2, 0.2, 0.2, 0.2, 0.5],
}




class NCFRecommender:
    """واجهة الـ NCF للاستخدام في filtering_engine."""

    MODEL_PATH = "data/ncf_model.pt"

    def __init__(self):
        self.model   = NCFModel(input_dim=len(FEATURE_COLS))
        self.trained = False
        self._mins   = None
        self._maxs   = None

        exists = os.path.exists(self.MODEL_PATH)
        _do(exists, self._try_load, lambda: None)

    def _try_load(self):
        """حمّل النموذج المحفوظ بأمان (بلا try/except صريح كبنية تحكم)."""
        def load_ok():
            self.model.load_state_dict(
                torch.load(self.MODEL_PATH, map_location="cpu",
                           weights_only=True)
            )
            self.model.eval()
            self.trained = True
            print("   NCF model loaded from disk.")
            return None

        def load_fail(_exc):
            self.trained = False
            return None

        return _attempt(load_ok, load_fail)

   
    def train(self, df: pd.DataFrame, epochs: int = 5):
        """تدريب النموذج عبر functools.reduce على الحقب والدفعات بلا عودية."""
        print("\n   Training NCF model...")

        data = df[FEATURE_COLS].copy()
        data = data.fillna(data.median())

        self._mins = data.min()
        self._maxs = data.max()
        diff = (self._maxs - self._mins).replace(0, 1)
        normalized = (data - self._mins) / diff

        X = torch.tensor(normalized.values, dtype=torch.float32)
        y = torch.tensor(
            (df["Rating"].fillna(3.0).values / 5.0),
            dtype=torch.float32,
        ).unsqueeze(1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.model.train()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=True
        )

        
        def run_batch(total_loss, batch):
            X_batch, y_batch = batch
            optimizer.zero_grad()
            pred = self.model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            return total_loss + loss.item()

       
        def run_epoch(epoch_idx, _):
            total = _fold(loader, 0.0, run_batch)
            avg = total / max(len(loader), 1)
            print(f"   Epoch {epoch_idx + 1}/{epochs} — Loss: {avg:.4f}")
            return epoch_idx + 1

       
        epoch_seq = list(range(epochs))
        _fold(epoch_seq, 0, run_epoch)

        self.model.eval()
        self.trained = True

        torch.save(self.model.state_dict(), self.MODEL_PATH)
        print(f"   NCF model saved to {self.MODEL_PATH}")

    # ──────────────────────────────────────────────────────
    def predict_scores(
        self,
        df: pd.DataFrame,
        goal: Optional[str] = None,
    ) -> np.ndarray:
        """أرجع درجة NCF لكل وصفة (أصفار لو لم يُدرَّب)."""

        def not_trained():
            return np.zeros(len(df))

        def trained():
            data = df[FEATURE_COLS].copy().fillna(0)

            
            def normalize():
                diff = (self._maxs - self._mins).replace(0, 1)
                return (data - self._mins) / diff

            data2 = _do(self._mins is not None, normalize, lambda: data)

            # تطبيق أوزان الهدف لو موجوداً في الجدول
            has_goal = bool(goal) and (goal in GOAL_WEIGHTS)
            weights = np.array(GOAL_WEIGHTS.get(_pick(has_goal, goal, ""),
                                               [1.0] * len(FEATURE_COLS)))
            weighted = _pick(has_goal, data2 * weights, data2)

            X = torch.tensor(weighted.values, dtype=torch.float32)

            def infer():
                with torch.no_grad():
                    return self.model(X).squeeze().numpy()

            return infer()

        return _do(self.trained, trained, not_trained)
