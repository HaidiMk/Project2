"""
ncf_model.py — Neural Collaborative Filtering
==============================================
نموذج NCF مبسط يعتمد على:
    - القيم الغذائية للوصفة
    - هدف المستخدم الغذائي
    - تقييمات الوصفات (Rating)

يُدمج مع Expert System في filtering_engine.py
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional


# ══ تعريف النموذج ════════════════════════════════════════════
class NCFModel(nn.Module):
    """
    شبكة عصبية بسيطة تأخذ:
        - input_dim: عدد الميزات (القيم الغذائية + الهدف)
    وترجع:
        - درجة توقعية بين 0 و 5
    """
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
        return self.network(x) * 5.0   # درجة بين 0 و 5


# ══ الأعمدة الغذائية المستخدمة ═══════════════════════════════
FEATURE_COLS = [
    "Calories", "ProteinContent", "FatContent",
    "CarbohydrateContent", "FiberContent", "SugarContent",
    "SodiumContent", "SaturatedFatContent",
    "CholesterolContent", "Rating",
]

# ══ أوزان الأهداف ════════════════════════════════════════════
GOAL_WEIGHTS = {
    "weight_loss":    [0.3, 0.8, 0.1, 0.2, 0.7, 0.1, 0.2, 0.1, 0.1, 0.5],
    "weight_gain":    [0.9, 0.9, 0.7, 0.8, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5],
    "muscle_gain":    [0.6, 1.0, 0.4, 0.6, 0.5, 0.2, 0.2, 0.2, 0.2, 0.5],
    "maintenance":    [0.5, 0.6, 0.4, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.5],
    "heart_health":   [0.4, 0.6, 0.2, 0.4, 0.8, 0.2, 0.1, 0.1, 0.1, 0.5],
    "healthy_eating": [0.5, 0.6, 0.3, 0.4, 0.7, 0.2, 0.2, 0.2, 0.2, 0.5],
    "pregnancy_diet": [0.7, 0.9, 0.4, 0.5, 0.6, 0.3, 0.3, 0.3, 0.3, 0.5],
    "elderly_diet":   [0.5, 0.8, 0.3, 0.4, 0.6, 0.2, 0.2, 0.2, 0.2, 0.5],
    "weight_gain":    [0.9, 0.9, 0.7, 0.8, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5],
}


class NCFRecommender:
    """
    واجهة الـ NCF للاستخدام في filtering_engine.
    """

    MODEL_PATH = "data/ncf_model.pt"

    def __init__(self):
        self.model    = NCFModel(input_dim=len(FEATURE_COLS))
        self.trained  = False
        self._mins    = None
        self._maxs    = None

        # حمّل النموذج لو موجود
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(self.MODEL_PATH, map_location="cpu",
                               weights_only=True)
                )
                self.model.eval()
                self.trained = True
                print("   NCF model loaded from disk.")
            except Exception:
                self.trained = False

    def train(self, df: pd.DataFrame, epochs: int = 5):
        """تدريب النموذج على البيانات الموجودة."""
        print("\n   Training NCF model...")

        # تحضير البيانات
        data = df[FEATURE_COLS].copy()
        data = data.fillna(data.median())

        # تطبيع القيم
        self._mins = data.min()
        self._maxs = data.max()
        diff = (self._maxs - self._mins).replace(0, 1)
        normalized = (data - self._mins) / diff

        # الهدف = Rating مطبّع
        X = torch.tensor(normalized.values, dtype=torch.float32)
        y = torch.tensor(
            (df["Rating"].fillna(3.0).values / 5.0),
            dtype=torch.float32
        ).unsqueeze(1)

        # تدريب
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        self.model.train()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"   Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")

        self.model.eval()
        self.trained = True

        # احفظ النموذج
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        print(f"   NCF model saved to {self.MODEL_PATH}")

    def predict_scores(
        self,
        df: pd.DataFrame,
        goal: Optional[str] = None,
    ) -> np.ndarray:
        """
        أرجع درجة NCF لكل وصفة.
        لو النموذج ما اتدرب — يرجع أصفار.
        """
        if not self.trained:
            return np.zeros(len(df))

        data = df[FEATURE_COLS].copy().fillna(0)

        # تطبيع
        if self._mins is not None:
            diff = (self._maxs - self._mins).replace(0, 1)
            data = (data - self._mins) / diff

        # تطبيق أوزان الهدف
        if goal and goal in GOAL_WEIGHTS:
            weights = np.array(GOAL_WEIGHTS[goal])
            data = data * weights

        X = torch.tensor(data.values, dtype=torch.float32)

        with torch.no_grad():
            scores = self.model(X).squeeze().numpy()

        return scores