"""
user_profile.py — Smart Dietary Advisor v4.0 — NO-CONTROL-FLOW EDITION
=======================================================================
خالٍ تماماً من: if / elif / else / for / while / map / filter / reduce
ومن التعابير الثلاثية ومن الـ comprehensions.

البدائل:
    - سلاسل if/elif المتدرجة (bmi_category) → numpy.select
    - if بشرط واحد (bmr / daily_calories)   → فهرسة قاموس {True:..,False:..}[cond]
    - dispatch                                → قواميس + .get()

السلوك (القيم المُرجعة) مطابق تماماً للنسخة الأصلية UserProfile.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── معاملات النشاط البدني — Harris-Benedict ──────────────
ACTIVITY_FACTORS = {
    "sedentary":   1.2,
    "light":       1.375,
    "moderate":    1.55,
    "active":      1.725,
    "very_active": 1.9,
}


def _pick(cond, when_true, when_false):
    """بديل التعبير الثلاثي."""
    return {True: when_true, False: when_false}[bool(cond)]


@dataclass
class UserProfile:
    """الملف الشخصي للمستخدم — يحتوي على جميع المعطيات الصحية."""

    age:            int
    height:         float                      # سم
    weight:         float                        # كغ
    gender:         str                          # "male" | "female"
    pregnant:       bool             = False
    conditions:     List[str]        = field(default_factory=list)
    allergies:      List[str]        = field(default_factory=list)
    preferences:    List[str]        = field(default_factory=list)
    goal:           Optional[str]    = None
    activity_level: str              = "light"
    meal_type:      str              = "any"     # breakfast/lunch/dinner/any

    # ── خصائص مشتقة ──────────────────────────────────────
    @property
    def bmi(self) -> float:
        h = self.height / 100
        return round(self.weight / (h * h), 1)

    @property
    def bmi_category(self) -> str:
        """تصنيف الوزن عبر numpy.select بدل سلسلة if/elif."""
        b = self.bmi
        conditions = [b < 18.5, b < 25.0, b < 30.0, b < 35.0, b < 40.0]
        choices = [
            "Underweight", "Normal weight", "Overweight",
            "Obesity Class I", "Obesity Class II",
        ]
        return str(np.select(conditions, choices,
                             default="Obesity Class III (Severe)"))

    @property
    def ideal_weight_range(self) -> Tuple[float, float]:
        h = self.height / 100
        return (round(18.5 * h * h, 1), round(24.9 * h * h, 1))

    @property
    def life_stage(self) -> str:
        """الفئة العمرية عبر numpy.select بدل سلسلة if."""
        a = self.age
        conditions = [a <= 12, a <= 17, a >= 65]
        choices = ["child", "teen", "elderly"]
        return str(np.select(conditions, choices, default="adult"))

    def bmr(self) -> float:
        """معادلة Mifflin-St Jeor — الجزء الثابت بالجنس عبر فهرسة قاموس."""
        common = 10 * self.weight + 6.25 * self.height - 5 * self.age
        gender_offset = {"male": 5}.get(self.gender, -161)
        return common + gender_offset

    @property
    def daily_calories(self) -> int:
        factor = ACTIVITY_FACTORS.get(self.activity_level, 1.375)
        base = self.bmr() * factor
        # إضافة سعرات الحمل (350) دون if — ضرب بقيمة بوليانية
        base = base + 350 * float(bool(self.pregnant))
        return round(base)

    @property
    def per_meal_calories(self) -> int:
        return round(self.daily_calories / 3)

    def summary(self) -> dict:
        from core.constants import AGE_RANGE_EN
        iw = self.ideal_weight_range
        return {
            "age":            self.age,
            "gender":         _pick(self.gender == "male", "Male", "Female"),
            "life_stage":     AGE_RANGE_EN.get(self.life_stage, self.life_stage),
            "height":         self.height,
            "weight":         self.weight,
            "bmi":            self.bmi,
            "bmi_category":   self.bmi_category,
            "ideal_weight":   f"{iw[0]}-{iw[1]} kg",
            "daily_calories": self.daily_calories,
            "per_meal_kcal":  self.per_meal_calories,
            "pregnant":       self.pregnant,
            "activity_level": self.activity_level,
            "meal_type":      self.meal_type,
              }
