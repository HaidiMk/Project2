"""
user_profile.py — Smart Dietary Advisor v4.0
=============================================
تعريف كلاس UserProfile مع جميع الحسابات الصحية:
    - BMI وفئة الوزن
    - BMR (معادلة Mifflin-St Jeor)
    - السعرات اليومية والسعرات لكل وجبة
    - الفئة العمرية (child / teen / adult / elderly)
    - مستوى النشاط البدني (activity_level)
    - نوع الوجبة (meal_type)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── معاملات النشاط البدني — Harris-Benedict ──────────────
ACTIVITY_FACTORS = {
    "sedentary":   1.2,
    "light":       1.375,
    "moderate":    1.55,
    "active":      1.725,
    "very_active": 1.9,
}


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
    meal_type:      str              = "any"     # ← جديد: breakfast/lunch/dinner/any

    # ── خصائص مشتقة ──────────────────────────────────────
    @property
    def bmi(self) -> float:
        h = self.height / 100
        return round(self.weight / (h * h), 1)

    @property
    def bmi_category(self) -> str:
        b = self.bmi
        if b < 18.5: return "Underweight"
        if b < 25.0: return "Normal weight"
        if b < 30.0: return "Overweight"
        if b < 35.0: return "Obesity Class I"
        if b < 40.0: return "Obesity Class II"
        return "Obesity Class III (Severe)"

    @property
    def ideal_weight_range(self) -> Tuple[float, float]:
        h = self.height / 100
        return (round(18.5 * h * h, 1), round(24.9 * h * h, 1))

    @property
    def life_stage(self) -> str:
        if self.age <= 12: return "child"
        if self.age <= 17: return "teen"
        if self.age >= 65: return "elderly"
        return "adult"

    def bmr(self) -> float:
        if self.gender == "male":
            return 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        return 10 * self.weight + 6.25 * self.height - 5 * self.age - 161

    @property
    def daily_calories(self) -> int:
        factor = ACTIVITY_FACTORS.get(self.activity_level, 1.375)
        base = self.bmr() * factor
        if self.pregnant:
            base += 350
        return round(base)

    @property
    def per_meal_calories(self) -> int:
        return round(self.daily_calories / 3)

    def summary(self) -> dict:
        from core.constants import AGE_RANGE_EN
        iw = self.ideal_weight_range
        return {
            "age":            self.age,
            "gender":         "Male" if self.gender == "male" else "Female",
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
            "meal_type":      self.meal_type,    # ← جديد
        }