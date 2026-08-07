"""
run_topsis.py — ربط طبقة TOPSIS (Layer 2) بالنظام الخبير (Layer 1)
====================================================================
الفكرة:
    1. نشغّل النظام الخبير (Expert System) عادي — يفلتر الوصفات
       ويطلع لنا "الوصفات الآمنة" بس (المسموحة طبياً وحلال ومن غير
       حساسية).
    2. ناخد هاي الوصفات الآمنة ونعيد ترتيبها بخوارزمية TOPSIS
       حسب هدف المستخدم (خسارة وزن، زيادة عضل...) — بدل الاعتماد
       على score النظام الخبير لحاله.

ملاحظة: لازم تشغّل هاد الملف وهو موجود جوا مجلد TOPSIS، وجنبه
مباشرة مجلد "Expert System" (نفس المستوى).
"""

import sys
from pathlib import Path

# ── إضافة مجلد "Expert System" لمسار البحث عشان نقدر نستورد منه ──
EXPERT_SYSTEM_DIR = Path(__file__).resolve().parent.parent / "Expert System"
sys.path.insert(0, str(EXPERT_SYSTEM_DIR))

import pandas as pd

from engine.filtering_engine import DietaryExpertSystem
from core.user_profile import UserProfile

from topsis_model import topsis_score, rank_with_topsis


if __name__ == "__main__":
    # ── 1) تحميل بيانات الوصفات ──────────────────────────────
    data_path = EXPERT_SYSTEM_DIR / "data" / "cleaned_recipes.csv"
    print(f"تحميل البيانات من: {data_path}")
    raw_df = pd.read_csv(data_path, low_memory=False)

    # ── 2) تشغيل النظام الخبير (Layer 1) ─────────────────────
    system = DietaryExpertSystem(raw_df)

    profile = UserProfile(
        age=45, height=172, weight=88, gender="male",
        conditions=["diabetes", "hypertension"],
        goal="weight_loss",
    )

    result = system.filter_recipes(profile)
    safe_recipes = result["safe_recipes"]
    print(f"\nعدد الوصفات الآمنة (بعد النظام الخبير): {len(safe_recipes)}")

    # ── 3) إعادة الترتيب بالترتيب المدمج (Layer 2) ────────────
    ranked = rank_with_topsis(safe_recipes, profile.goal)

    print("\nأفضل 10 حسب TOPSIS فقط (القديم):")
    old = ranked.sort_values("_topsis_score", ascending=False)
    cols = ["Name", "Calories", "ProteinContent", "SugarContent", "_topsis_score"]
    cols = [c for c in cols if c in old.columns]
    print(old[cols].head(10).to_string(index=False))

    print("\nأفضل 10 حسب الترتيب المدمج (الجديد):")
    cols = ["Name", "Calories", "ProteinContent", "SugarContent",
            "_topsis_score", "_ai_health_score", "_expert_score", "final_score"]
    cols = [c for c in cols if c in ranked.columns]
    print(ranked[cols].head(10).to_string(index=False))