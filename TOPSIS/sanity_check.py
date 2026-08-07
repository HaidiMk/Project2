"""
sanity_check.py — تحقق شامل من تكامل طبقة الصحة الذكية (AI health)
====================================================================
يفحص:
    1. 3 ملفات تعريف (سكري/زيادة عضل/مسنّ) — دون أخطاء
    2. أعمدة الدرجات _expert_score و _ai_health_score موجودة وبلا NaN
    3. الترتيب القديم (TOPSIS فقط) مقابل الجديد (0.4*TOPSIS + 0.4*AI
       + 0.2*expert_normalized) — وأفضل 5 متداخلة جزئياً
    4. لا عودة لوصفات غير آمنة: الترتيب المدمج يعمل على الوصفات
       الآمنة فقط حصراً

التشغيل (من داخل مجلد TOPSIS):
    python sanity_check.py
"""

import sys
from pathlib import Path

# ── إضافة مجلد "Expert System" لمسار البحث ──────────────────
EXPERT_SYSTEM_DIR = Path(__file__).resolve().parent.parent / "Expert System"
sys.path.insert(0, str(EXPERT_SYSTEM_DIR))

import numpy as np
import pandas as pd

from main import load_data
from engine.filtering_engine import DietaryExpertSystem
from core.user_profile import UserProfile

from topsis_model import rank_with_topsis


PROFILES = [
    ("diabetic", UserProfile(
        age=45, height=172, weight=88, gender="male",
        conditions=["diabetes", "hypertension"], goal="weight_loss",
    )),
    ("muscle_gain", UserProfile(
        age=28, height=180, weight=75, gender="male",
        conditions=[], goal="muscle_gain",
    )),
    ("elderly", UserProfile(
        age=70, height=165, weight=70, gender="female",
        conditions=["hypertension", "osteoporosis"], goal="heart_health",
    )),
]


def check_profile(name, profile, system, all_ok):
    print(f"\n--- Profile: {name} ---")
    result = system.filter_recipes(profile)
    safe = result["safe_recipes"]
    print(f"  safe={result['total_safe']} | original={result['total_original']}")

    required = ["_expert_score", "_ai_health_score"]
    missing = [c for c in required if c not in safe.columns]
    if missing:
        print(f"  FAIL: missing columns {missing}")
        return False

    for col in required:
        vals = safe[col].astype(float)
        n_nan = int(vals.isna().sum())
        print(f"  {col}: n={len(vals)}, NaN={n_nan}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")
        if n_nan:
            all_ok = False

    ranked = rank_with_topsis(safe, profile.goal)
    if np.isnan(ranked["final_score"].astype(float)).any():
        print("  FAIL: NaN in final_score")
        all_ok = False

    in_bounds = bool(((ranked["final_score"].astype(float) >= -1e-6)
                      & (ranked["final_score"].astype(float) <= 1 + 1e-6)).all())
    print(f"  final_score within [0,1]: {in_bounds}")
    if not in_bounds:
        all_ok = False

    old_top = ranked.sort_values("_topsis_score", ascending=False).head(5)
    new_top = ranked.head(5)

    print("  Old TOPSIS-only top-5:")
    for i, (_, r) in enumerate(old_top.iterrows(), 1):
        nm = str(r.get("Name", "Unknown"))
        nm = (nm[:43] + "..") if len(nm) > 45 else nm.ljust(45)
        print(f"    {i}. {nm} topsis={r['_topsis_score']:.4f}")

    print("  New blended top-5:")
    for i, (_, r) in enumerate(new_top.iterrows(), 1):
        nm = str(r.get("Name", "Unknown"))
        nm = (nm[:43] + "..") if len(nm) > 45 else nm.ljust(45)
        print(f"    {i}. {nm} final={r['final_score']:.4f} "
              f"(topsis={r['_topsis_score']:.4f}, ai={r['_ai_health_score']:.4f})")

    overlap = len(set(old_top.index).intersection(set(new_top.index)))
    print(f"  Overlap old/new top-5: {overlap}/5")

    all_inside = set(ranked.index).issubset(set(safe.index))
    print(f"  Blended ranking within safe set only: {all_inside}")
    if not all_inside:
        all_ok = False

    return all_ok


def main():
    print("=" * 100)
    print("SANITY CHECK — AI health score integration (Layer 1.5 + blended Layer 2)")
    print("=" * 100)

    df = load_data()
    system = DietaryExpertSystem(df)

    all_ok = True
    for name, profile in PROFILES:
        all_ok = check_profile(name, profile, system, all_ok) and all_ok

    print("\n" + "=" * 100)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "ISSUES FOUND")
    print("=" * 100)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
