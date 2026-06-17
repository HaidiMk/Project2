"""
main.py — نقطة الدخول الموحّدة (Expert System ثم TOPSIS)
===========================================================
التسلسل:
    1. تحميل البيانات وتشغيل النظام الخبير (Layer 1)
    2. المستخدم يدخل بياناته مرة واحدة فقط (build_profile_interactive)
    3. طباعة نتيجة النظام الخبير كما هي (print_results)
    4. تطبيق TOPSIS على نفس الوصفات الآمنة وطباعة ترتيبها (Layer 2)

شغّله من جوا مجلد TOPSIS:
    python main.py
"""

import sys
from pathlib import Path

# ── إضافة مجلد "Expert System" لمسار البحث ──────────────────
EXPERT_SYSTEM_DIR = Path(__file__).resolve().parent.parent / "Expert System"
sys.path.insert(0, str(EXPERT_SYSTEM_DIR))

from main import load_data                       # تحميل CSV (من Expert System)
from engine.filtering_engine import DietaryExpertSystem
from ui.cli_interface import build_profile_interactive, print_results

from topsis_model import topsis_score


def rank_with_topsis(df, profile):
    """رتّب الوصفات الآمنة بخوارزمية TOPSIS حسب هدف المستخدم."""
    df = df.copy()
    df["_topsis_score"] = topsis_score(df, goal=profile.goal)
    return df.sort_values("_topsis_score", ascending=False).reset_index(drop=True)


def print_topsis_results(ranked_df, top_n: int = 10):
    """طباعة نتيجة TOPSIS بشكل جدول مبسّط."""
    print("\n" + "=" * 90)
    print("  TOPSIS Ranking — Layer 2".center(88))
    print("=" * 90)

    if ranked_df.empty:
        print("  No recipes to rank.")
        return

    hdr = (f"  {'#':<3} {'Recipe Name':<45} "
           f"{'Cal':>6} {'Prot':>5} {'Sugar':>6} {'TOPSIS':>8}")
    print(hdr)
    print("  " + "-" * 86)

    for i, (_, row) in enumerate(ranked_df.head(top_n).iterrows(), 1):
        name = str(row.get("Name", "Unknown"))
        name = (name[:42] + "..") if len(name) > 44 else name.ljust(44)
        cal   = row.get("Calories", 0)
        prot  = row.get("ProteinContent", 0)
        sugar = row.get("SugarContent", 0)
        score = row.get("_topsis_score", 0)

        print(f"  {i:<3} {name} "
              f"{cal:>6.0f} {prot:>5.0f} {sugar:>6.1f} {score:>8.4f}")

    print("=" * 90)


def main():
    # ── 1) تحميل البيانات + تشغيل النظام الخبير ────────────
    df = load_data()
    system = DietaryExpertSystem(df, train_ncf=False)

    try:
        # ── 2) إدخال بيانات المستخدم (مرة واحدة فقط) ───────
        profile = build_profile_interactive()

        print("\n⏳  Generating personalized meal plan...\n")

        # ── 3) نتيجة النظام الخبير (Layer 1) ────────────────
        result = system.filter_recipes(profile)
        print_results(result, top_n=15)

        # ── 4) تطبيق TOPSIS على نفس الوصفات الآمنة (Layer 2) ─
        ranked = rank_with_topsis(result["safe_recipes"], profile)
        print_topsis_results(ranked, top_n=10)

    except ValueError as ve:
        print(f"\n❌ Validation error: {ve}")

    except KeyboardInterrupt:
        print("\n\n👋 Exited.")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()