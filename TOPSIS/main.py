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

from topsis_model import topsis_score, rank_with_topsis


def print_old_vs_new(ranked_df, top_n: int = 5):
    """قارن ترتيب TOPSIS القديم (وحده) مع الترتيب المدمج الجديد (3 مصادر)."""
    print("\n" + "=" * 90)
    print("  Old (TOPSIS-only) vs New (blended) ranking".center(88))
    print("=" * 90)

    if ranked_df.empty:
        print("  No recipes to rank.")
        return

    old = ranked_df.sort_values("_topsis_score", ascending=False).head(top_n)
    new = ranked_df.head(top_n)
    new_pos = {idx: i for i, idx in enumerate(new.index)}

    hdr = f"  {'Old':<3} {'New':<3} {'Recipe Name':<44} {'TOPSIS':>7} {'Final':>7}"
    print(hdr)
    print("  " + "-" * 86)

    for old_pos, (idx, row) in enumerate(old.iterrows(), 1):
        npos = new_pos.get(idx, "-")
        name = str(row.get("Name", "Unknown"))
        name = (name[:42] + "..") if len(name) > 44 else name.ljust(44)
        print(f"  {old_pos:<3} {str(npos):<3} {name} "
              f"{row.get('_topsis_score', 0):>7.4f} {row.get('final_score', 0):>7.4f}")

    print("=" * 90)


def print_topsis_results(ranked_df, top_n: int = 10):
    """طباعة نتيجة TOPSIS المدمجة (Layer 2) — مع مكوّنات الدرجة الثلاثة."""
    print("\n" + "=" * 108)
    print("  TOPSIS Ranking — Layer 2 (blended: 0.4*TOPSIS + 0.4*AI + 0.2*Expert)".center(106))
    print("=" * 108)

    if ranked_df.empty:
        print("  No recipes to rank.")
        return

    hdr = (f"  {'#':<3} {'Recipe Name':<40} "
           f"{'Cal':>6} {'Prot':>5} {'Sugar':>6} {'TOPSIS':>7} {'AI':>6} {'Exp*':>6} {'Final':>7}")
    print(hdr)
    print("  " + "-" * 104)

    for i, (_, row) in enumerate(ranked_df.head(top_n).iterrows(), 1):
        name = str(row.get("Name", "Unknown"))
        name = (name[:38] + "..") if len(name) > 40 else name.ljust(40)
        cal   = row.get("Calories", 0)
        prot  = row.get("ProteinContent", 0)
        sugar = row.get("SugarContent", 0)
        tops  = row.get("_topsis_score", 0)
        ai    = row.get("_ai_health_score", 0)
        exp   = row.get("_expert_score", 0)
        final = row.get("final_score", 0)

        print(f"  {i:<3} {name} "
              f"{cal:>6.0f} {prot:>5.0f} {sugar:>6.1f} {tops:>7.4f} "
              f"{ai:>6.3f} {exp:>6.3f} {final:>7.4f}")

    print("=" * 108)


def main():
    # ── 1) تحميل البيانات + تشغيل النظام الخبير ────────────
    df = load_data()
    system = DietaryExpertSystem(df)

    try:
        # ── 2) إدخال بيانات المستخدم (مرة واحدة فقط) ───────
        profile = build_profile_interactive()

        print("\n⏳  Generating personalized meal plan...\n")

        # ── 3) نتيجة النظام الخبير (Layer 1) ────────────────
        result = system.filter_recipes(profile)
        print_results(result, top_n=15)

        # ── 4) تطبيق الترتيب المدمج على نفس الوصفات الآمنة (Layer 2) ─
        ranked = rank_with_topsis(result["safe_recipes"], profile.goal)
        print_old_vs_new(ranked, top_n=5)
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