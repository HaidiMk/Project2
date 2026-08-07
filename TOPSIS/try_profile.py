"""
try_profile.py — أداة اختبار يدوية تفاعلية لمسار التوصية الكامل
================================================================
تأخذ ملف تعريف مستخدم (عمر/طول/وزن/جنس/حالات/هدف) عبر input()،
وتمرّره عبر المسار الكامل:
    DietaryExpertSystem.filter_recipes()  →  rank_with_topsis()
وتعرض ملخصاً نظيفاً: نسبة الوصفات الآمنة + أفضل 10 بدرجاتها
(TOPSIS / AI health / Expert / Final) + ملاحظة عن وضع درجة
الصحة الذكية (حالات المستخدم المدرّبة أم الـ fallback الشامل).

التشغيل — من داخل مجلد TOPSIS/ (وليس من خارجها):
    python try_profile.py            # تفاعلي
    python try_profile.py --help     # تعليمات فقط

ملاحظة مهمة على Windows: اِضبط ترميز الإخراج أولاً وإلا تتعطل
الطباعة عند أي حرف عربي:
    $env:PYTHONIOENCODING='utf-8'; python try_profile.py
"""

import html
import json
import sys
from pathlib import Path

EXPERT_SYSTEM_DIR = Path(__file__).resolve().parent.parent / "Expert System"
sys.path.insert(0, str(EXPERT_SYSTEM_DIR))

from main import load_data
from engine.filtering_engine import DietaryExpertSystem
from engine.rule_builder import get_applicable_condition_keys
from ui.cli_interface import create_profile
from rules.medical_rules import MEDICAL_RULES
from rules.goals_and_preferences import GOAL_VECTORS
from topsis_model import rank_with_topsis


VALID_CONDITIONS = sorted(MEDICAL_RULES.keys())
VALID_GOALS = sorted(GOAL_VECTORS.keys())


def usage():
    print("=" * 78)
    print("  try_profile.py — manual pipeline tester".center(76))
    print("=" * 78)
    print("""
  Run from INSIDE the TOPSIS/ directory (it needs the sibling
  'Expert System' folder on sys.path):

      python try_profile.py            interactive test
      python try_profile.py --help     this help, then exit

  On Windows, set the console encoding first, otherwise Arabic
  output may crash the print:

      $env:PYTHONIOENCODING='utf-8'; python try_profile.py

  Valid medical condition keys:
""" + "      " + ", ".join(VALID_CONDITIONS) + """

  Valid goal keys (or empty for no goal):
""" + "      " + ", ".join(VALID_GOALS) + """

  Press Ctrl+C at any prompt to abort.
""")


def _ask_int(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  Error: '{raw}' is not a number.")
            continue
        if low <= v <= high:
            return v
        print(f"  Error: must be between {low} and {high}.")


def _ask_float(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            v = float(raw)
        except ValueError:
            print(f"  Error: '{raw}' is not a number.")
            continue
        if low <= v <= high:
            return v
        print(f"  Error: must be between {low} and {high}.")


def _ask_gender():
    while True:
        raw = input("  Gender (male/female): ").strip().lower()
        if raw in ("male", "m"):
            return "male"
        if raw in ("female", "f"):
            return "female"
        print("  Error: enter 'male' or 'female'.")


def _ask_conditions():
    while True:
        raw = input(
            "\n  Medical conditions (comma-separated, or ENTER for none)\n"
            f"  Valid keys: {', '.join(VALID_CONDITIONS)}\n"
            "  Conditions: "
        ).strip()
        if not raw:
            return []
        keys = [k.strip().lower() for k in raw.replace("،", ",").split(",")]
        keys = [k for k in keys if k]
        unknown = [k for k in keys if k not in VALID_CONDITIONS]
        if unknown:
            print(f"  Error: unknown condition key(s): {', '.join(unknown)}")
            continue
        return keys


def _ask_taste():
    raw = input(
        "\n  Food preferences — separate liked and disliked with ';' or 'but',\n"
        "  and start the disliked part with 'dislike', 'hate', or 'avoid'\n"
        "  (e.g. \"garlic, olive oil, chicken; dislike seafood\" — or ENTER to skip): "
    ).strip()
    return raw or None


def _ask_goal():
    while True:
        raw = input(
            "\n  Goal (or ENTER for none)\n"
            f"  Valid keys: {', '.join(VALID_GOALS)}\n"
            "  Goal: "
        ).strip().lower()
        if not raw:
            return None
        if raw in GOAL_VECTORS:
            return raw
        print(f"  Error: unknown goal key '{raw}'.")


def ai_mode_note(profile) -> str:
    """هل دُعمت درجة الصحة الذكية بحالات المستخدم أم بالـ fallback الشامل؟"""
    labels = json.loads(
        (EXPERT_SYSTEM_DIR / "data" / "health_classifier_labels.json")
        .read_text(encoding="utf-8")
    )
    trained = set(labels)
    keys = get_applicable_condition_keys(profile)
    matched = [k for k in keys if f"label_{k}" in trained]
    if matched:
        return (
            f"condition-specific ({len(matched)} trained: "
            f"{', '.join(matched)})"
        )
    return "fallback (all 22 conditions — none of the user's conditions are among the trained labels)"


def build_profile():
    print("\n" + "=" * 78)
    print("  Custom profile (manual test)".center(76))
    print("=" * 78)
    age = _ask_int("  Age (years): ", 1, 110)
    height = _ask_float("  Height (cm): ", 50, 250)
    weight = _ask_float("  Weight (kg): ", 2, 400)
    gender = _ask_gender()
    conditions = _ask_conditions()
    goal = _ask_goal()
    return create_profile(
        age=age, height=height, weight=weight, gender=gender,
        conditions=conditions, goal=goal,
    )


def run():
    profile = build_profile()

    print("\n  Loading recipe database...")
    df = load_data()
    system = DietaryExpertSystem(df)
    result = system.filter_recipes(profile)

    safe = result["safe_recipes"]
    taste_text = _ask_taste()
    ranked = rank_with_topsis(safe, profile.goal, user_taste_text=taste_text)

    ps = result["profile_summary"]
    print(f"\n  Age: {ps['age']} | Gender: {ps['gender']} | "
          f"BMI: {ps['bmi']} ({ps['bmi_category']})")
    print(f"  Conditions: {', '.join(profile.conditions) or 'none'}"
          f"  |  Goal: {profile.goal or 'none'}")
    print(f"  Safe recipes: {result['total_safe']:,} / "
          f"{result['total_original']:,} ({result['filter_rate']}%)")
    print(f"  AI health score mode: {ai_mode_note(profile)}")

    has_taste = "_taste_score" in ranked.columns
    blend_txt = ("0.35*TOPSIS + 0.35*AI + 0.15*expert_norm + 0.15*taste"
                 if has_taste else "0.4*TOPSIS + 0.4*AI + 0.2*expert_norm")
    print(f"\n  Top 10 — final blended ranking (final = {blend_txt}):")
    print("  " + "-" * (118 if has_taste else 110))
    hdr = (f"  {'#':<3} {'Recipe Name':<40} {'Cal':>7} {'Prot':>6} "
           f"{'Sugar':>7} {'TOPSIS':>8} {'AI':>6} {'Exp*':>6}")
    if has_taste:
        hdr += f" {'Taste':>6}"
    hdr += f" {'Final':>7}"
    print(hdr)
    print("  " + "-" * (118 if has_taste else 110))

    for i, (_, row) in enumerate(ranked.head(10).iterrows(), 1):
        raw = html.unescape(str(row.get("Name", "Unknown")))
        name = (raw[:38] + "..") if len(raw) > 40 else raw.ljust(40)
        line = (f"  {i:<3} {name} "
                f"{row.get('Calories', 0):>7.0f} "
                f"{row.get('ProteinContent', 0):>6.0f} "
                f"{row.get('SugarContent', 0):>7.1f} "
                f"{row.get('_topsis_score', 0):>8.4f} "
                f"{row.get('_ai_health_score', 0):>6.3f} "
                f"{row.get('_expert_score', 0):>6.3f}")
        if has_taste:
            line += f" {row.get('_taste_score', 0):>6.3f}"
        line += f" {row.get('final_score', 0):>7.4f}"
        print(line)

    print("  " + "-" * (118 if has_taste else 110))
    print("  * Exp = raw expert score (min-max normalized inside final)")
    print("=" * 78)


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        usage()
        return
    usage()
    print("  Starting interactive session...")
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Aborted.")
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
