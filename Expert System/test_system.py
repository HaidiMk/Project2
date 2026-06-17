"""
test_system.py — اختبار شامل للنظام
=====================================
شغّل بـ: python test_system.py
"""

import sys
import pandas as pd
from engine.filtering_engine import DietaryExpertSystem
from ui.cli_interface import create_profile

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(test_name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((test_name, status, detail))
    icon = "✅" if condition else "❌"
    print(f"  {icon} {status} — {test_name}")
    if not condition and detail:
        print(f"         {detail}")

# ── تحميل البيانات ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  Smart Dietary Advisor — System Test")
print("=" * 60)

try:
    df = pd.read_csv("data/cleaned_recipes.csv", low_memory=False)
    system = DietaryExpertSystem(df)
    check("Load database", True, f"{len(df):,} recipes")
except Exception as e:
    print(f"\n❌ Cannot load database: {e}")
    sys.exit(1)

# ══ اختبار 1: السكري ════════════════════════════════════════
print("\n── Test 1: Diabetes ──")
p = create_profile(age=45, height=170, weight=80, gender="male",
                   conditions=["diabetes"], goal="weight_loss")
r = system.filter_recipes(p)
df_safe = r["safe_recipes"]

check("Diabetes — recipes found",
      r["total_safe"] > 0)
check("Diabetes — sugar <= 15g",
      df_safe["SugarContent"].max() <= 15 if "SugarContent" in df_safe.columns else True,
      f"Max sugar: {df_safe['SugarContent'].max():.1f}g")
check("Diabetes — carbs <= 60g",
      df_safe["CarbohydrateContent"].max() <= 60 if "CarbohydrateContent" in df_safe.columns else True,
      f"Max carbs: {df_safe['CarbohydrateContent'].max():.1f}g")

# ══ اختبار 2: ضغط الدم ══════════════════════════════════════
print("\n── Test 2: Hypertension ──")
p = create_profile(age=50, height=165, weight=75, gender="female",
                   conditions=["hypertension"], goal="heart_health")
r = system.filter_recipes(p)
df_safe = r["safe_recipes"]

check("Hypertension — recipes found",
      r["total_safe"] > 0)
check("Hypertension — sodium <= 600mg",
      df_safe["SodiumContent"].max() <= 600 if "SodiumContent" in df_safe.columns else True,
      f"Max sodium: {df_safe['SodiumContent'].max():.1f}mg")

## ══ اختبار 3: الحلال ════════════════════════════════════════
print("\n── Test 3: Halal Filter ──")
p = create_profile(age=30, height=175, weight=70, gender="male")
r = system.filter_recipes(p)
df_safe = r["safe_recipes"]

halal_violations = 0
if "Name" in df_safe.columns:
    # نختبر بس الكلمات الواضحة بدون wine/beer لأنها ممكن تكون في أسماء حلال
    for word in ["pork", "bacon", "lard", "pepperoni",
                 "salami", "prosciutto", "vodka", "whiskey"]:
        count = df_safe["Name"].fillna("").str.lower().str.contains(
            r"\b" + word + r"\b", regex=True).sum()
        halal_violations += count

check("Halal — no clear haram ingredients in recipe names",
      halal_violations == 0,
      f"Violations found: {halal_violations}")
# ══ اختبار 4: PCOS للإناث فقط ═══════════════════════════════
print("\n── Test 4: PCOS Females Only ──")
from core.constants import DISEASES_BY_STAGE, DISEASES_FEMALE_ONLY

check("PCOS not in adult diseases list",
      "pcos" not in DISEASES_BY_STAGE["adult"])
check("PCOS in female-only list",
      "pcos" in DISEASES_FEMALE_ONLY.get("adult", []))

# ══ اختبار 5: التعارضات المستحيلة ═══════════════════════════
print("\n── Test 5: Impossible Combinations ──")
from ui.cli_interface import IMPOSSIBLE_COMBOS

check("Impossible combos defined",
      len(IMPOSSIBLE_COMBOS) >= 3)
check("obesity+underweight is impossible",
      any({"obesity", "underweight"}.issubset(c[0]) for c in IMPOSSIBLE_COMBOS))
check("hypothyroidism+hyperthyroidism is impossible",
      any({"hypothyroidism", "hyperthyroidism"}.issubset(c[0]) for c in IMPOSSIBLE_COMBOS))

# ══ اختبار 6: النباتي ═══════════════════════════════════════
print("\n── Test 6: Vegetarian Filter ──")
p = create_profile(age=25, height=165, weight=60, gender="female",
                   preferences=["vegetarian"])
r = system.filter_recipes(p)
df_safe = r["safe_recipes"]

meat_violations = 0
if "Name" in df_safe.columns:
    for word in ["chicken", "beef", "pork", "salmon", "tuna",
                 "shrimp", "halibut", "pollock", "venison", "lamb"]:
        count = df_safe["Name"].fillna("").str.lower().str.contains(
            r"\b" + word + r"\b", regex=True).sum()
        meat_violations += count

check("Vegetarian — no meat/fish in recipe names",
      meat_violations == 0,
      f"Violations found: {meat_violations}")

# ══ اختبار 7: الحمل + السكري ════════════════════════════════
print("\n── Test 7: Pregnancy + Diabetes Combined Rules ──")
from engine.rule_builder import get_applicable_rules
from ui.cli_interface import create_profile

p = create_profile(age=30, height=165, weight=70, gender="female",
                   conditions=["diabetes"], pregnant=True)
rules = get_applicable_rules(p)

check("Pregnancy+Diabetes — combined rules activated",
      len(rules["conflict_messages"]) > 0 or
      "Calories" in rules["numeric_rules"],
      "Combined rules applied")
check("Pregnancy+Diabetes — protein >= 20g required",
      rules["numeric_rules"].get("ProteinContent", ("", 0))[1] >= 20 or
      rules["min_requirements"].get("ProteinContent", ("", 0))[1] >= 20)

# ══ اختبار 8: النشاط البدني يؤثر على السعرات ════════════════
print("\n── Test 8: Activity Level Affects Calories ──")
from core.user_profile import UserProfile

p_sedentary = UserProfile(age=30, height=170, weight=70,
                           gender="male", activity_level="sedentary")
p_active    = UserProfile(age=30, height=170, weight=70,
                           gender="male", activity_level="very_active")

check("Activity level affects daily calories",
      p_active.daily_calories > p_sedentary.daily_calories,
      f"Sedentary: {p_sedentary.daily_calories} | Active: {p_active.daily_calories}")

# ══ النتيجة النهائية ═════════════════════════════════════════
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
total  = len(results)

print(f"  Results: {passed}/{total} passed")
if failed > 0:
    print(f"\n  Failed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"     ❌ {name} — {detail}")
else:
    print("  ✅ All tests passed!")
print("=" * 60)