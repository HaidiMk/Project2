import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERT_SYSTEM_DIR = PROJECT_ROOT / "Expert System"
if str(EXPERT_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERT_SYSTEM_DIR))

import pandas as pd

from Nlp.pipeline import search_recipes
from core.user_profile import UserProfile
from engine.filtering_engine import DietaryExpertSystem
from rules.medical_rules import MEDICAL_RULES

DATA_FILE = EXPERT_SYSTEM_DIR / "data" / "cleaned_recipes.csv"


def read_rule_limit(condition: str, column: str) -> float:
    op, value = MEDICAL_RULES[condition]["numeric_rules"][column]
    if op not in ("<=", ">="):
        raise ValueError(f"Unsupported rule operator: {op}")
    return float(value)


def check_sugar_rule(result: dict):
    limit = read_rule_limit("diabetes", "SugarContent")
    df = result["expert_result"]["safe_recipes"]
    if df.empty or "SugarContent" not in df.columns:
        return True, df, ""
    viol = df[df["SugarContent"] > limit]
    return len(viol) == 0, viol, f"SugarContent above {limit:g} g"


def check_peanut_allergy(result: dict):
    df = result["expert_result"]["safe_recipes"]
    if df.empty:
        return True, df, ""
    ing_mask = (
        df["IngredientsList"].fillna("").astype(str).str.lower()
        .str.contains("peanut", na=False)
        if "IngredientsList" in df.columns
        else pd.Series(False, index=df.index)
    )
    name_mask = (
        df["Name"].fillna("").astype(str).str.lower()
        .str.contains("peanut", na=False)
        if "Name" in df.columns
        else pd.Series(False, index=df.index)
    )
    viol = df[ing_mask | name_mask]
    return len(viol) == 0, viol, "contains 'peanut' in name or ingredients"


def check_sodium_rule(result: dict):
    limit = read_rule_limit("hypertension", "SodiumContent")
    df = result["expert_result"]["safe_recipes"]
    if df.empty or "SodiumContent" not in df.columns:
        return True, df, ""
    viol = df[df["SodiumContent"] > limit]
    return len(viol) == 0, viol, f"SodiumContent above {limit:g} mg"


def check_control_nonempty(result: dict):
    ok = result["total_after_expert"] > 0 and result["total_safe"] > 0
    return ok, result["expert_result"]["safe_recipes"], "expected a non-empty result"


def run_scenario(system: DietaryExpertSystem, name: str, profile: UserProfile,
                 query: str, check, row_fmt) -> bool:
    print(f"\n{'=' * 74}\n{name}")
    print(f"   query  : {query!r}")
    print(f"   profile: conditions={profile.conditions} "
          f"allergies={profile.allergies} meal_type={profile.meal_type}")

    t0 = time.time()
    result = search_recipes(query, profile, system, top_n=10)
    elapsed = time.time() - t0

    expert_n = result["total_after_expert"]
    refined_n = result["total_safe"]
    print(f"   expert-safe: {expert_n:,} | after query refinement: {refined_n:,} "
          f"| ({elapsed:.1f}s)")

    ok, viol, why = check(result)
    if ok:
        print("   == PASS == (no unsafe recipe leaked through)")
        return True

    print("   == FAIL ==")
    print(f"   {len(viol):,} unsafe recipe(s) leaked through: {why}")
    for _, row in viol.head(10).iterrows():
        print(f"     - {str(row.get('Name', ''))[:64]}  {row_fmt(row)}")
    return False


def main() -> None:
    t0 = time.time()
    print(f"Loading recipe database from {DATA_FILE.name} "
          f"({DATA_FILE.stat().st_size / 1e6:.0f} MB)...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"Loaded {len(df):,} recipes in {time.time() - t0:.1f}s")

    print("\nConstructing DietaryExpertSystem ONCE (reused by all scenarios)...")
    t0 = time.time()
    system = DietaryExpertSystem(df)
    del df
    print(f"Expert System ready in {time.time() - t0:.1f}s")

    diabetes_sugar = read_rule_limit("diabetes", "SugarContent")
    hypertension_sodium = read_rule_limit("hypertension", "SodiumContent")
    print(f"\nMedical rules in effect for this run: diabetes SugarContent "
          f"<= {diabetes_sugar:g} g | hypertension SodiumContent "
          f"<= {hypertension_sodium:g} mg")

    scenarios = [
        {
            "name": "Scenario 1: Diabetic user asks for a high-sugar dessert",
            "profile": UserProfile(age=45, height=172.0, weight=88.0,
                                   gender="male", conditions=["diabetes"],
                                   goal="weight_loss"),
            "query": "chocolate cake with lots of sugar for dinner",
            "check": check_sugar_rule,
            "row_fmt": lambda r: f"sugar={r['SugarContent']} g",
        },
        {
            "name": "Scenario 2: Peanut-allergic user asks for a peanut dish",
            "profile": UserProfile(age=30, height=165.0, weight=60.0,
                                   gender="female", allergies=["peanuts"]),
            "query": "peanut butter smoothie for breakfast",
            "check": check_peanut_allergy,
            "row_fmt": lambda r: (
                f"HasNuts={r.get('HasNuts', '?')} | "
                + str(r.get("IngredientsList", ""))[:80]
            ),
        },
        {
            "name": "Scenario 3: Hypertension user asks for a salty dish",
            "profile": UserProfile(age=50, height=175.0, weight=85.0,
                                   gender="male", conditions=["hypertension"]),
            "query": "salty bacon and pickles for lunch",
            "check": check_sodium_rule,
            "row_fmt": lambda r: f"sodium={r['SodiumContent']} mg",
        },
        {
            "name": "Scenario 4 (control): same diabetic profile, safe query",
            "profile": UserProfile(age=45, height=172.0, weight=88.0,
                                   gender="male", conditions=["diabetes"],
                                   goal="weight_loss"),
            "query": "low sugar dinner for diabetic",
            "check": check_control_nonempty,
            "row_fmt": lambda r: "",
        },
    ]

    passed = 0
    total = len(scenarios)
    for scenario in scenarios:
        ok = run_scenario(
            system,
            scenario["name"],
            scenario["profile"],
            scenario["query"],
            scenario["check"],
            scenario["row_fmt"],
        )
        passed += int(ok)

    print(f"\n{'=' * 74}")
    if passed == total:
        print(f"{passed}/{total} scenarios passed — medical filtering holds "
              f"even when the query text explicitly requests something unsafe.")
    else:
        print(f"{passed}/{total} scenarios passed, {total - passed} FAILED — "
              f"review the leaked recipes above.")


if __name__ == "__main__":
    main()
