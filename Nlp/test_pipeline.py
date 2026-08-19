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

DATA_FILE = EXPERT_SYSTEM_DIR / "data" / "cleaned_recipes.csv"

QUERIES = [
    "I want low sugar dinner for diabetic",
    "breakfast with 20 grams protein and no dairy",
    "high protein chicken lunch, low sodium",
]


def show_recipes(result: dict, n: int = 20) -> None:
    print(f"   expert returned {result['total_after_expert']} safe recipes "
          f"| after query refinement: {result['total_safe']}")
    print(f"   AI health scoring active: {result['ai_health_scoring']}")
    print("   top recipes:")
    for i, r in enumerate(result["top_recipes"][:n], 1):
        name = str(r.get("Name", ""))[:68]
        print(
            f"     {i}. {name:<70} kcal={r.get('Calories')} "
            f"prot={r.get('ProteinContent')} sugar={r.get('SugarContent')} "
            f"sodium={r.get('SodiumContent')} fiber={r.get('FiberContent')}"
        )


def main() -> None:
    t0 = time.time()
    print(f"Loading recipe database from {DATA_FILE.name} "
          f"({DATA_FILE.stat().st_size / 1e6:.0f} MB)...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"Loaded {len(df):,} recipes in {time.time() - t0:.1f}s")

    print("\nConstructing DietaryExpertSystem...")
    t0 = time.time()
    system = DietaryExpertSystem(df)
    del df 
    print(f"Expert System ready in {time.time() - t0:.1f}s")

    base_profile = UserProfile(
        age=32, height=170.0, weight=70.0, gender="male",
        activity_level="light", goal="weight_loss",
        allergies=["peanuts"],
    )
    print(f"\nBase profile: {base_profile.summary()}")

    for query in QUERIES:
        print(f"\n{'=' * 72}\nQUERY: {query!r}")
        t0 = time.time()
        result = search_recipes(query, base_profile, system, top_n=20)
        elapsed = time.time() - t0

        merged = result["merged_profile"]
        print(f"   parsed filters : {result['filters']}")
        print(f"   merged profile : conditions={merged.conditions} "
              f"allergies={merged.allergies} preferences={merged.preferences} "
              f"meal_type={merged.meal_type}")
        show_recipes(result)
        print(f"   (search took {elapsed:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
