"""
main.py — Smart Dietary Advisor v4.0
=====================================
نقطة الدخول الرئيسية للنظام.
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd

from core.constants import COLUMN_NAMES
from engine.filtering_engine import DietaryExpertSystem
from ui.cli_interface import (
    build_profile_interactive,
    create_profile,
    print_results,
)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

RECIPES_FILE = DATA_DIR / "cleaned_recipes.csv"
NCF_MODEL_FILE = DATA_DIR / "ncf_model.pt"


def load_data(path=RECIPES_FILE) -> pd.DataFrame:
    """تحميل قاعدة بيانات الوصفات مع معالجة أخطاء الترميز."""

    print(f"\n📥  Loading recipe database from: {path}")

    for encoding in ["utf-8", "latin-1", "ISO-8859-1"]:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=encoding)
            print(f"✅  Loaded {len(df):,} recipes (encoding: {encoding})")
            return df

        except FileNotFoundError:
            print(f"❌  File not found: {path}")
            sys.exit(1)

        except Exception:
            continue

    try:
        df = pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            header=None,
            names=COLUMN_NAMES,
        )

        print(f"✅  Loaded {len(df):,} recipes (python engine fallback)")
        return df

    except Exception as e:
        print(f"❌  Could not load file: {e}")
        sys.exit(1)


def save_results(result: dict, fmt: str = "csv"):
    """حفظ الوصفات الآمنة إلى CSV أو JSON"""

    df = result["safe_recipes"]

    if df.empty:
        print("⚠️  No recipes to save.")
        return

    cols = [
        "Name",
        "Calories",
        "ProteinContent",
        "CarbohydrateContent",
        "FatContent",
        "SodiumContent",
        "FiberContent",
        "SugarContent",
        "Rating",
        "_reason",
    ]

    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if fmt == "json":
        path = OUTPUT_DIR / "safe_recipes.json"
        df_out.to_json(
            path,
            orient="records",
            indent=2,
            force_ascii=False,
        )
    else:
        path = OUTPUT_DIR / "safe_recipes.csv"
        df_out.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

    print(f"\n✅  Saved {len(df_out):,} recipes to: {path}")


def run_demo(system: DietaryExpertSystem):
    """مثال برمجي"""

    print(
        "\n🔬  Running demo: "
        "Diabetic + Hypertension patient, weight loss goal\n"
    )

    profile = create_profile(
        age=45,
        height=172,
        weight=88,
        gender="male",
        conditions=["diabetes", "hypertension"],
        goal="weight_loss",
    )

    result = system.filter_recipes(profile)
    print_results(result, top_n=10)


def main():
    df = load_data()

    # فحص وجود نموذج NCF
    ncf_exists = NCF_MODEL_FILE.exists()

    system = DietaryExpertSystem(
        df,
        train_ncf=not ncf_exists,
    )

    if "--demo" in sys.argv:
        run_demo(system)
        return

    try:
        user = build_profile_interactive()

        print("\n⏳  Generating personalized meal plan...\n")

        result = system.filter_recipes(user)

        print_results(result, top_n=15)

        print("\nSave results?")
        print("1. Save as CSV")
        print("2. Save as JSON")
        print("3. No thanks")

        choice = input("\nSelect (1-3): ").strip()

        if choice == "1":
            save_results(result, fmt="csv")

        elif choice == "2":
            save_results(result, fmt="json")

        else:
            print("\nOK. Results not saved.")

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