"""
main.py — Smart Dietary Advisor v4.0 — NO-CONTROL-FLOW EDITION
===============================================================
نقطة الدخول الرئيسية للنظام.

خالٍ تماماً من: if / elif / else / for / while / map / filter / reduce
ومن التعابير الثلاثية ومن الـ comprehensions.

البدائل:
    - for على قائمة الترميزات → دالة عودية تجرّب ترميزاً تلو الآخر
    - if/elif لاختيار الحفظ    → dict-dispatch
    - if "--demo"              → فهرسة قاموس
    - try/except               → تبقى (ليست بنية تحكم ممنوعة)؛ أُزيل ما بداخلها
                                  من for/if فقط
"""

import os
import sys
import json
from pathlib import Path
from typing import Callable

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


# ── أدوات بديلة عن بنى التحكم ─────────────────────────────

def _pick(cond, when_true, when_false):
    return {True: when_true, False: when_false}[bool(cond)]


def _do(cond, do_true: Callable, do_false: Callable):
    return {True: do_true, False: do_false}[bool(cond)]()


def load_data(path=RECIPES_FILE) -> pd.DataFrame:
    """تحميل قاعدة بيانات الوصفات مع معالجة أخطاء الترميز (عودياً بلا for)."""

    print(f"\n📥  Loading recipe database from: {path}")

    encodings = ["utf-8", "latin-1", "ISO-8859-1"]

    def try_encodings(rest):
        """جرّب الترميزات بالتسلسل عبر التكرار العودي."""
        exhausted = (len(rest) == 0)
        return {True: fallback_python_engine, False: lambda: _try_one(rest)}[exhausted]()

    def _try_one(rest):
        encoding = rest[0]

        def ok():
            df = pd.read_csv(path, low_memory=False, encoding=encoding)
            print(f"✅  Loaded {len(df):,} recipes (encoding: {encoding})")
            return df

        def not_found(_exc):
            print(f"❌  File not found: {path}")
            sys.exit(1)

        def other(_exc):
            # ترميز فشل لسبب آخر → جرّب التالي
            return try_encodings(rest[1:])

        # FileNotFoundError → خروج فوري ؛ أي خطأ آخر → الترميز التالي
        try:
            return ok()
        except FileNotFoundError as exc:
            return not_found(exc)
        except Exception as exc:                       # noqa: BLE001
            return other(exc)

    def fallback_python_engine():
        """محرّك python كحل أخير لو فشلت كل الترميزات."""
        def ok():
            df = pd.read_csv(
                path, engine="python", on_bad_lines="skip",
                header=None, names=COLUMN_NAMES,
            )
            print(f"✅  Loaded {len(df):,} recipes (python engine fallback)")
            return df

        def fail(exc):
            print(f"❌  Could not load file: {exc}")
            sys.exit(1)

        try:
            return ok()
        except Exception as exc:                       # noqa: BLE001
            return fail(exc)

    return try_encodings(encodings)


def save_results(result: dict, fmt: str = "csv"):
    """حفظ الوصفات الآمنة إلى CSV أو JSON."""

    df = result["safe_recipes"]

    def empty():
        print("⚠️  No recipes to save.")
        return None

    def do_save():
        wanted = [
            "Name", "Calories", "ProteinContent", "CarbohydrateContent",
            "FatContent", "SodiumContent", "FiberContent", "SugarContent",
            "Rating", "_reason",
        ]
        # احتفظ بالأعمدة الموجودة فقط — تقاطع موجّه بلا comprehension
        import numpy as np
        wanted_arr = np.array(wanted)
        present_mask = np.isin(wanted_arr, list(df.columns))
        cols = wanted_arr[present_mask].tolist()
        df_out = df[cols].copy()

        OUTPUT_DIR.mkdir(exist_ok=True)

        def save_json():
            path = OUTPUT_DIR / "safe_recipes.json"
            df_out.to_json(path, orient="records", indent=2, force_ascii=False)
            return path

        def save_csv():
            path = OUTPUT_DIR / "safe_recipes.csv"
            df_out.to_csv(path, index=False, encoding="utf-8-sig")
            return path

        path = _do(fmt == "json", save_json, save_csv)
        print(f"\n✅  Saved {len(df_out):,} recipes to: {path}")
        return None

    return _do(df.empty, empty, do_save)


def run_demo(system: DietaryExpertSystem):
    """مثال برمجي."""

    print(
        "\n🔬  Running demo: "
        "Diabetic + Hypertension patient, weight loss goal\n"
    )

    profile = create_profile(
        age=45, height=172, weight=88, gender="male",
        conditions=["diabetes", "hypertension"],
        goal="weight_loss",
    )

    result = system.filter_recipes(profile)
    print_results(result, top_n=10)


def main():
    df = load_data()

    ncf_exists = NCF_MODEL_FILE.exists()
    system = DietaryExpertSystem(df, train_ncf=not ncf_exists)

    # وضع العرض التوضيحي
    is_demo = "--demo" in sys.argv

    def demo():
        run_demo(system)
        return None

    def interactive():
        def body():
            user = build_profile_interactive()
            print("\n  Generating personalized meal plan...\n")
            result = system.filter_recipes(user)
            print_results(result, top_n=15)

            print("\nSave results?")
            print("1. Save as CSV")
            print("2. Save as JSON")
            print("3. No thanks")
            choice = input("\nSelect (1-3): ").strip()

            # dispatch بدل if/elif/else
            actions = {
                "1": lambda: save_results(result, fmt="csv"),
                "2": lambda: save_results(result, fmt="json"),
            }
            actions.get(choice, lambda: print("\nOK. Results not saved."))()
            return None

        def on_value(ve):
            print(f"\n❌ Validation error: {ve}")
            return None

        def on_interrupt(_):
            print("\n\n👋 Exited.")
            return None

        def on_other(e):
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None

        try:
            return body()
        except ValueError as ve:
            return on_value(ve)
        except KeyboardInterrupt as ki:
            return on_interrupt(ki)
        except Exception as e:                         # noqa: BLE001
            return on_other(e)

    _do(is_demo, demo, interactive)


# نمط بديل عن if __name__ == "__main__": عبر فهرسة قاموس
{"__main__": main}.get(__name__, lambda: None)()
