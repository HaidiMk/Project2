"""
filtering_engine.py — Smart Dietary Advisor v4.0 — FIXED
=========================================================
التغييرات:
    - خطوة 5b جديدة: فلترة Name column بالممنوعات
    - soft_block منفصل عن strict_block (لا يُحظر — يُحذر فقط)
    - allergy_blocked آمن مع .get()
    - خطوة 4b جديدة: فلترة نوع الوجبة (فطور/غداء/عشاء)
    - خطوة 8b جديدة: دمج درجات NCF مع Expert System
"""

import re
import math
from typing import Optional, List
import pandas as pd
import numpy as np

from core.user_profile import UserProfile
from core.constants import NUTRIENT_COLS
from rules.halal_and_allergies import HALAL_BLACKLIST
from rules.goals_and_preferences import GOAL_VECTORS, get_healthy_diet_style
from engine.rule_builder import get_applicable_rules
from engine.scorer import score_recipe, explain_recipe
from engine.ncf_model import NCFRecommender


class DietaryExpertSystem:

    def __init__(self, df: pd.DataFrame, train_ncf: bool = False):
        self.df = df.copy().reset_index(drop=True)
        self._normalize_columns()
        self._ing_col = self._detect_ingredient_column()
        print(f"Expert System Ready | {len(self.df):,} recipes loaded")
        if self._ing_col:
            print(f"   Ingredient column: '{self._ing_col}'")

        # ── تحميل أو تدريب NCF ────────────────────────────
        self.ncf = NCFRecommender()
        if train_ncf and not self.ncf.trained:
            self.ncf.train(self.df, epochs=5)
        elif not self.ncf.trained:
            print("   NCF: no saved model — run with train_ncf=True to train.")

    def _normalize_columns(self):
        for col in NUTRIENT_COLS + ["Rating", "HealthScore"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _detect_ingredient_column(self) -> Optional[str]:
        for name in ["IngredientsList", "RecipeIngredientParts", "Ingredients"]:
            if name in self.df.columns:
                return name
        return None

    def _text_has_any(self, raw, terms: List[str]) -> bool:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return False
        text = (" ".join(raw) if isinstance(raw, list) else str(raw)).lower()
        return any(t.lower() in text for t in terms)

    def _build_name_pattern(self, terms: List[str]) -> str:
        filtered = [t for t in terms if len(t) > 2]
        if not filtered:
            return ""
        return "|".join(r"\b" + re.escape(t.lower()) + r"\b" for t in filtered)

    def filter_recipes(self, profile: UserProfile) -> dict:
        rules = get_applicable_rules(profile)
        df    = self.df.copy()

        # ══ 1: فلتر الحلال ═══════════════════════════════
        if self._ing_col:
            df = df[~df[self._ing_col].apply(
                lambda r: self._text_has_any(r, HALAL_BLACKLIST)
            )]

        if "Name" in df.columns:
            HALAL_NAME_BLACKLIST = [
                "pork", "bacon", "ham", "lard",
                "pepperoni", "salami", "prosciutto", "pancetta",
                "chorizo", "pig", "swine",
                "wine", "beer", "alcohol", "vodka", "rum", "gin",
                "whiskey", "whisky", "champagne", "liqueur",
                "brandy", "tequila", "sake", "bourbon",
            ]
            halal_pattern = "|".join(
                r"\b" + re.escape(h) + r"\b"
                for h in HALAL_NAME_BLACKLIST
            )
            df = df[~df["Name"].fillna("").str.lower().str.contains(
                halal_pattern, regex=True, na=False
            )]

        # ══ 2: أعمدة الحساسيات ═══════════════════════════
        for col in rules["allergy_filters"]:
            if col in df.columns:
                df = df[df[col] == False]

        # ══ 3: الحدود الرقمية ════════════════════════════
        op_fn = {
            "<=": lambda x, v: x <= v,
            ">=": lambda x, v: x >= v,
            "<":  lambda x, v: x < v,
            ">":  lambda x, v: x > v,
        }
        for col, rule in rules["numeric_rules"].items():
            if col not in df.columns or not isinstance(rule, tuple) or len(rule) < 2:
                continue
            op, val = rule[0], rule[1]
            if op == "between":
                continue
            fill = 9999 if op.startswith("<") else 0
            df = df[op_fn[op](df[col].fillna(fill), val)]

        # ══ 4: تقييد البروتين غير الواقعي ════════════════
        if "ProteinContent" in df.columns:
            df = df[df["ProteinContent"] <= 80]

        # ══ 4b: فلترة نوع الوجبة ══════════════════════════
        meal_type = getattr(profile, "meal_type", "any")
        if meal_type != "any" and "MealType" in df.columns:
            df_meal = df[
                df["MealType"].fillna("").str.lower().str.contains(
                    meal_type, na=False
                )
            ]
            if len(df_meal) > 0:
                df = df_meal

        # ══ 5a: فلترة نص المكونات ════════════════════════
        strict_blocked = list(set(
            rules.get("strict_block", []) +
            rules.get("allergy_blocked", [])
        ))
        if strict_blocked and self._ing_col:
            df = df[~df[self._ing_col].apply(
                lambda r: self._text_has_any(r, strict_blocked)
            )]

        # ══ 5b: فلترة اسم الوصفة بالممنوعات ══════════════
        if strict_blocked and "Name" in df.columns:
            name_pattern = self._build_name_pattern(strict_blocked)
            if name_pattern:
                df = df[~df["Name"].fillna("").str.lower().str.contains(
                    name_pattern, regex=True, na=False
                )]

        # ══ 6: فلتر GERD على اسم الوصفة ══════════════════
        if "gerd" in profile.conditions and "Name" in df.columns:
            gerd_blist = [
                "cocktail", "martini", "mimosa", "margarita", "daiquiri",
                "mojito", "sangria", "citrus", "sour",
            ]
            pattern = "|".join(re.escape(x) for x in gerd_blist)
            df = df[~df["Name"].fillna("").str.lower().str.contains(
                pattern, regex=True, na=False
            )]

        # ══ 7: السعرات المستهدفة ══════════════════════════
        target_cal = profile.per_meal_calories
        if profile.goal and profile.goal in GOAL_VECTORS:
            offset = GOAL_VECTORS[profile.goal].get("target_calorie_offset", 0)
            target_cal = max(200, target_cal + offset)

        # ══ 8: تقييم Expert System ════════════════════════
        preferred = rules.get("preferred_ingredients", [])
        df = df.copy()
        df["_expert_score"] = df.apply(
            lambda r: score_recipe(
                r, profile.goal, target_cal, preferred, self._ing_col, meal_type
            ),
            axis=1,
        )

        # ══ 8b: دمج NCF ← جديد ════════════════════════════
        ncf_scores = self.ncf.predict_scores(df, goal=profile.goal)

        if self.ncf.trained and ncf_scores is not None and len(ncf_scores) == len(df):
            # دمج: 70% Expert System + 30% NCF
            df["_ncf_score"]   = ncf_scores
            df["_final_score"] = df["_expert_score"] * 0.7 + df["_ncf_score"] * 0.3
            ncf_active = True
        else:
            df["_final_score"] = df["_expert_score"]
            ncf_active = False

        df = df.sort_values("_final_score", ascending=False)
        df["_reason"] = df.apply(lambda r: explain_recipe(r, profile.goal), axis=1)
        df = df.drop(columns=["_expert_score", "_final_score"] +
                     (["_ncf_score"] if ncf_active else []))

        # ══ 9: التحذيرات ══════════════════════════════════
        warnings = list(rules["conflict_messages"])

        if meal_type != "any" and "MealType" in self.df.columns:
            check = self.df[
                self.df["MealType"].fillna("").str.lower().str.contains(
                    meal_type, na=False
                )
            ]
            if len(check) == 0:
                warnings.append(
                    f"No '{meal_type}' recipes found — showing all meal types."
                )

        soft_warned = rules.get("soft_block", [])
        if soft_warned:
            sample = ", ".join(soft_warned[:5])
            warnings.append(
                f"Caution: not ideal but not blocked: "
                f"{sample}{'...' if len(soft_warned) > 5 else ''}"
            )

        if len(df) < 5:
            warnings.append(
                f"Only {len(df)} recipes match all restrictions."
            )

        healthy_style = None
        if not profile.conditions and not profile.pregnant:
            healthy_style = get_healthy_diet_style(profile)

        return {
            "safe_recipes":         df.reset_index(drop=True),
            "total_safe":           len(df),
            "total_original":       len(self.df),
            "filter_rate":          round(len(df) / max(len(self.df), 1) * 100, 1),
            "rules_applied":        rules["numeric_rules"],
            "blocked_list":         strict_blocked[:40],
            "warnings":             warnings,
            "notes":                rules["notes"],
            "healthy_style":        healthy_style,
            "profile_summary":      profile.summary(),
            "target_meal_calories": target_cal,
            "meal_type":            meal_type,
            "ncf_active":           ncf_active,   # ← جديد
        }