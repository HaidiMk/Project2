

import re
from functools import reduce
from typing import Optional, List, Callable, Any

import pandas as pd
import numpy as np

from core.user_profile import UserProfile
from core.constants import NUTRIENT_COLS
from rules.halal_and_allergies import HALAL_BLACKLIST
from rules.goals_and_preferences import GOAL_VECTORS, get_healthy_diet_style
from rules.medical_rules import MEDICAL_RULES
from engine.rule_builder import get_applicable_rules, get_applicable_condition_keys
from engine.scorer import score_recipe, explain_recipe
from ml.health_classifier.inference import ai_health_score



MEAL_TYPE_DATA_MAP = {
    "breakfast": "Breakfast",  
    "lunch":     "MainDish",   
    "dinner":    "MainDish",   
}



PET_ONLY_ANIMALS = (
    r"dogs?|puppy|puppies|doggie|doggy"
    r"|cats?|kittens?|kitty"
    r"|horses?|ponies|pony"
    r"|birds?|parrots?|budgies?|canar(?:y|ies)"
    r"|hamsters?|guinea pigs?"
    r"|rabbits?|bunn(?:y|ies)"
    r"|ferrets?"
    r"|turtles?|lizards?|reptiles?|snakes?"
    r"|livestock|chicks?|goldfish"
)
STANDALONE_PET_WORDS = r"doggie|doggy|kitty|livestock|goldfish"
SENSITIVE_ANIMALS = r"fish|chicken|pigs?|cows?|cattle"
_ANIMAL_CONTEXT = r"treats?|foods?|biscuits?|training|kibble|pellets?|flakes?|feeds?"


_POSSESSIVE = r"(?:'s)?"
_SEP = r"[\W]+"
_OPTIONAL_MIDDLE_WORDS = r"(?:\w+\W+){0,2}"

PET_RECIPE_PATTERN = (
    rf"\bfor (?:my |your |our )?(?:{PET_ONLY_ANIMALS}|pets?)\b"
    rf"|\b(?:{PET_ONLY_ANIMALS})\b{_POSSESSIVE}{_SEP}{_OPTIONAL_MIDDLE_WORDS}(?:{_ANIMAL_CONTEXT})"
    rf"|\bpet\b{_POSSESSIVE}{_SEP}(?:treats?|foods?)\b"
    rf"|\bfish food\b"
    rf"|\b(?:{STANDALONE_PET_WORDS})\b"
    rf"|\b(?:{SENSITIVE_ANIMALS})\b{_POSSESSIVE}{_SEP}{_OPTIONAL_MIDDLE_WORDS}"
    rf"(?:treats?|kibble|pellets?|flakes?|feeds?)\b"
)



NON_FOOD_TERMS = (
    r"scrub|face mask|facial mask|lip balm|lip scrub"
    r"|bath bomb|bath salt|bath oil|body butter|body lotion"
    r"|hair mask|hair treatment|shampoo"
    r"|facial (?:cleansing )?lotion|suntan lotion|moisturizing lotion"
    r"|healing lotion|oil lotion"
    r"|cough drop|vapor rub|salve|ointment"
    r"|solid perfume|deodorant"
)

_FOOD_INDICATOR_WORDS = (
    r"cake|cookies?|cupcakes?|brownies?|dessert|candy"
    r"|cocktail|pie|tart|muffins?|chowder|soup|salad|sandwich|sauce|dip"
)
NON_FOOD_RECIPE_PATTERN = rf"\b(?:{NON_FOOD_TERMS})\b"




def _pick(cond, when_true, when_false):
    return {True: when_true, False: when_false}[bool(cond)]


def _do(cond, do_true: Callable, do_false: Callable):
    return {True: do_true, False: do_false}[bool(cond)]()


def _fold(seq, acc, fn: Callable[[Any, Any], Any]):
    return reduce(lambda accumulator, item: fn(accumulator, item), list(seq), acc)


def _first_present(df_cols, candidates: List[str]) -> Optional[str]:
    cand = np.array(candidates, dtype=object)
    mask = np.isin(cand, list(df_cols))
    found = cand[mask]
    return _pick(len(found) > 0, found[0], None)


def _build_terms_pattern(terms: List[str]) -> str:
    has_terms = len(terms) > 0
    safe_terms = terms if has_terms else [""]
    return "|".join(r"\b" + re.escape(t.lower()) + r"s?\b" for t in safe_terms)


def _text_matches_pattern_vectorized(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.contains(pattern, regex=True, na=False)


def _has_any_terms(text: str, terms: List[str]) -> bool:
    pattern = _build_terms_pattern(terms)
    return bool(re.search(pattern, str(text).lower()))



class DietaryExpertSystem:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        self._normalize_columns()
        self._ing_col = self._detect_ingredient_column()
        print(f"Expert System Ready | {len(self.df):,} recipes loaded")

        _do(
            bool(self._ing_col),
            lambda: print(f"   Ingredient column: '{self._ing_col}'"),
            lambda: None,
        )

   
    def _normalize_columns(self):
        cols = NUTRIENT_COLS + ["Rating", "HealthScore"]
        present = list(np.array(cols)[np.isin(cols, list(self.df.columns))])

        def conv(_, col):
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            return None

        _fold(present, None, conv)

    def _detect_ingredient_column(self) -> Optional[str]:
        return _first_present(
            self.df.columns,
            ["IngredientsList", "RecipeIngredientParts", "Ingredients"],
        )

    def _text_has_any(self, raw, terms: List[str]) -> bool:
        is_nan = isinstance(raw, float) and np.isnan(raw)
        bad = (raw is None) or is_nan

        def good():
            text = _pick(isinstance(raw, list), " ".join(map_safe(raw)), str(raw))
            return _has_any_terms(text, terms)

        return {True: (lambda: False), False: good}[bad]()

    def _build_name_pattern(self, terms: List[str]) -> str:
        """نمط regex من المصطلحات الأطول من حرفين — بلا comprehension."""
        arr = np.array(_pick(len(terms) > 0, terms, []), dtype=object).astype(str)
        keep = np.array(list(np.char.str_len(arr) > 2))
        filtered = arr[keep].tolist()
        return _join_escaped(filtered, word_boundary=True, escape=True)

   
    def filter_recipes(self, profile: UserProfile) -> dict:
        rules = get_applicable_rules(profile)
        df    = self.df.copy()

        
        not_pet_mask = ~df["Name"].fillna("").str.lower().str.contains(
            PET_RECIPE_PATTERN, regex=True, na=False
        )
        df = df[not_pet_mask]

        
        matches_nonfood = df["Name"].fillna("").str.lower().str.contains(
            NON_FOOD_RECIPE_PATTERN, regex=True, na=False
        )
        has_food_indicator = df["Name"].fillna("").str.lower().str.contains(
            _FOOD_INDICATOR_WORDS, regex=True, na=False
        )
        not_nonfood_mask = ~(matches_nonfood & ~has_food_indicator)
        df = df[not_nonfood_mask]

        
        has_keywords_col = "KeywordsList" in df.columns
        NON_FOOD_KEYWORD_PATTERN = r"bath/beauty"

        def exclude_by_keywords():
            kw = df["KeywordsList"].fillna("").astype(str).str.lower()
            matches_kw = kw.str.contains(NON_FOOD_KEYWORD_PATTERN, regex=True, na=False)
            return df[~matches_kw]

        df = _do(has_keywords_col, exclude_by_keywords, lambda: df)

       
        df = _do(
            bool(self._ing_col),
            lambda: _filter_ing(df, self._ing_col, HALAL_BLACKLIST, self._text_has_any),
            lambda: df,
        )

        
        HALAL_NAME_BLACKLIST = [
            "pork", "bacon", "ham", "lard",
            "pepperoni", "salami", "prosciutto", "pancetta",
            "chorizo", "pig", "swine",
            "wine", "beer", "alcohol", "vodka", "rum", "gin",
            "whiskey", "whisky", "champagne", "liqueur",
            "brandy", "tequila", "sake", "bourbon",
        ]
        df = _filter_name_contains(df, HALAL_NAME_BLACKLIST, word_boundary=True)

        
        def drop_allergy(d, col):
            present = col in d.columns
            mask = d[col] == False if present else pd.Series(True, index=d.index)
            return d[mask]

        df = _fold(rules["allergy_filters"], df, drop_allergy)

        
        has_any_allergy = len(rules["allergy_filters"]) > 0
        ingredients_present = bool(self._ing_col) and (self._ing_col in df.columns)

        def exclude_unknown_ingredients():
            raw = df[self._ing_col].fillna("").astype(str).str.strip()
            is_empty = raw.isin(["", "[]", "['']"])
            return df[~is_empty]

        df = _do(
            has_any_allergy and ingredients_present,
            exclude_unknown_ingredients,
            lambda: df,
        )

       
        df = _fold(
            list(rules["numeric_rules"].items()),
            df,
            _apply_numeric_rule,
        )

        
        df = _do(
            "ProteinContent" in df.columns,
            lambda: df[df["ProteinContent"] <= 80],
            lambda: df,
        )

        
        meal_type = getattr(profile, "meal_type", "any")
        df = _apply_meal_filter(df, meal_type)

        
        strict_blocked = list(set(
            rules.get("strict_block", []) + rules.get("allergy_blocked", [])
        ))
        has_strict = len(strict_blocked) > 0

        df = _do(
            has_strict and bool(self._ing_col),
            lambda: _filter_ing(df, self._ing_col, strict_blocked, self._text_has_any),
            lambda: df,
        )

       
        df = _do(
            has_strict and ("Name" in df.columns),
            lambda: _filter_name_pattern(df, self._build_name_pattern(strict_blocked)),
            lambda: df,
        )

       
        gerd_on = ("gerd" in profile.conditions) and ("Name" in df.columns)
        gerd_blist = [
            "cocktail", "martini", "mimosa", "margarita", "daiquiri",
            "mojito", "sangria", "citrus", "sour",
        ]
        df = _do(
            gerd_on,
            lambda: _filter_name_contains(df, gerd_blist, escape=True),
            lambda: df,
        )

       
        base_cal = profile.per_meal_calories
        goal_ok = bool(profile.goal) and (profile.goal in GOAL_VECTORS)
        offset = GOAL_VECTORS.get(_pick(goal_ok, profile.goal, ""), {}).get(
            "target_calorie_offset", 0
        )
        goal_based_target = _pick(goal_ok, max(200, base_cal + offset), base_cal)

        
        is_underweight = "underweight" in profile.conditions
        medical_min_cal = (
            MEDICAL_RULES.get("underweight", {})
            .get("numeric_rules", {})
            .get("Calories", (">=", 0))[1]
        )
        underweight_target = medical_min_cal * 1.10
        target_cal = (
            max(goal_based_target, underweight_target) if is_underweight
            else goal_based_target
        )

       
       
        preferred = rules.get("preferred_ingredients", [])
        df = df.copy()
        is_df_empty = len(df) == 0
        df["_expert_score"] = (
            pd.Series([], dtype="float64", index=df.index) if is_df_empty
            else df.apply(
                lambda r: score_recipe(
                    r, profile.goal, target_cal, preferred, self._ing_col, meal_type
                ),
                axis=1,
            )
        )

        
        cond_keys = get_applicable_condition_keys(profile)
        df["_ai_health_score"] = (
            pd.Series([], dtype="float64", index=df.index) if is_df_empty
            else ai_health_score(df, cond_keys)
        )

        df["_final_score"] = df["_expert_score"]

        df = df.sort_values("_final_score", ascending=False)
       
        is_df_empty_explain = len(df) == 0
        df["_reason"] = (
            pd.Series([], dtype="object", index=df.index) if is_df_empty_explain
            else df.apply(lambda r: explain_recipe(r, profile.goal), axis=1)
        )

        df = df.drop(columns=["_final_score"])

        
        warnings = list(rules["conflict_messages"])

       
        real_meal_value = MEAL_TYPE_DATA_MAP.get(meal_type.lower(), meal_type)
        meal_check = _do(
            (meal_type != "any") and ("MealType" in self.df.columns),
            lambda: len(self.df[
                self.df["MealType"].fillna("").str.lower().str.contains(
                    real_meal_value.lower(), na=False
                )
            ]),
            lambda: -1,   
        )
        warnings = warnings + _pick(
            meal_check == 0,
            [f"No '{meal_type}' recipes found — showing all meal types."],
            [],
        )

        
        soft_warned = rules.get("soft_block", [])
        has_soft = len(soft_warned) > 0
        ellipsis = _pick(len(soft_warned) > 5, "...", "")
        warnings = warnings + _pick(
            has_soft,
            [f"Caution: not ideal but not blocked: "
             f"{', '.join(soft_warned[:5])}{ellipsis}"],
            [],
        )

       
        warnings = warnings + _pick(
            len(df) < 5,
            [f"Only {len(df)} recipes match all restrictions."],
            [],
        )

       
        no_med = (not profile.conditions) and (not profile.pregnant)
        healthy_style = _do(
            no_med,
            lambda: get_healthy_diet_style(profile),
            lambda: None,
        )

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
        }



def map_safe(raw):
    return np.array(raw, dtype=object).astype(str).tolist()


def _join_escaped(terms, word_boundary=False, escape=True):
    def build():
        arr = np.array(terms, dtype=object).astype(str)
        low = np.char.lower(arr)
        esc = _pick(escape, np.frompyfunc(re.escape, 1, 1)(low).astype(str), low)
        end_token = _pick(word_boundary, r"s?\b", "")
        start_token = _pick(word_boundary, r"\b", "")
        bounded = np.char.add(np.char.add(start_token, esc), end_token)
        return "|".join(bounded.tolist())

    return {True: build, False: (lambda: "")}[len(terms) > 0]()


def _filter_name_contains(df, terms, escape=True, word_boundary=False):
    def do():
        pattern = _join_escaped(terms, word_boundary=word_boundary, escape=escape)
        mask = df["Name"].fillna("").str.lower().str.contains(
            pattern, regex=True, na=False
        )
        return df[~mask]

    return _do("Name" in df.columns, do, lambda: df)


def _filter_name_pattern(df, pattern: str):
    def do():
        mask = df["Name"].fillna("").str.lower().str.contains(
            pattern, regex=True, na=False
        )
        return df[~mask]

    return _do(len(pattern) > 0, do, lambda: df)


def _filter_ing(df, ing_col, terms, text_has_any):
    pattern = _build_terms_pattern(terms)
    mask = _text_matches_pattern_vectorized(df[ing_col], pattern)
    return df[~mask]


def _apply_numeric_rule(df, item):
    col, rule = item
    valid = (
        (col in df.columns)
        and isinstance(rule, tuple)
        and (len(rule) >= 2)
    )

    def do():
        op, val = rule[0], rule[1]
        is_between = (op == "between")
        high_val = rule[2] if (is_between and len(rule) >= 3) else val

        def apply_op():
            
            fill = (
                -999999 if is_between
                else 9999 if op.startswith("<")
                else 0
            )
            series = df[col].fillna(fill)
            ops = {
                "<=": lambda: series <= val,
                ">=": lambda: series >= val,
                "<":  lambda: series < val,
                ">":  lambda: series > val,
                "between": lambda: (series >= val) & (series <= high_val),
            }
            mask = ops.get(op, lambda: pd.Series([True] * len(df), index=df.index))()
            return df[mask]

        return apply_op()

    return {True: do, False: (lambda: df)}[valid]()


def _apply_meal_filter(df, meal_type: str):
    real_value = MEAL_TYPE_DATA_MAP.get(meal_type.lower(), meal_type)
    active = (meal_type != "any") and ("MealType" in df.columns)

    def do():
        mask = df["MealType"].fillna("").str.lower().str.contains(
            real_value.lower(), na=False
        )
        df_meal = df[mask]
        
        return _pick(len(df_meal) > 0, df_meal, df)

    return {True: do, False: (lambda: df)}[active]()
