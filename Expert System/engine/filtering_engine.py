

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
from engine.rule_builder import get_applicable_rules
from engine.scorer import score_recipe, explain_recipe
from engine.ncf_model import NCFRecommender



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
    """بديل التعبير الثلاثي."""
    return {True: when_true, False: when_false}[bool(cond)]


def _do(cond, do_true: Callable, do_false: Callable):
    """بديل if/else تنفيذي: ينفّذ الدالة المناسبة للشرط."""
    return {True: do_true, False: do_false}[bool(cond)]()


def _fold(seq, acc, fn: Callable[[Any, Any], Any]):
    """تطبيق fn على عناصر seq دون عودية ودون for/while صريحة."""
    return reduce(lambda accumulator, item: fn(accumulator, item), list(seq), acc)


def _first_present(df_cols, candidates: List[str]) -> Optional[str]:
    """
    أول اسم من candidates موجود في الأعمدة — numpy masking دون عودية.
    🛠️ مُصحَّحة: found[0] هو str عادي (ضمن dtype=object array)، وليس
    numpy scalar، فلا يحتاج .item() — استدعاؤها كان يكسر الدالة فوراً.
    """
    cand = np.array(candidates, dtype=object)
    mask = np.isin(cand, list(df_cols))
    found = cand[mask]
    return _pick(len(found) > 0, found[0], None)


def _build_terms_pattern(terms: List[str]) -> str:
    """
    يبني نمط regex بحدود كلمة (\\b) مرة واحدة من قائمة مصطلحات.
    🛠️ إصلاح أداء: كانت _has_any_terms تبني هذا النمط من الصفر داخل
    كل استدعاء .apply() لكل صف على حدة (384,541 مرة!) — هذا كان يرفع
    زمن التنفيذ من أجزاء الثانية إلى عشرات الثواني/دقائق لكل قائمة
    مصطلحات. الحل: نبني النمط مرة واحدة هنا، ونمرره جاهزاً.

    🛠️ إصلاح إضافي (مكتشَف بالاستخدام الفعلي): "Grilled Scallops" لم
    تكن تُحجب عن مستخدم نباتي لأن \\bscallop\\b لا يطابق "Scallops"
    (الجمع) — \\b بعد "scallop" لا يعتبر "s" التالية حدّاً صحيحاً.
    أضفنا "s?" اختيارية لنهاية كل مصطلح: تطابق "Scallops"، "Eggs"،
    "Mayonnaises" دون كسر حماية "Eggplant"/"Wheatgrass" (لأن "plant"
    أو "grass" أطول من حرف "s" وحيد، فالنمط لا يتقدّم بعدها أصلاً).
    """
    has_terms = len(terms) > 0
    safe_terms = terms if has_terms else [""]
    return "|".join(r"\b" + re.escape(t.lower()) + r"s?\b" for t in safe_terms)


def _text_matches_pattern_vectorized(series: pd.Series, pattern: str) -> pd.Series:
    """
    قناع بوولياني مُسرَّع (vectorized) عبر str.contains بدل .apply()
    صف-بصف — أسرع بمراحل على أعمدة كبيرة (384K+ صف)، مع نفس دقة
    حدود الكلمة في النمط الممرَّر.
    """
    return series.fillna("").astype(str).str.lower().str.contains(pattern, regex=True, na=False)


def _has_any_terms(text: str, terms: List[str]) -> bool:
    """
    نسخة "صف واحد" (تُستخدم فقط حين لا يمكن التجميع المُسرَّع، مثل
    القوائم Python الحقيقية لا النصوص). تبني النمط من terms مباشرة —
    تبقى صحيحة لكنها أبطأ من _text_matches_pattern_vectorized إذا
    استُخدمت بكثرة؛ المسار الأساسي بالفلترة الآن يستخدم النسخة
    المُسرَّعة مباشرة (انظر _filter_ing).
    """
    pattern = _build_terms_pattern(terms)
    return bool(re.search(pattern, str(text).lower()))



class DietaryExpertSystem:

    def __init__(self, df: pd.DataFrame, train_ncf: bool = False):
        self.df = df.copy().reset_index(drop=True)
        self._normalize_columns()
        self._ing_col = self._detect_ingredient_column()
        print(f"Expert System Ready | {len(self.df):,} recipes loaded")

        _do(
            bool(self._ing_col),
            lambda: print(f"   Ingredient column: '{self._ing_col}'"),
            lambda: None,
        )

        self.ncf = NCFRecommender()
        need_train = train_ncf and (not self.ncf.trained)
        no_model   = (not train_ncf) and (not self.ncf.trained)

        _do(need_train, lambda: self.ncf.train(self.df, epochs=5), lambda: None)
        _do(
            no_model,
            lambda: print("   NCF: no saved model — run with train_ncf=True to train."),
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
        """آمن مع NaN والقوائم — بلا if/comprehension."""
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

       
        ncf_scores = self.ncf.predict_scores(df, goal=profile.goal)
        ncf_ok = (
            self.ncf.trained
            and (ncf_scores is not None)
            and (len(ncf_scores) == len(df))
        )

        def with_ncf():
            d = df.copy()
            d["_ncf_score"]   = ncf_scores
            d["_final_score"] = d["_expert_score"] * 0.7 + d["_ncf_score"] * 0.3
            return d, True

        def without_ncf():
            d = df.copy()
            d["_final_score"] = d["_expert_score"]
            return d, False

        df, ncf_active = _do(ncf_ok, with_ncf, without_ncf)

        df = df.sort_values("_final_score", ascending=False)
       
        is_df_empty_explain = len(df) == 0
        df["_reason"] = (
            pd.Series([], dtype="object", index=df.index) if is_df_empty_explain
            else df.apply(lambda r: explain_recipe(r, profile.goal), axis=1)
        )

        drop_cols = ["_expert_score", "_final_score"] + _pick(
            ncf_active, ["_ncf_score"], []
        )
        df = df.drop(columns=drop_cols)

        
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
            "ncf_active":           ncf_active,
        }



def map_safe(raw):
    """تحويل عناصر القائمة لنصوص دون map/comprehension (numpy)."""
    return np.array(raw, dtype=object).astype(str).tolist()


def _join_escaped(terms, word_boundary=False, escape=True):
    """
    ابنِ نمط regex من قائمة مصطلحات بلا comprehension. الفارغ → ''.
    🛠️ إصلاح: أضفنا "s?" اختيارية قبل الحد الأخير \\b، لمطابقة صيغ
    الجمع بأسماء الوصفات (مثل "Scallops" بدل "Scallop") بنفس منطق
    _build_terms_pattern، عبر np.char.add المتجهة (vectorized) بدلاً
    من حلقة، حفاظاً على صفر if/for/while.
    """
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
    """استبعد الصفوف التي يحوي اسمها أياً من المصطلحات."""
    def do():
        pattern = _join_escaped(terms, word_boundary=word_boundary, escape=escape)
        mask = df["Name"].fillna("").str.lower().str.contains(
            pattern, regex=True, na=False
        )
        return df[~mask]

    return _do("Name" in df.columns, do, lambda: df)


def _filter_name_pattern(df, pattern: str):
    """استبعد بالنمط الجاهز (قد يكون فارغاً → لا فلترة)."""
    def do():
        mask = df["Name"].fillna("").str.lower().str.contains(
            pattern, regex=True, na=False
        )
        return df[~mask]

    return _do(len(pattern) > 0, do, lambda: df)


def _filter_ing(df, ing_col, terms, text_has_any):
    """
    استبعد الصفوف التي تحوي مكوّناتها أياً من المصطلحات.
    🛠️ إصلاح أداء: بدل .apply() صف-بصف (يستدعي text_has_any() لكل صف،
    وكانت كل استدعاء تعيد بناء pattern من الصفر — كارثي على 384K+ صف)،
    نبني pattern مرة واحدة هنا، ثم نستخدم str.contains المُسرَّع
    (vectorized) على العمود كاملاً دفعة واحدة. text_has_any لم تُحذف
    من التوقيع لإبقاء التوافق مع استدعاءات أخرى، لكنها غير مستخدمة هنا.
    """
    pattern = _build_terms_pattern(terms)
    mask = _text_matches_pattern_vectorized(df[ing_col], pattern)
    return df[~mask]


def _apply_numeric_rule(df, item):
    """
    طبّق قاعدة رقمية واحدة (col, (op, val)) عبر boolean masking.

    🛠️ إصلاح bug خطير جداً (مكتشَف بمراجعة دقيقة لكل قواعد الدمج):
    كانت is_between تُرجع df بدون أي فلترة إطلاقاً (تخطٍّ كامل
    ومتعمَّد لقاعدة between)، حتى بعد إصلاح rule_builder.py لتمرير
    القاعدة كاملة (op, low, high) بدل قصها. النتيجة: 7 قواعد طبية
    حرجة (مثل "سعرات الحمل+السكري يجب بين 400-550") لم تُطبَّق
    إطلاقاً منذ بداية المشروع. الحل: نضيف عملية "between" فعلية
    لقاموس ops، تتحقق من low<=x<=high. القيم المفقودة (NaN) تُملأ
    بقيمة بعيدة جداً عن أي نطاق واقعي (تضمن استبعاد الصف بأمان،
    بدل تخمين أنها "داخل النطاق" أو "خارجه" اعتباطياً).
    """
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
    """
    فلترة نوع الوجبة مع تراجع (fallback) لو لا نتائج.

    🛠️ إصلاح (مكتشَف بالاستخدام الفعلي): عمود MealType بالبيانات
    الحقيقية يحتوي فقط 4 قيم: MainDish, Other, Drink, Breakfast —
    لا توجد قيمة "Lunch" أو "Dinner" حرفياً بالبيانات إطلاقاً. الكود
    السابق كان يبحث عن "lunch"/"dinner" حرفياً، فلا يطابق أي صف أبداً،
    فيتفعّل fallback (يعرض كل الوصفات) دائماً عند اختيار Lunch/Dinner
    — يُفقد الفلتر فعليته بالكامل لهاتين الخيارين. الحل: نترجم خيار
    الواجهة (lunch/dinner) إلى القيمة الحقيقية المطابقة (MainDish)
    عبر MEAL_TYPE_DATA_MAP قبل البحث، مع إبقاء Breakfast كما هي
    (تطابق مباشرة وموجودة أصلاً بالبيانات).
    """
    real_value = MEAL_TYPE_DATA_MAP.get(meal_type.lower(), meal_type)
    active = (meal_type != "any") and ("MealType" in df.columns)

    def do():
        mask = df["MealType"].fillna("").str.lower().str.contains(
            real_value.lower(), na=False
        )
        df_meal = df[mask]
        
        return _pick(len(df_meal) > 0, df_meal, df)

    return {True: do, False: (lambda: df)}[active]()
