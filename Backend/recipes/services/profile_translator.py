"""
Profile translation layer: Django UserProfile -> AI Expert System UserProfile
==============================================================================

Single well-known function, ``translate_profile()``, that converts a
``profiles.models.UserProfile`` (Django ORM) into the AI project's
``core.user_profile.UserProfile`` dataclass so the meal-planning engine can
consume it directly.

WHY THIS LAYER EXISTS
    The two profile types share an intent but NOT a vocabulary:
      - Django stores goals as lose_weight/maintain_weight/gain_weight/gain_muscle;
        the AI engine's GOAL_VECTORS use weight_loss/maintenance/weight_gain/
        muscle_gain/heart_health/pregnancy_diet/elderly_diet/healthy_eating.
      - Django stores conditions/allergies/preferences as free-form JSON string
        lists. The AI engine SILENTLY DROPS any string it does not recognize
        (rule_builder._known_conditions / _known_preferences / _allergies_block),
        so a near-miss value ("diabetic") would silently disable that condition's
        rules with no error. This translator normalizes common near-miss aliases
        and then RAISES on anything still unrecognized, so the view layer can
        turn it into a 400 Bad Request instead of a silently-degraded plan.

SCOPE BOUNDARY — meal_type is NOT translated here
    Django's meal_type field (standard/vegetarian/vegan/keto/halal/other) is a
    stored DIET-PREFERENCE concept and IS translated — into the AI profile's
    ``preferences`` list. The AI profile's OWN ``meal_type`` field
    (breakfast/lunch/dinner/any — which meal the user wants RIGHT NOW) is a
    PER-REQUEST concept and is explicitly OUT OF SCOPE for a stored-profile
    translator: it will be accepted as a request parameter by the endpoint
    views built later, and the AI dataclass default ("any") is left untouched.
    Django's taste_text field is also out of scope here (it feeds the TOPSIS
    taste-ranking step as a separate argument, not the expert system profile).

EAGER GOAL OVERRIDES (documented in detail at GOAL_MAP below)
    - pregnant=True  -> goal becomes "pregnancy_diet" (medical priority).
    - age >= 65 with a general weight/maintenance goal -> "elderly_diet".

Usage::

    from recipes.services.profile_translator import (
        translate_profile, ProfileTranslationError,
    )

    ai_profile = translate_profile(django_profile)          # may raise
    result = get_expert_system().filter_recipes(ai_profile)
"""

from typing import Dict, List

# The AI project's real vocabularies are imported (not re-declared) so this
# layer can never drift from the engine's actual rules. These imports are
# lightweight dict modules; they resolve because settings.py -> ai_bridge
# already added the "Expert System" folder to sys.path.
from core.user_profile import UserProfile
from rules.goals_and_preferences import GOAL_VECTORS, PREFERENCE_BLOCKS
from rules.halal_and_allergies import ALLERGY_RULES
from rules.medical_rules import MEDICAL_RULES


class ProfileTranslationError(Exception):
    """Raised when a Django profile cannot be translated safely.

    The message names the offending field, the bad value(s), and — where
    useful — the expected vocabulary, so callers can surface it as a
    400 Bad Request without re-deriving anything.
    """


# ---------------------------------------------------------------------------
# Goal mapping
# ---------------------------------------------------------------------------
# Django's Goal choices are a user-facing subset of the AI engine's richer
# GOAL_VECTORS vocabulary. The mapping below is exhaustive over Django's four
# choices (verified against profiles/models.py).
GOAL_MAP: Dict[str, str] = {
    "lose_weight":    "weight_loss",
    "maintain_weight": "maintenance",
    "gain_weight":    "weight_gain",
    "gain_muscle":    "muscle_gain",
}

# Eager goal overrides — two AI goals ARE reachable through real Django fields.
#
# pregnancy_diet: Django's pregnant boolean genuinely exists. When a user is
# pregnant the generic weight goals are medically inappropriate — especially
# lose_weight, whose -150 kcal/day deficit is contraindicated in pregnancy.
# pregnancy_diet is the ACOG-specialized goal (protein- and fiber-forward,
# +50 kcal offset), and the engine already applies MEDICAL_RULES["pregnancy"]
# from the `pregnant` flag, so this override keeps scoring aligned with the
# safety rules the engine is already enforcing. Applied to EVERY goal: a
# muscle_gain goal during pregnancy has no medical meaning worth preserving.
PREGNANT_GOAL = "pregnancy_diet"

# elderly_diet: Django's age field genuinely exists. For the three general
# weight/maintenance goals, elderly_diet (ESPEN 2019 + NIA) is the
# dietitians' specialized variant: protein-forward (sarcopenia prevention),
# sodium-moderated, mild -30 kcal deficit. The engine ALREADY applies
# MEDICAL_RULES["elderly"] via life_stage for age >= 65, so this only adjusts
# the SCORING weights, never the safety rules. gain_muscle is deliberately
# NOT overridden: muscle_gain (protein 0.98) is already the protein-heavy
# specialized variant for a deliberate training goal.
ELDERLY_AGE_THRESHOLD = 65
ELDERLY_GOAL = "elderly_diet"
_ELDERLY_UPGRADABLE_GOALS = frozenset({"weight_loss", "weight_gain", "maintenance"})

# KNOWLEDGE GAP (not a bug): heart_health and healthy_eating have no Django
# field that could justify an auto-upgrade, so they are currently UNREACHABLE
# via the stored Django profile. The team may add goal choices to the Django
# model in a future migration if these are wanted. Until then every Django
# goal maps to exactly one of the four GOAL_MAP targets (plus the two overrides).


# ---------------------------------------------------------------------------
# meal_type (Django diet preference) -> AI preferences list
# ---------------------------------------------------------------------------
# Django's MealType choices are a DIET-PREFERENCE concept; they translate into
# the AI profile's `preferences` list, using PREFERENCE_BLOCKS keys only.
MEAL_TYPE_PREFERENCES: Dict[str, List[str]] = {
    "standard":  [],                     # no restriction -> no preference block
    "vegetarian": ["vegetarian"],        # direct PREFERENCE_BLOCKS key
    "vegan":     ["vegan"],              # direct PREFERENCE_BLOCKS key
    # PREFERENCE_BLOCKS has no "keto" key; "low_carb" is the real block that
    # enforces a ketogenic-style meal (CarbohydrateContent <= 30, SugarContent
    # <= 5 numeric rules + carb-heavy ingredient blacklist). It is the closest
    # real key and the intended target, verified against goals_and_preferences.py.
    "keto":      ["low_carb"],
    # halal needs NO preference entry: the Expert System applies the halal
    # filter (pork/alcohol, ingredients + recipe-name) UNCONDITIONALLY in
    # filtering_engine.py (HALAL_BLACKLIST step) regardless of any flag.
    "halal":     [],
    "other":     [],                     # unknown intent -> no restriction
}

# ---------------------------------------------------------------------------
# Alias normalization + validation vocabulary
# ---------------------------------------------------------------------------
# Applied BEFORE validation: strip, lowercase, collapse whitespace, then this
# lookup. A handful of the most likely frontend-typed near-misses only — not
# an exhaustive dictionary. Anything that still does not match the real
# vocabulary after normalization RAISES (never silently dropped, unlike the
# engine's own _known_conditions/_known_preferences which ignore unknowns).

CONDITION_ALIASES: Dict[str, str] = {
    "diabetic":                 "diabetes",
    "diabetes mellitus":        "diabetes",
    "high blood pressure":      "hypertension",
    "heart disease":            "heart_disease",
    "cardiovascular disease":   "heart_disease",
    "kidney disease":           "chronic_kidney_disease",
    "ckd":                      "chronic_kidney_disease",
    "high cholesterol":         "high_cholesterol",
    "cholesterol":              "high_cholesterol",
    "ibs":                      "irritable_bowel_syndrome",
    "crohn's disease":          "crohns_disease",
    "crohn's":                  "crohns_disease",
    "crohns":                   "crohns_disease",
    "crohn":                    "crohns_disease",
    "hypothyroid":              "hypothyroidism",
    "hyperthyroid":             "hyperthyroidism",
    "lactose intolerant":       "lactose_intolerance",
    "celiac":                   "gluten_intolerance",
    "celiac disease":           "gluten_intolerance",
    "coeliac":                  "gluten_intolerance",
    "nut allergy":              "nut_allergy",
    "peanut allergy":           "nut_allergy",
}

# Separate dict because one string can mean different things per context:
# "nut allergy" as a CONDITION is MEDICAL_RULES["nut_allergy"], but as an
# ALLERGY it is ALLERGY_RULES["peanuts"] (whose strict_block reuses the full
# nut_allergy block — peanuts + tree nuts — per halal_and_allergies.py).
ALLERGY_ALIASES: Dict[str, str] = {
    "nut allergy":   "peanuts",
    "peanut allergy": "peanuts",
    "peanut":        "peanuts",
    "tree nuts":     "peanuts",
    "tree nut":      "peanuts",
    "nuts":          "peanuts",
    "dairy":         "milk",
    "milk allergy":  "milk",
    "egg":           "eggs",
    "egg allergy":   "eggs",
    "shellfish":     "seafood",
    "fish":          "seafood",
    "soya":          "soy",
    "soybean":       "soy",
    "soybeans":      "soy",
    "sesame seeds":  "sesame",
    "sesame seed":   "sesame",
    "gluten allergy": "gluten",
    "wheat":         "gluten",
    "celiac":        "gluten",
}

# The Django `preferences` JSONField is free-form too; the engine silently
# drops unknown preference strings, so the same normalize-then-raise policy
# applies here (see merge in translate_profile()).
PREFERENCE_ALIASES: Dict[str, str] = {
    "keto":             "low_carb",
    "lowcarb":          "low_carb",
    "low-carb":         "low_carb",
    "mediterranean diet": "mediterranean",
    "no preference":    "no_preference",
    "none":             "no_preference",
}

# Activity vocabulary is shared VERBATIM between Django's ActivityLevel
# choices and the AI dataclass's ACTIVITY_FACTORS keys (verified in both
# files), so it is not aliased — but it is validated so bad DB data cannot
# reach the engine.
_ACTIVITY_VOCABULARY = frozenset({
    "sedentary", "light", "moderate", "active", "very_active",
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _normalize_token(raw) -> str:
    """Lowercase, strip, and collapse internal whitespace. Rejects non-strings
    explicitly — a numeric/None entry inside a list is a data error, not
    something to guess about."""
    if not isinstance(raw, str):
        raise ProfileTranslationError(
            f"profile lists may only contain strings; found {type(raw).__name__}: {raw!r}")
    return " ".join(raw.strip().lower().split())


def _normalize_and_validate(values, aliases: Dict[str, str], vocabulary: Dict,
                            field_name: str) -> List[str]:
    """Alias-normalize + validate one list field against the AI vocabulary.

    None -> [] (the JSONField default is [], but a stored NULL must translate
    cleanly). Unknown values RAISE listing exactly what was wrong."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ProfileTranslationError(
            f"{field_name}: expected a list of strings, got {type(values).__name__}: {values!r}")
    out: List[str] = []
    seen = set()
    for raw in values:
        token = _normalize_token(raw)
        canonical = token if token in vocabulary else aliases.get(token)
        if canonical is None:
            expected = ", ".join(sorted(vocabulary))
            raise ProfileTranslationError(
                f"{field_name}: unrecognized value {raw!r} — after alias "
                f"normalization it still does not match any known key. "
                f"Expected one of: {expected}")
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def translate_profile(django_profile) -> UserProfile:
    """Convert a Django profiles.models.UserProfile into the AI's
    core.user_profile.UserProfile dataclass.

    Raises ProfileTranslationError with a field-specific message for any
    value the AI engine could not honor (never silently drops, unlike the
    engine's own unknown-value handling).
    """
    # --- goal: explicit mapping + documented eager overrides ---------------
    django_goal = getattr(django_profile, "goal", None)
    goal = GOAL_MAP.get(django_goal)
    if goal is None:
        raise ProfileTranslationError(
            f"goal: unrecognized value {django_goal!r} — expected one of: "
            f"{', '.join(sorted(GOAL_MAP))}")

    if getattr(django_profile, "pregnant", False):
        goal = PREGNANT_GOAL
    elif (getattr(django_profile, "age", 0) >= ELDERLY_AGE_THRESHOLD
          and goal in _ELDERLY_UPGRADABLE_GOALS):
        goal = ELDERLY_GOAL

    # --- meal_type -> AI preferences (diet preference, not time-of-day) ----
    meal_type = getattr(django_profile, "meal_type", None)
    meal_type_prefs = MEAL_TYPE_PREFERENCES.get(meal_type)
    if meal_type_prefs is None:
        raise ProfileTranslationError(
            f"meal_type: unrecognized value {meal_type!r} — expected one of: "
            f"{', '.join(sorted(MEAL_TYPE_PREFERENCES))}")

    # --- stored JSON lists: normalize, validate, merge ---------------------
    conditions = _normalize_and_validate(
        getattr(django_profile, "conditions", None),
        CONDITION_ALIASES, MEDICAL_RULES, "conditions")
    allergies = _normalize_and_validate(
        getattr(django_profile, "allergies", None),
        ALLERGY_ALIASES, ALLERGY_RULES, "allergies")

    stored_prefs = _normalize_and_validate(
        getattr(django_profile, "preferences", None),
        PREFERENCE_ALIASES, PREFERENCE_BLOCKS, "preferences")
    # meal_type-derived preferences first, then stored ones, deduplicated.
    preferences: List[str] = list(meal_type_prefs)
    for p in stored_prefs:
        if p not in preferences:
            preferences.append(p)

    # --- scalar fields (vocabularies are identical on both sides) ----------
    gender = getattr(django_profile, "gender", None)
    if gender not in ("male", "female"):
        raise ProfileTranslationError(
            f"gender: unrecognized value {gender!r} — expected 'male' or 'female'")

    activity_level = getattr(django_profile, "activity_level", None)
    if activity_level not in _ACTIVITY_VOCABULARY:
        raise ProfileTranslationError(
            f"activity_level: unrecognized value {activity_level!r} — expected "
            f"one of: {', '.join(sorted(_ACTIVITY_VOCABULARY))}")

    return UserProfile(
        age=getattr(django_profile, "age"),
        height=getattr(django_profile, "height"),
        weight=getattr(django_profile, "weight"),
        gender=gender,
        pregnant=bool(getattr(django_profile, "pregnant", False)),
        conditions=conditions,
        allergies=allergies,
        preferences=preferences,
        goal=goal,
        activity_level=activity_level,
        # AI `meal_type` intentionally NOT set — per-request concept (see
        # module docstring); the dataclass default "any" applies.
    )
