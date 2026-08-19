from typing import Dict, List

from core.user_profile import UserProfile
from rules.goals_and_preferences import GOAL_VECTORS, PREFERENCE_BLOCKS
from rules.halal_and_allergies import ALLERGY_RULES
from rules.medical_rules import MEDICAL_RULES


class ProfileTranslationError(Exception):
GOAL_MAP: Dict[str, str] = {
    "lose_weight":    "weight_loss",
    "maintain_weight": "maintenance",
    "gain_weight":    "weight_gain",
    "gain_muscle":    "muscle_gain",
}

PREGNANT_GOAL = "pregnancy_diet"

ELDERLY_AGE_THRESHOLD = 65
ELDERLY_GOAL = "elderly_diet"
_ELDERLY_UPGRADABLE_GOALS = frozenset({"weight_loss", "weight_gain", "maintenance"})

MEAL_TYPE_PREFERENCES: Dict[str, List[str]] = {
    "standard":  [],                    
    "vegetarian": ["vegetarian"],       
    "vegan":     ["vegan"],             
    "keto":      ["low_carb"],
    "halal":     [],
    "other":     [],                   
}


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

PREFERENCE_ALIASES: Dict[str, str] = {
    "keto":             "low_carb",
    "lowcarb":          "low_carb",
    "low-carb":         "low_carb",
    "low carb":         "low_carb",
    "mediterranean diet": "mediterranean",
    "no preference":    "no_preference",
    "none":             "no_preference",
    "high protein":     "no_preference",
    "low sugar":        "no_preference",
    "healthy":          "no_preference",
    "healthy eating":   "no_preference",
    "omnivore":         "no_preference",
    "pescatarian":      "seafood_lover",
    "pescetarian":      "seafood_lover",
    "chicken only":     "chicken_lover",
    "poultry only":     "chicken_lover",
}

_ACTIVITY_VOCABULARY = frozenset({
    "sedentary", "light", "moderate", "active", "very_active",
})


def _normalize_token(raw) -> str:
    if not isinstance(raw, str):
        raise ProfileTranslationError(
            f"profile lists may only contain strings; found {type(raw).__name__}: {raw!r}")
    return " ".join(raw.strip().lower().split())


def _normalize_and_validate(values, aliases: Dict[str, str], vocabulary: Dict,
                            field_name: str) -> List[str]:
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


def translate_profile(django_profile) -> UserProfile:
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

    meal_type = getattr(django_profile, "meal_type", None)
    meal_type_prefs = MEAL_TYPE_PREFERENCES.get(meal_type)
    if meal_type_prefs is None:
        raise ProfileTranslationError(
            f"meal_type: unrecognized value {meal_type!r} — expected one of: "
            f"{', '.join(sorted(MEAL_TYPE_PREFERENCES))}")

    conditions = _normalize_and_validate(
        getattr(django_profile, "conditions", None),
        CONDITION_ALIASES, MEDICAL_RULES, "conditions")
    allergies = _normalize_and_validate(
        getattr(django_profile, "allergies", None),
        ALLERGY_ALIASES, ALLERGY_RULES, "allergies")

    stored_prefs = _normalize_and_validate(
        getattr(django_profile, "preferences", None),
        PREFERENCE_ALIASES, PREFERENCE_BLOCKS, "preferences")
    preferences: List[str] = list(meal_type_prefs)
    for p in stored_prefs:
        if p not in preferences:
            preferences.append(p)

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
    )
