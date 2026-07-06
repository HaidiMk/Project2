

from functools import reduce
from typing import Dict, List, Tuple, Any

from rules.medical_rules import MEDICAL_RULES
from rules.combined_rules import COMBINED_RULES
from rules.halal_and_allergies import ALLERGY_RULES, ALLERGY_COLUMN_MAP
from rules.goals_and_preferences import PREFERENCE_BLOCKS
from core.constants import DISEASE_EN, ALLERGY_EN, PREFERENCE_EN, DISEASES_FEMALE_ONLY


IMPOSSIBLE_CONDITION_PAIRS = [
    ("obesity", "underweight"),
    ("hypothyroidism", "hyperthyroidism"),
]

_IMPOSSIBLE_MESSAGES = {
    ("obesity", "underweight"): (
        "⚠️ Contradictory input: 'Obesity' and 'Underweight' cannot "
        "both apply (opposite BMI categories). Both conditions ignored "
        "— please verify your selection."
    ),
    ("hypothyroidism", "hyperthyroidism"): (
        "⚠️ Contradictory input: 'Hypothyroidism' and 'Hyperthyroidism' "
        "cannot both apply (opposite thyroid states). Both conditions "
        "ignored — please verify your selection."
    ),
}


def _impossible_conditions_in(conditions: List[str]) -> set:
   
    cond_set = set(conditions)
    active_pairs = [
        pair for pair in IMPOSSIBLE_CONDITION_PAIRS
        if pair[0] in cond_set and pair[1] in cond_set
    ]
    return set(sum((list(pair) for pair in active_pairs), []))


def _impossible_conflict_messages(conditions: List[str]) -> List[str]:
    
    cond_set = set(conditions)
    active_pairs = [
        pair for pair in IMPOSSIBLE_CONDITION_PAIRS
        if pair[0] in cond_set and pair[1] in cond_set
    ]
    return [_IMPOSSIBLE_MESSAGES[pair] for pair in active_pairs]




def _is_valid_rule(rule: Any) -> bool:
   
    return isinstance(rule, tuple) and len(rule) >= 2


def _full_rule(rule: tuple) -> tuple:
  
    is_between = rule[0] == "between"
    return rule if is_between else rule[:2]


def _between_intersection(a: tuple, b: tuple) -> tuple:
    
    _, a_low, a_high = a
    _, b_low, b_high = b
    return ("between", max(a_low, b_low), min(a_high, b_high))


def _stricter(existing: Tuple[str, float], incoming: Tuple[str, float]) -> Tuple[str, float]:
  
    both_between = existing[0] == "between" and incoming[0] == "between"
    only_existing_between = existing[0] == "between" and not both_between
    only_incoming_between = incoming[0] == "between" and not both_between
    if_between_result = _between_intersection(existing, incoming) if both_between else None

    ex_op, ex_val = existing[0], existing[1]
    in_op, in_val = incoming[0], incoming[1]

    same_le = ex_op == "<=" and in_op == "<=" and in_val < ex_val
    same_ge = ex_op == ">=" and in_op == ">=" and in_val > ex_val

    return (
        if_between_result if both_between
        else existing if only_existing_between
        else incoming if only_incoming_between
        else incoming if same_le
        else existing if (ex_op == "<=" and in_op == "<=")
        else incoming if same_ge
        else existing if (ex_op == ">=" and in_op == ">=")
        else incoming
    )


def _merge_one(base: dict, item: Tuple[str, Any]) -> dict:
   
    col, rule = item
    full = _full_rule(rule) if _is_valid_rule(rule) else rule
    return (
        base if not _is_valid_rule(rule)
        else {**base, col: full} if col not in base
        else {**base, col: _stricter(base[col], full)}
    )


def _merge_numeric(base: dict, incoming: dict) -> dict:
   
    return reduce(_merge_one, incoming.items(), dict(base))


def _merge_many_numeric(dicts: List[dict]) -> dict:
    
    return reduce(_merge_numeric, dicts, {})




def _life_stage_rule_keys(profile: "UserProfile") -> List[str]:
    
    stage_key_map = {"child": "children", "elderly": "elderly"}
    is_female = profile.gender == "female"
    candidates = [
        ("pregnancy" if (profile.pregnant and is_female) else None),
        stage_key_map.get(profile.life_stage),
    ]
    return [k for k in candidates if k]


def _note_for(label: str, rule: dict) -> str:
    return f"[{label}] {rule.get('note', '')}"


def _life_stage_block(profile: "UserProfile") -> dict:
    """numeric + strict_block + notes الناتجة عن الحمل/الفئة العمرية."""
    keys = _life_stage_rule_keys(profile)
    rules = [MEDICAL_RULES[k] for k in keys]
    labels = {"pregnancy": "Pregnancy", "children": "Children", "elderly": "Elderly"}

    return {
        "numeric_rules": _merge_many_numeric([r.get("numeric_rules", {}) for r in rules]),
        "strict_block": sum((r.get("strict_block", []) for r in rules), []),
        "notes": [_note_for(labels[k], MEDICAL_RULES[k]) for k in keys],
    }



def _female_only_conditions() -> set:
    
    return set(sum(DISEASES_FEMALE_ONLY.values(), []))


def _known_conditions(conditions: List[str], gender: str = "female") -> List[str]:
   
    female_only = _female_only_conditions()
    is_female = gender == "female"
    impossible = _impossible_conditions_in(conditions)
    return [
        c for c in conditions
        if c in MEDICAL_RULES and (is_female or c not in female_only) and c not in impossible
    ]


def _min_requirements_for(rules_list: List[dict]) -> dict:
   
    all_pairs = sum(
        ([(col, val) for col, val in r.get("min_requirements", {}).items()]
         for r in rules_list),
        [],
    )
    return reduce(_merge_one, all_pairs, {})


def _conditions_block(profile: "UserProfile") -> dict:
    """numeric + strict/soft_block + preferred + warnings + notes من الحالات الطبية."""
    conds = _known_conditions(profile.conditions, profile.gender)
    rules_list = [MEDICAL_RULES[c] for c in conds]

    return {
        "numeric_rules": _merge_many_numeric(
            [r.get("numeric_rules", {}) for r in rules_list]
        ),
        "strict_block": sum((r.get("strict_block", []) for r in rules_list), []),
        "soft_block": sum((r.get("soft_block", []) for r in rules_list), []),
        "preferred_ingredients": sum(
            (r.get("preferred_ingredients", []) for r in rules_list), []
        ),
        "warning_rules": reduce(
            lambda acc, r: {**acc, **r.get("warning_rules", {})}, rules_list, {}
        ),
        "min_requirements": _min_requirements_for(rules_list),
        "notes": [_note_for(DISEASE_EN.get(c, c), MEDICAL_RULES[c]) for c in conds],
        "conflict_messages": _impossible_conflict_messages(profile.conditions),
    }




_ALLERGY_TO_MEDICAL_KEY = {
    "peanuts": "nut_allergy",
    "eggs":    "egg_allergy",
    "seafood": "seafood_allergy",
    "sesame":  "sesame_allergy",
}


def _allergy_key(allergy: str) -> str:
   
    return _ALLERGY_TO_MEDICAL_KEY.get(allergy, f"{allergy}_allergy")


def _allergy_filter_column(allergy: str) -> List[str]:
    col = ALLERGY_COLUMN_MAP.get(allergy)
    return [col] if col else []


def _allergy_blocked_ingredients(allergy: str) -> List[str]:
    from_rules = ALLERGY_RULES.get(allergy, {}).get("blocked_ingredients", [])
    key = _allergy_key(allergy)
    from_medical = MEDICAL_RULES.get(key, {}).get("strict_block", [])
    return from_rules + from_medical


def _allergy_note(allergy: str) -> List[str]:
    rule = ALLERGY_RULES.get(allergy)
    return (
        [f"[Allergy: {ALLERGY_EN.get(allergy, allergy)}] {rule['note']}"]
        if rule else []
    )


def _allergies_block(allergies: List[str]) -> dict:
    return {
        "allergy_filters": sum((_allergy_filter_column(a) for a in allergies), []),
        "allergy_blocked": sum((_allergy_blocked_ingredients(a) for a in allergies), []),
        "notes": sum((_allergy_note(a) for a in allergies), []),
    }



_META_KEYS = ("conflict_warning", "note", "blocked_ingredients")


def _combo_applies(combo_key: tuple, cond_set: set) -> bool:
    return set(combo_key).issubset(cond_set)


def _combo_numeric(combo_rule: dict) -> dict:
   
    return {
        col: val for col, val in combo_rule.items()
        if col not in _META_KEYS and isinstance(val, tuple)
    }


def _pregnancy_aware_conditions(profile: "UserProfile") -> set:
    
    is_female = profile.gender == "female"
    pregnancy_extra = {"pregnancy"} if (profile.pregnant and is_female) else set()
    stage_key_map = {"child": "children", "elderly": "elderly"}
    stage_extra = {stage_key_map[profile.life_stage]} if profile.life_stage in stage_key_map else set()
    impossible = _impossible_conditions_in(profile.conditions)
    
    base_conditions = set(profile.conditions) - impossible
    return base_conditions | pregnancy_extra | stage_extra


def _combined_rules_block(profile: "UserProfile") -> dict:
    cond_set = _pregnancy_aware_conditions(profile)
    active = [
        (key, rule) for key, rule in COMBINED_RULES.items()
        if _combo_applies(key, cond_set)
    ]
    active_rules = [rule for _, rule in active]

    return {
        "numeric_rules": _merge_many_numeric(
            [_combo_numeric(rule) for rule in active_rules]
        ),
        "strict_block": sum(
            (rule.get("blocked_ingredients", []) for rule in active_rules), []
        ),
        "conflict_messages": [
            rule["conflict_warning"] for rule in active_rules
            if rule.get("conflict_warning")
        ],
    }



def _known_preferences(preferences: List[str]) -> List[str]:
    return [p for p in preferences if p in PREFERENCE_BLOCKS]

def _preferences_block(preferences: List[str]) -> dict:
    prefs = _known_preferences(preferences)
    blocks = [PREFERENCE_BLOCKS[p] for p in prefs]

    return {
        "numeric_rules": _merge_many_numeric(
            [pb.get("extra_numeric_rules", {}) for pb in blocks]
        ),
        "strict_block": sum((pb.get("blocked_ingredients", []) for pb in blocks), []),
        "notes": [
            f"[Preference: {PREFERENCE_EN.get(p, p)}] {pb.get('note', '')}"
            for p, pb in zip(prefs, blocks)
        ],
    }




def _combine_blocks(blocks: List[dict]) -> dict:
    """
    يدمج كل النتائج الجزئية (life_stage, conditions, allergies, combined, preferences)
    في قاموس واحد نهائي — بدون أي حلقة إجرائية، فقط reduce + comprehension.
    """
    numeric_merged = _merge_many_numeric([b.get("numeric_rules", {}) for b in blocks])
    
    min_reqs = reduce(lambda acc, b: {**acc, **b.get("min_requirements", {})}, blocks, {})
    numeric_with_min = _merge_numeric(numeric_merged, min_reqs)

    return {
        "numeric_rules": numeric_with_min,
       
        "strict_block": sorted(set(sum((b.get("strict_block", []) for b in blocks), []))),
        "soft_block": sorted(set(sum((b.get("soft_block", []) for b in blocks), []))),
        "allergy_blocked": sorted(set(sum((b.get("allergy_blocked", []) for b in blocks), []))),
        "preferred_ingredients": sorted(
            set(sum((b.get("preferred_ingredients", []) for b in blocks), []))
        ),
        "warning_rules": reduce(lambda acc, b: {**acc, **b.get("warning_rules", {})}, blocks, {}),
        "min_requirements": min_reqs,
        "conflict_messages": sum((b.get("conflict_messages", []) for b in blocks), []),
        "notes": sum((b.get("notes", []) for b in blocks), []),
        "allergy_filters": sum((b.get("allergy_filters", []) for b in blocks), []),
    }


def get_applicable_rules(profile: "UserProfile") -> dict:
    """
    نقطة الدخول العامة — نفس التوقيع والمخرجات تماماً كالنسخة الأصلية،
    لكن بدون أي if/elif/for/while في كامل مسار التنفيذ.
    """
    blocks = [
        _life_stage_block(profile),
        _conditions_block(profile),
        _allergies_block(profile.allergies),
        _combined_rules_block(profile),
        _preferences_block(profile.preferences),
    ]
    return _combine_blocks(blocks)
