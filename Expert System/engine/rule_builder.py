"""
rule_builder.py — Smart Dietary Advisor v4.0 — FIXED
=====================================================
التغييرات:
    - strict_block وsoft_block منفصلان في المخرج
    - allergy_blocked مفتاح جديد للمكونات المحظورة من الحساسيات
    - soft_block لا يُضاف لـ all_blocked (يُمرر للتحذيرات فقط)
    - pregnancy تُضاف لـ cond_set لتفعيل combined_rules
"""

from typing import TYPE_CHECKING
from rules.medical_rules import MEDICAL_RULES
from rules.combined_rules import COMBINED_RULES
from rules.halal_and_allergies import ALLERGY_RULES, ALLERGY_COLUMN_MAP
from rules.goals_and_preferences import PREFERENCE_BLOCKS
from core.constants import DISEASE_EN, ALLERGY_EN, PREFERENCE_EN

if TYPE_CHECKING:
    from core.user_profile import UserProfile


def _merge_numeric(base: dict, incoming: dict) -> dict:
    result = dict(base)
    for col, rule in incoming.items():
        if not isinstance(rule, tuple) or len(rule) < 2:
            continue
        op, val = rule[0], rule[1]
        if col not in result:
            result[col] = (op, val)
        else:
            ex_op, ex_val = result[col]
            if op == "<=" and ex_op == "<=" and val < ex_val:
                result[col] = (op, val)
            elif op == ">=" and ex_op == ">=" and val > ex_val:
                result[col] = (op, val)
            else:
                result[col] = (op, val)
    return result


def get_applicable_rules(profile: "UserProfile") -> dict:
    final_numeric:    dict = {}
    strict_block:     list = []
    soft_block:       list = []
    preferred:        list = []
    allergy_blocked:  list = []
    warning_rules:    dict = {}
    min_requirements: dict = {}
    conflict_msgs:    list = []
    notes:            list = []
    allergy_filters:  list = []

    stage = profile.life_stage

    # ── 1. قواعد الفئة العمرية ──────────────────────────
    if profile.pregnant:
        r = MEDICAL_RULES["pregnancy"]
        final_numeric = _merge_numeric(final_numeric, r.get("numeric_rules", {}))
        strict_block  += r.get("strict_block", [])
        notes.append(f"[Pregnancy] {r.get('note', '')}")

    if stage == "child":
        r = MEDICAL_RULES["children"]
        final_numeric = _merge_numeric(final_numeric, r.get("numeric_rules", {}))
        strict_block  += r.get("strict_block", [])
        notes.append(f"[Children] {r.get('note', '')}")
    elif stage == "elderly":
        r = MEDICAL_RULES["elderly"]
        final_numeric = _merge_numeric(final_numeric, r.get("numeric_rules", {}))
        strict_block  += r.get("strict_block", [])
        notes.append(f"[Elderly] {r.get('note', '')}")

    # ── 2. الحالات الطبية ───────────────────────────────
    for cond in profile.conditions:
        if cond not in MEDICAL_RULES:
            continue
        r = MEDICAL_RULES[cond]
        final_numeric = _merge_numeric(final_numeric, r.get("numeric_rules", {}))
        strict_block  += r.get("strict_block", [])
        soft_block    += r.get("soft_block", [])
        preferred     += r.get("preferred_ingredients", [])
        warning_rules.update(r.get("warning_rules", {}))

        for col, (op, val) in r.get("min_requirements", {}).items():
            if col not in min_requirements:
                min_requirements[col] = (op, val)
            else:
                ex_op, ex_val = min_requirements[col]
                if op == ">=" and val > ex_val:
                    min_requirements[col] = (op, val)

        notes.append(f"[{DISEASE_EN.get(cond, cond)}] {r.get('note', '')}")

    # ── 3. الحساسيات ────────────────────────────────────
    for allergy in profile.allergies:
        col_name = ALLERGY_COLUMN_MAP.get(allergy)
        if col_name:
            allergy_filters.append(col_name)
        if allergy in ALLERGY_RULES:
            allergy_blocked += ALLERGY_RULES[allergy]["blocked_ingredients"]
            notes.append(
                f"[Allergy: {ALLERGY_EN.get(allergy, allergy)}] "
                f"{ALLERGY_RULES[allergy]['note']}"
            )
        allergy_key = f"{allergy}_allergy" if "_" not in allergy else allergy
        if allergy_key in MEDICAL_RULES:
            allergy_blocked += MEDICAL_RULES[allergy_key].get("strict_block", [])

    # ── 4. قواعد الدمج ──────────────────────────────────
    # ✅ FIX: أضفنا "pregnancy" لـ cond_set لو المستخدمة حامل
    cond_set = set(profile.conditions)
    if profile.pregnant:
        cond_set.add("pregnancy")

    for combo_key, combo_rule in COMBINED_RULES.items():
        if not set(combo_key).issubset(cond_set):
            continue
        for col, val in combo_rule.items():
            if col in ("conflict_warning", "note", "blocked_ingredients"):
                continue
            if isinstance(val, tuple):
                final_numeric = _merge_numeric(final_numeric, {col: val})
        if combo_rule.get("conflict_warning"):
            conflict_msgs.append(combo_rule["conflict_warning"])
        if "blocked_ingredients" in combo_rule:
            strict_block += combo_rule["blocked_ingredients"]

    # ── 5. التفضيلات ────────────────────────────────────
    for pref in profile.preferences:
        if pref not in PREFERENCE_BLOCKS:
            continue
        pb = PREFERENCE_BLOCKS[pref]
        strict_block += pb.get("blocked_ingredients", [])
        extra = pb.get("extra_numeric_rules", {})
        if extra:
            final_numeric = _merge_numeric(final_numeric, extra)
        notes.append(f"[Preference: {PREFERENCE_EN.get(pref, pref)}] {pb.get('note', '')}")

    # ── دمج min_requirements ─────────────────────────────
    for col, (op, val) in min_requirements.items():
        final_numeric = _merge_numeric(final_numeric, {col: (op, val)})

    return {
        "numeric_rules":         final_numeric,
        "strict_block":          list(set(strict_block)),
        "soft_block":            list(set(soft_block)),
        "allergy_blocked":       list(set(allergy_blocked)),
        "preferred_ingredients": list(set(preferred)),
        "warning_rules":         warning_rules,
        "min_requirements":      min_requirements,
        "conflict_messages":     conflict_msgs,
        "notes":                 notes,
        "allergy_filters":       allergy_filters,
    }