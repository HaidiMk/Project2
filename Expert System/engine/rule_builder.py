"""
rule_builder.py — Smart Dietary Advisor v4.0 — DECLARATIVE EDITION
=====================================================================
نسخة معاد كتابتها بالكامل بأسلوب Declarative — بدون أي:
    if / elif / else / for / while

الاستبدالات المستخدمة:
    for loop              → list/dict comprehension
    for + accumulate       → functools.reduce
    if/elif/else (تفرع)    → dict.get() / ternary / and-or short-circuit
    تحديث قاموس تراكمي     → دمج قواميس عبر reduce بدل حلقة + شرط

الفكرة المنطقية لكل قسم محفوظة 100% — فقط أسلوب التعبير تغيّر.
"""

from functools import reduce
from typing import Dict, List, Tuple, Any

from rules.medical_rules import MEDICAL_RULES
from rules.combined_rules import COMBINED_RULES
from rules.halal_and_allergies import ALLERGY_RULES, ALLERGY_COLUMN_MAP
from rules.goals_and_preferences import PREFERENCE_BLOCKS
from core.constants import DISEASE_EN, ALLERGY_EN, PREFERENCE_EN, DISEASES_FEMALE_ONLY

# ملاحظة: تم حذف "if TYPE_CHECKING:" بالكامل (كان فقط لتلميح النوع
# UserProfile بدون استيراد دائري) — استبدلناه بـ "UserProfile" كنص حرفي
# (forward reference) في التوقيعات أدناه؛ بايثون لا يقيّمه إلا عند الحاجة.


# ════════════════════════════════════════════════════════════════
# تناقضات فيزيولوجية مطلقة — لا يمكن وجود الحالتين معاً حقيقياً
# ════════════════════════════════════════════════════════════════
# 🛠️ إضافة (طلب صريح بناءً على فحص دقيق): بعض أزواج الحالات الطبية
# متضاربة فيزيولوجياً بشكل مطلق (لا "تعايش" حقيقي ممكن، خلافاً لكل
# أزواج COMBINED_RULES الأخرى التي تمثّل تعايشاً واقعياً مع تعارض
# علاجي قابل للحل). إدخال هذين معاً يعني خطأ بالمدخلات نفسها، لا
# حالة طبية مزدوجة حقيقية. النظام يتجاهل كلا الحالتين معاً (لا يفضّل
# واحدة عشوائياً) ويُصدر رسالة تحذير واضحة بدل تطبيق قواعد متناقضة
# بصمت.
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
    """
    يرجع مجموعة كل الحالات المتورطة بأي زوج مستحيل موجود معاً ضمن
    conditions — بدون for statement، فقط comprehension + any().
    """
    cond_set = set(conditions)
    active_pairs = [
        pair for pair in IMPOSSIBLE_CONDITION_PAIRS
        if pair[0] in cond_set and pair[1] in cond_set
    ]
    return set(sum((list(pair) for pair in active_pairs), []))


def _impossible_conflict_messages(conditions: List[str]) -> List[str]:
    """يبني رسائل التحذير لكل زوج مستحيل فعلياً موجود بـ conditions."""
    cond_set = set(conditions)
    active_pairs = [
        pair for pair in IMPOSSIBLE_CONDITION_PAIRS
        if pair[0] in cond_set and pair[1] in cond_set
    ]
    return [_IMPOSSIBLE_MESSAGES[pair] for pair in active_pairs]


# ════════════════════════════════════════════════════════════════
# 1. دمج قاعدتين رقميتين (الأشد يكسب) — بدون if/elif
# ════════════════════════════════════════════════════════════════

def _is_valid_rule(rule: Any) -> bool:
    """قاعدة صالحة = tuple بطول 2 على الأقل."""
    return isinstance(rule, tuple) and len(rule) >= 2


def _full_rule(rule: tuple) -> tuple:
    """
    🛠️ إصلاح bug خطير: كانت _merge_one تقصّ كل قاعدة لـ rule[:2] بدون
    استثناء، فتفقد قواعد "between" (3 عناصر: op, low, high) القيمة
    العليا بالكامل — ('between', 400, 550) تصبح ('between', 400)،
    فتُفسَّر لاحقاً بـ filtering_engine كقاعدة غير صالحة فتُتجاهل تماماً.
    الحل: نحافظ على الـtuple كاملة لـ between (3 عناصر)، ونقصّ
    لعنصرين فقط للعمليات الثنائية العادية (<=, >=, <, >).
    """
    is_between = rule[0] == "between"
    return rule if is_between else rule[:2]


def _between_intersection(a: tuple, b: tuple) -> tuple:
    """
    تقاطع نطاقين between — الأشد بينهما هو الأضيق (الحد الأدنى الأكبر،
    الحد الأعلى الأصغر). مطلوبة لو حالتان مختلفتان حدّدتا between على
    نفس العمود معاً (نادر لكن ممكن نظرياً عبر قواعد دمج متعددة).
    """
    _, a_low, a_high = a
    _, b_low, b_high = b
    return ("between", max(a_low, b_low), min(a_high, b_high))


def _stricter(existing: Tuple[str, float], incoming: Tuple[str, float]) -> Tuple[str, float]:
    """
    يرجّح القاعدة الأشد بين قاعدتين على نفس العمود.
    <=  → الأصغر أشد   |   >=  → الأكبر أشد   |   between → تفوز دائماً
    على قاعدة فردية (أحادية الحد)، لأنها محسوبة خصيصاً لتعارض حالتين
    معاً فهي أدق وأضيق بطبيعتها | بين قاعدتين between → التقاطع.

    🛠️ إصلاح: النسخة السابقة كانت تتحقق فقط من "كلتا القاعدتين
    between معاً" (both_between)، فإذا واحدة فقط منهما between
    (الحالة الأكثر شيوعاً: قاعدة فردية مثل diabetes تلتقي بقاعدة دمج
    مركّبة مثل diabetes+elderly)، تسقط بالـ else الأخير الذي يُرجّح
    "الجديد" افتراضياً بصرف النظر عن المنطق — هذا صحيح فقط لو الجديد
    (incoming) هو between، لكنه خاطئ تماماً لو العكس (existing
    between و incoming قاعدة فردية لاحقة تستبدلها بالغلط). الحل:
    نتحقق من between لكل طرف على حدة، وتفوز بشكل غير مشروط بالاتجاه.
    """
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
    """
    دمج عمود واحد (col, rule) داخل القاموس الأساسي — خطوة reduce واحدة.
    🛠️ إصلاح: نستخدم _full_rule(rule) بدل rule[:2] المباشرة — تحافظ
    على القيمة العليا لقواعد between بدل فقدانها.
    """
    col, rule = item
    full = _full_rule(rule) if _is_valid_rule(rule) else rule
    return (
        base if not _is_valid_rule(rule)
        else {**base, col: full} if col not in base
        else {**base, col: _stricter(base[col], full)}
    )


def _merge_numeric(base: dict, incoming: dict) -> dict:
    """نسخة declarative من الدمج — reduce بدل for+if."""
    return reduce(_merge_one, incoming.items(), dict(base))


def _merge_many_numeric(dicts: List[dict]) -> dict:
    """دمج أي عدد من قواميس numeric_rules بترتيب واحد تلو الآخر."""
    return reduce(_merge_numeric, dicts, {})


# ════════════════════════════════════════════════════════════════
# 2. مصادر القواعد — استبدال if pregnant / if stage==child/elderly
# ════════════════════════════════════════════════════════════════

def _life_stage_rule_keys(profile: "UserProfile") -> List[str]:
    """
    أي مفاتيح MEDICAL_RULES يجب تفعيلها بسبب الحمل أو الفئة العمرية.
    بدل if/elif: نبني قائمة من شروط (condition, key) ونفلترها.

    🛠️ حماية دفاعية إضافية: قاعدة "pregnancy" لا تُفعَّل إلا إذا
    gender=="female" أيضاً — حتى لو profile.pregnant=True بطريقة ما
    على بروفايل ذكر (مستحيل طبياً، لكن لا يوجد تحقق برمجي يمنعه عند
    استخدام UserProfile مباشرة بدون المرور بـ ui/cli_interface.py،
    التي تمنعه ظاهرياً فقط بمستوى الواجهة).
    """
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


# ════════════════════════════════════════════════════════════════
# 3. الحالات الطبية — استبدال for cond in profile.conditions
# ════════════════════════════════════════════════════════════════

def _female_only_conditions() -> set:
    """
    يجمع كل الحالات "نسائية فقط" من كل الفئات العمرية بقاموس واحد —
    يعادل تسطيح dict-of-lists لـ set موحّد، بدون for statement.
    """
    return set(sum(DISEASES_FEMALE_ONLY.values(), []))


def _known_conditions(conditions: List[str], gender: str = "female") -> List[str]:
    """
    فلترة الحالات المعروفة فقط (يعادل: if cond not in MEDICAL_RULES: continue)
    + فلترة دفاعية إضافية: تتجاهل أي حالة "نسائية فقط" (مثل PCOS) إذا
    الجنس المُمرَّر غير "female" — حتى لو تجاوز المستدعي طبقة الواجهة
    التفاعلية (ui/cli_interface.py) التي تمنع هذا السيناريو ظاهرياً
    فقط. هذا يضمن عدم تطبيق قاعدة "Females Only" على بروفايل ذكر
    حتى عند الاستخدام البرمجي المباشر لـ UserProfile/rule_builder
    (مثل اختبارات، أو وحدات أخرى بالمشروع كـ TOPSIS).

    🛠️ إضافة: تستبعد أيضاً أي حالة متورطة بزوج مستحيل فيزيولوجياً
    (IMPOSSIBLE_CONDITION_PAIRS) إذا كان الطرف الآخر موجوداً معها —
    لا نطبّق قواعد أي منهما، بدل تفضيل واحدة عشوائياً على الأخرى.
    """
    female_only = _female_only_conditions()
    is_female = gender == "female"
    impossible = _impossible_conditions_in(conditions)
    return [
        c for c in conditions
        if c in MEDICAL_RULES and (is_female or c not in female_only) and c not in impossible
    ]


def _min_requirements_for(rules_list: List[dict]) -> dict:
    """
    دمج كل min_requirements من كل الحالات الطبية، الأشد (>=) يكسب.
    استبدال للحلقة المتداخلة for cond -> for col,(op,val) in min_requirements.
    """
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


# ════════════════════════════════════════════════════════════════
# 4. الحساسيات — استبدال for allergy in profile.allergies
# ════════════════════════════════════════════════════════════════

def _allergy_key(allergy: str) -> str:
    """يبني اسم مفتاح MEDICAL_RULES المرتبط بالحساسية، بدون if."""
    return allergy if "_" in allergy else f"{allergy}_allergy"


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


# ════════════════════════════════════════════════════════════════
# 5. قواعد الدمج (COMBINED_RULES) — استبدال for combo_key,combo_rule
# ════════════════════════════════════════════════════════════════

_META_KEYS = ("conflict_warning", "note", "blocked_ingredients")


def _combo_applies(combo_key: tuple, cond_set: set) -> bool:
    return set(combo_key).issubset(cond_set)


def _combo_numeric(combo_rule: dict) -> dict:
    """يستخرج فقط أزواج (col, (op, val)) من قاعدة الدمج، متجاهلاً المفاتيح الوصفية."""
    return {
        col: val for col, val in combo_rule.items()
        if col not in _META_KEYS and isinstance(val, tuple)
    }


def _pregnancy_aware_conditions(profile: "UserProfile") -> set:
    """
    يبني cond_set مع إضافة pregnancy و life_stage عند الحاجة — بدون
    if إجرائي.

    🛠️ إصلاح بَج خطير ثانٍ (مكتشَف أثناء تتبّع مشكلة between): كانت
    تضيف فقط "pregnancy" لكنها تتجاهل تماماً "elderly"/"children"
    (life_stage) — يعني أي قاعدة دمج تتضمن هاتين الفئتين (مثل
    diabetes+elderly، elderly+heart_disease، chronic_kidney_disease+
    elderly، elderly+underweight — 4 قواعد كاملة) كانت غير قابلة
    للتطبيق إطلاقاً، لأن cond_set لا يحتوي "elderly" أصلاً رغم وجوده
    بـ profile.life_stage. الحل: نضيف stage إلى المجموعة، متّسقاً
    تماماً مع stage_key_map المستخدمة بـ _life_stage_rule_keys أعلاه
    (نفس المصدر، لا قيمة مكرَّرة يدوياً).
    """
    is_female = profile.gender == "female"
    pregnancy_extra = {"pregnancy"} if (profile.pregnant and is_female) else set()
    stage_key_map = {"child": "children", "elderly": "elderly"}
    stage_extra = {stage_key_map[profile.life_stage]} if profile.life_stage in stage_key_map else set()
    impossible = _impossible_conditions_in(profile.conditions)
    # 🛠️ نطرح الحالات المتورطة بزوج مستحيل هنا أيضاً — وإلا تبقى
    # قواعد الدمج (مثل obesity+diabetes) قابلة للتطبيق رغم استبعادها
    # من _conditions_block، لأن هذه الدالة تبني cond_set من
    # profile.conditions الخام مباشرة لا من _known_conditions.
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


# ════════════════════════════════════════════════════════════════
# 6. التفضيلات الغذائية — استبدال for pref in profile.preferences
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 7. التجميع النهائي — الدالة العامة المُصدَّرة
# ════════════════════════════════════════════════════════════════

def _combine_blocks(blocks: List[dict]) -> dict:
    """
    يدمج كل النتائج الجزئية (life_stage, conditions, allergies, combined, preferences)
    في قاموس واحد نهائي — بدون أي حلقة إجرائية، فقط reduce + comprehension.
    """
    numeric_merged = _merge_many_numeric([b.get("numeric_rules", {}) for b in blocks])
    # min_requirements تُدمج أخيراً ضمن numeric (تحل محل حلقة الدمج اليدوية بالأصل)
    min_reqs = reduce(lambda acc, b: {**acc, **b.get("min_requirements", {})}, blocks, {})
    numeric_with_min = _merge_numeric(numeric_merged, min_reqs)

    return {
        "numeric_rules": numeric_with_min,
        "strict_block": list(set(sum((b.get("strict_block", []) for b in blocks), []))),
        "soft_block": list(set(sum((b.get("soft_block", []) for b in blocks), []))),
        "allergy_blocked": list(set(sum((b.get("allergy_blocked", []) for b in blocks), []))),
        "preferred_ingredients": list(
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
