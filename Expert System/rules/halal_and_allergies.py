"""
halal_and_allergies.py — Smart Dietary Advisor v4.0
====================================================
فلتر الحلال العالمي + قواعد الحساسيات الغذائية الست.

يُطبَّق فلتر الحلال قبل أي قاعدة طبية أخرى ولا يمكن تجاوزه.
"""

from typing import Dict, List, Optional, Any

from rules.medical_rules import MEDICAL_RULES

# ════════════════════════════════════════════════════════
# فلتر الحلال — يُطبَّق أولاً، أولوية مطلقة
# ════════════════════════════════════════════════════════
HALAL_BLACKLIST: List[str] = [
    # لحم الخنزير ومشتقاته
    "pork", "ham", "bacon", "sausage", "pepperoni", "salami",
    "prosciutto", "lard", "gelatin", "pig", "swine",
    # الكحول بجميع أنواعه
    "wine", "beer", "alcohol", "vodka", "gin", "rum", "tequila",
    "whiskey", "whisky", "champagne", "liqueur", "brandy", "cognac",
    "sherry", "vermouth", "guinness", "stout", "lager", "ale",
    "cider", "sake", "soju", "bourbon", "mead", "absinthe",
]

# ════════════════════════════════════════════════════════
# ربط الحساسيات بأعمدة CSV
# ════════════════════════════════════════════════════════
ALLERGY_COLUMN_MAP: Dict[str, Optional[str]] = {
    "milk":    "HasLactose",
    "gluten":  "HasGluten",
    "peanuts": "HasNuts",
    "soy":     "HasSoy",
    "seafood": "HasSeafood",
    "eggs":    "HasEggs",   # 🛠️ مُحدَّث: كان None (بحث نصي فقط)؛ الآن
                            # مربوط بعمود HasEggs المُضاف عبر
                            # add_eggs_column.py — طبقة حماية مضاعفة
                            # (عمود + نص) مطابقة لباقي الحساسيات.
    "sesame":  None,        # 🆕 فئة جديدة (FDA 2023) — لا يوجد عمود
                            # CSV مخصص لها بعد، فالفحص نصي بس عبر
                            # blocked_ingredients. drop_allergy() بـ
                            # filtering_engine.py بيتعامل مع None/عمود
                            # غير موجود بأمان أصلاً (mask=True كامل).
}

# ════════════════════════════════════════════════════════
# قواعد الحساسيات الغذائية — 6 أنواع
# ════════════════════════════════════════════════════════
ALLERGY_RULES: Dict[str, Dict[str, Any]] = {
    "peanuts": {
        # 🛠️ مُصحَّح: كانت هذي القائمة تحتوي 6 كلمات فقط (peanut/
        # groundnut)، بدون أي تغطية للمكسرات الشجرية (Tree Nuts) رغم
        # إنه اسم الفئة بالواجهة هو "Peanuts / Tree Nuts". اكتُشف
        # بفحص فعلي: وصفات "Toasted Pine Nuts" و"Toasted Almond
        # Berry Bark" تسرّبت لمستخدم عنده هالحساسية بالضبط.
        # الحل: نعيد استخدام MEDICAL_RULES["nut_allergy"]["strict_block"]
        # (26 كلمة، موثّقة FARE + ACAAI) بدل تكرار قائمة ثانية، حتى لا
        # يتكرر نفس التعارض/النقص مستقبلاً بين الملفين.
        "blocked_ingredients": MEDICAL_RULES["nut_allergy"]["strict_block"],
        "note": "Anaphylaxis risk from trace amounts",
    },
    "milk": {
        "blocked_ingredients": [
            "milk", "cream", "butter", "cheese", "yogurt",
            "whey", "casein", "lactose", "dairy",
        ],
        "note": "Immune response — different from lactose intolerance",
    },
    "eggs": {
        # 🛠️ مُصحَّح: نفس نمط bug المكسرات — القائمة القديمة هون كانت
        # أقصر من MEDICAL_RULES["egg_allergy"]["strict_block"] (ناقصة
        # مثلاً egg powder, dried egg, surimi...). نوحّد المصدر لتفادي
        # نفس فجوة nut_allergy.
        "blocked_ingredients": MEDICAL_RULES["egg_allergy"]["strict_block"],
        "note": "Common in children; many outgrow it",
    },
    "seafood": {
        # 🛠️ مُصحَّح: نفس النمط — القائمة القديمة كانت ناقصة أنواع
        # سمك كتير (cod, tilapia, halibut, anchovy...) موجودة أصلاً
        # بـ MEDICAL_RULES["seafood_allergy"]["strict_block"].
        "blocked_ingredients": MEDICAL_RULES["seafood_allergy"]["strict_block"],
        "note": "Both fish and shellfish excluded",
    },
    "soy": {
        "blocked_ingredients": [
            "soy", "soya", "soybeans", "tofu", "tempeh",
            "miso", "edamame", "soy sauce", "soy protein",
        ],
        "note": "Soy hidden in many processed foods — read labels carefully",
    },
    "gluten": {
        "blocked_ingredients": [
            "wheat", "barley", "rye", "oats", "pasta",
            "bread", "flour", "semolina", "malt",
        ],
        "note": "Matches celiac exclusion list",
    },
    "sesame": {
        # 🆕 مضافة حديثاً: FDA اعتبرت السمسم المسبب الرئيسي التاسع
        # للحساسية الغذائية اعتباراً من 1 يناير 2023 (FASTER Act) —
        # ما كان مغطّى إطلاقاً بالنظام قبل هيك.
        "blocked_ingredients": MEDICAL_RULES["sesame_allergy"]["strict_block"],
        "note": "FDA's 9th major allergen (2023) — hidden often in buns, sauces, hummus",
    },
}
