"""
halal_and_allergies.py — Smart Dietary Advisor v4.0
====================================================
فلتر الحلال العالمي + قواعد الحساسيات الغذائية الست.

يُطبَّق فلتر الحلال قبل أي قاعدة طبية أخرى ولا يمكن تجاوزه.
"""

from typing import Dict, List, Optional, Any

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
    "eggs":    None,   # لا يوجد عمود جاهز — يُستخدم البحث النصي
}

# ════════════════════════════════════════════════════════
# قواعد الحساسيات الغذائية — 6 أنواع
# ════════════════════════════════════════════════════════
ALLERGY_RULES: Dict[str, Dict[str, Any]] = {
    "peanuts": {
        "blocked_ingredients": [
            "peanut", "peanuts", "peanut butter", "peanut oil",
            "groundnut", "groundnuts",
        ],
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
        "blocked_ingredients": [
            "egg", "eggs", "egg white", "egg yolk",
            "mayonnaise", "meringue", "albumin",
        ],
        "note": "Common in children; many outgrow it",
    },
    "seafood": {
        "blocked_ingredients": [
            "fish", "salmon", "tuna", "shrimp", "crab",
            "lobster", "seafood", "clam", "oyster", "scallop",
        ],
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
}
