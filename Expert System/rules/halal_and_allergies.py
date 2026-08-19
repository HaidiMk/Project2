from typing import Dict, List, Optional, Any

from rules.medical_rules import MEDICAL_RULES

HALAL_BLACKLIST: List[str] = [
    "pork", "ham", "bacon", "sausage", "pepperoni", "salami",
    "prosciutto", "lard", "gelatin", "pig", "swine",
    "wine", "beer", "alcohol", "vodka", "gin", "rum", "tequila",
    "whiskey", "whisky", "champagne", "liqueur", "brandy", "cognac",
    "sherry", "vermouth", "guinness", "stout", "lager", "ale",
    "cider", "sake", "soju", "bourbon", "mead", "absinthe",
]

ALLERGY_COLUMN_MAP: Dict[str, Optional[str]] = {
    "milk":    "HasLactose",
    "gluten":  "HasGluten",
    "peanuts": "HasNuts",
    "soy":     "HasSoy",
    "seafood": "HasSeafood",
    "eggs":    "HasEggs", 
    "sesame":  None,     
}

ALLERGY_RULES: Dict[str, Dict[str, Any]] = {
    "peanuts": {
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
        "blocked_ingredients": MEDICAL_RULES["egg_allergy"]["strict_block"],
        "note": "Common in children; many outgrow it",
    },
    "seafood": {
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
        "blocked_ingredients": MEDICAL_RULES["sesame_allergy"]["strict_block"],
        "note": "FDA's 9th major allergen (2023) — hidden often in buns, sauces, hummus",
    },
}
