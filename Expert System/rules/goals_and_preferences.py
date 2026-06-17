"""
goals_and_preferences.py — Smart Dietary Advisor v4.0 — FIXED
==============================================================
التغييرات:
    - vegetarian: أضفنا 30+ نوع سمك ولحم مفقود
    - vegan: أضفنا المفقودين أيضاً
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.user_profile import UserProfile

GOAL_VECTORS: Dict[str, Dict[str, Any]] = {
    "weight_loss": {
        "score_weights": {
            "Calories":       -0.08,
            "ProteinContent":  0.75,
            "SugarContent":   -0.50,
            "FatContent":     -0.30,
            "FiberContent":    0.40,
        },
        "target_calorie_offset": -150,
        "source": "AHA/ACC/TOS 2013 + DGA 2025",
    },
    "weight_gain": {
        "score_weights": {
            "Calories":            0.90,
            "ProteinContent":      0.90,
            "FatContent":          0.60,
            "CarbohydrateContent": 0.60,
        },
        "target_calorie_offset": 100,
        "source": "ACSM + AND",
    },
    "muscle_gain": {
        "score_weights": {
            "ProteinContent":      0.98,
            "Calories":            0.50,
            "CarbohydrateContent": 0.50,
            "FatContent":         -0.20,
        },
        "target_calorie_offset": 80,
        "source": "ISSN Position Stand 2023",
    },
    "maintenance": {
        "score_weights": {
            "ProteinContent": 0.60,
            "FiberContent":   0.50,
            "SugarContent":  -0.30,
        },
        "target_calorie_offset": 0,
        "source": "DGA 2025-2030",
    },
    "heart_health": {
        "score_weights": {
            "SodiumContent":       -0.70,
            "SaturatedFatContent": -0.80,
            "CholesterolContent":  -0.60,
            "ProteinContent":       0.40,
            "FiberContent":         0.50,
        },
        "target_calorie_offset": -50,
        "source": "AHA/ACC 2025",
    },
    "pregnancy_diet": {
        "score_weights": {
            "ProteinContent": 0.90,
            "Calories":       0.50,
            "FiberContent":   0.40,
            "SugarContent":  -0.30,
        },
        "target_calorie_offset": 50,
        "source": "ACOG 2021",
    },
    "elderly_diet": {
        "score_weights": {
            "ProteinContent": 0.90,
            "Calories":       0.30,
            "SodiumContent": -0.50,
            "FiberContent":   0.40,
        },
        "target_calorie_offset": -30,
        "source": "ESPEN 2019 + NIA",
    },
    "healthy_eating": {
        "score_weights": {
            "FiberContent":   0.70,
            "ProteinContent": 0.60,
            "Calories":       0.40,
            "SugarContent":  -0.30,
            "SodiumContent": -0.30,
        },
        "target_calorie_offset": 0,
        "source": "DGA 2025-2030 MyPlate",
    },
}

# ══════════════════════════════════════════════════════════════
# التفضيلات الغذائية — قائمة ممنوعات شاملة ومصححة
# ══════════════════════════════════════════════════════════════

# قائمة شاملة لجميع أنواع اللحوم والدواجن
_ALL_MEATS = [
    # لحوم حمراء
    "beef", "pork", "lamb", "veal", "goat", "bison", "buffalo",
    "venison", "elk", "deer", "boar", "rabbit", "hare",
    "steak", "roast beef", "ground beef", "ground meat",
    "meatball", "meatloaf", "ribs", "chop",
    # دواجن
    "chicken", "turkey", "duck", "goose", "quail", "pheasant",
    "cornish hen", "game hen",
    # مشتقات
    "hot dog", "sausage", "pepperoni", "salami", "bologna",
    "deli meat", "cold cuts", "ham",
    "meat broth", "chicken broth", "beef broth", "bone broth",
    "chicken stock", "beef stock",
    "lard", "tallow", "suet",
]

# قائمة شاملة لجميع أنواع الأسماك والمأكولات البحرية
_ALL_SEAFOOD = [
    # أسماك
    "fish", "salmon", "tuna", "cod", "tilapia", "halibut",
    "pollock", "flounder", "sole", "bass", "trout", "catfish",
    "snapper", "grouper", "mahi", "mahi-mahi", "swordfish",
    "sardine", "anchovy", "herring", "mackerel", "pike",
    "walleye", "perch", "carp", "eel", "monkfish", "haddock",
    "whitefish", "rockfish", "lingcod", "yellowtail", "amberjack",
    "barramundi", "branzino", "turbot", "plaice", "dace",
    # مأكولات بحرية
    "shrimp", "prawn", "crab", "lobster", "clam", "oyster",
    "scallop", "mussel", "squid", "octopus", "cuttlefish",
    "crawfish", "crayfish", "langoustine",
    "seafood", "shellfish",
    # صلصات ومشتقات
    "fish sauce", "oyster sauce", "worcestershire sauce",
    "anchovy paste", "fish paste", "shrimp paste",
    "fish stock", "clam juice",
    "caviar", "roe", "surimi", "imitation crab",
]

PREFERENCE_BLOCKS: Dict[str, Dict[str, Any]] = {

    "vegetarian": {
        "blocked_ingredients": _ALL_MEATS + _ALL_SEAFOOD,
        "note": "No meat, poultry, or seafood",
    },

    "vegan": {
        "blocked_ingredients": _ALL_MEATS + _ALL_SEAFOOD + [
            "egg", "eggs", "egg white", "egg yolk", "whole egg",
            "milk", "whole milk", "skim milk", "cream", "heavy cream",
            "butter", "ghee", "cheese", "cheddar", "mozzarella",
            "parmesan", "ricotta", "cottage cheese", "cream cheese",
            "yogurt", "kefir", "whipped cream", "ice cream",
            "honey", "gelatin", "lard", "tallow",
            "whey", "casein", "dairy", "lactose",
            "mayonnaise", "meringue",
        ],
        "note": "No animal products whatsoever",
    },

    "seafood_lover": {
        "blocked_ingredients": _ALL_MEATS,
        "note": "Red meat and poultry removed — seafood allowed",
    },

    "meat_lover": {
        "blocked_ingredients": [],
        "note": "No additional exclusions",
    },

    "chicken_lover": {
        "blocked_ingredients": [
            "beef", "pork", "lamb", "veal", "goat", "bison",
            "venison", "elk", "deer", "steak", "roast beef",
        ] + _ALL_SEAFOOD,
        "note": "Only chicken and poultry allowed",
    },

    "low_carb": {
        "blocked_ingredients": [
            "bread", "pasta", "rice", "oats", "oatmeal",
            "sugar", "candy", "cake", "cookie", "muffin",
            "potato", "corn", "flour", "tortilla", "cereal",
            "bagel", "croissant", "waffle", "pancake",
        ],
        "extra_numeric_rules": {
            "CarbohydrateContent": ("<=", 30),
            "SugarContent":        ("<=", 5),
        },
        "note": "< 30 g carbs/meal",
    },

    "mediterranean": {
        "blocked_ingredients": [
            "lard", "shortening", "hydrogenated oil",
            "trans fat", "margarine",
            "fast food", "processed meat",
            "soda", "cola", "energy drink",
        ],
        "note": "Olive oil, vegetables, fish, whole grains",
    },

    "no_preference": {
        "blocked_ingredients": [],
        "note": "All foods available",
    },
}

# ══════════════════════════════════════════════════════════════
# خطط الأكل الصحي حسب BMI
# ══════════════════════════════════════════════════════════════
HEALTHY_DIET_STYLES: Dict[str, Dict[str, str]] = {
    "normal_bmi": {
        "name": "Balanced Nutrition Plan",
        "description": (
            "Your weight is ideal! Recommended approach:\n"
            "  - 50% complex carbohydrates (whole grains, legumes)\n"
            "  - 25% plant and animal protein\n"
            "  - 25% vegetables, fruits, and healthy oils\n"
            "  - 2-3 liters of water daily"
        ),
        "source": "MyPlate USDA + DGA 2025-2030",
    },
    "underweight_style": {
        "name": "Healthy Weight Gain Plan",
        "description": (
            "Your weight is below ideal. Recommended:\n"
            "  - 300-500 kcal surplus daily\n"
            "  - 5-6 small meals instead of 3 large ones\n"
            "  - High-quality protein (meat, eggs, nuts, legumes)\n"
            "  - Healthy calorie-dense fats (avocado, nuts, olive oil)"
        ),
        "source": "AND + ACSM Guidelines",
    },
    "overweight_style": {
        "name": "Weight Management Plan",
        "description": (
            "Your weight is slightly above ideal. Recommended:\n"
            "  - 250-500 kcal deficit daily\n"
            "  - Reduce sugar and saturated fats\n"
            "  - Increase vegetable intake\n"
            "  - Light physical activity 30 min/day"
        ),
        "source": "AHA/ACC/TOS Obesity Guidelines",
    },
    "obese_style": {
        "name": "Medical Weight Management Plan",
        "description": (
            "BMI indicates obesity. Recommended:\n"
            "  - Consult a doctor or registered dietitian\n"
            "  - 500-750 kcal deficit daily\n"
            "  - Reduce refined carbs and sugars significantly\n"
            "  - High protein to preserve muscle mass\n"
            "  - Regular moderate physical activity"
        ),
        "source": "AHA/ACC/TOS 2013 + DGA 2025-2030",
    },
}


def get_healthy_diet_style(profile: "UserProfile") -> Dict[str, str]:
    bmi = profile.bmi
    if bmi < 18.5:   key = "underweight_style"
    elif bmi < 25.0: key = "normal_bmi"
    elif bmi < 30.0: key = "overweight_style"
    else:            key = "obese_style"
    return HEALTHY_DIET_STYLES[key]
