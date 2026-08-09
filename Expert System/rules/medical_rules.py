"""
medical_rules.py — Smart Dietary Advisor v4.0
==============================================
قواعد نظام الخبير الطبية — 24 حالة صحية
المصادر: ADA 2025, AHA/ACC 2025, NKF/KDIGO 2024, ACG 2023, WHO, ESPEN...

هيكل كل حالة:
    numeric_rules        → حدود رقمية صارمة (تمنع الوصفة عند التجاوز)
    warning_rules        → حدود تحذيرية (لا تمنع الوصفة)
    strict_block         → مكونات ممنوعة كلياً
    soft_block           → مكونات يُنصح بتجنبها
    preferred_ingredients→ مكونات تزيد درجة التقييم
    min_requirements     → حدود دنيا إجبارية
    max_servings         → حد أقصى للحصص
    source               → المصدر الطبي
    note                 → ملاحظة سريرية
"""

from typing import Dict, Any

MEDICAL_RULES: Dict[str, Dict[str, Any]] = {

    # ── السكري — ADA Standards of Care 2025 ───────────────
    "diabetes": {
        "numeric_rules": {
            "CarbohydrateContent": ("<=", 60),
            "SugarContent":        ("<=", 15),
            "FiberContent":        (">=", 5),
            "Calories":            ("<=", 600),
        },
        "warning_rules": {
            "SugarContent":        (">=", 10),
            "CarbohydrateContent": (">=", 45),
        },
        "strict_block": [
            "sugar", "white sugar", "brown sugar", "powdered sugar",
            "corn syrup", "high fructose corn syrup", "maple syrup",
            "honey", "molasses", "agave",
            "fruit juice concentrate", "apple juice", "grape juice",
        ],
        "soft_block": [
            "white rice", "white bread", "white pasta",
            "instant oatmeal", "potato", "mashed potato",
        ],
        "preferred_ingredients": [
            "oats", "whole grain", "brown rice", "quinoa",
            "lentils", "chickpeas", "beans", "broccoli",
            "spinach", "kale", "cucumber", "zucchini",
            "chicken breast", "turkey", "fish", "tofu",
            "olive oil", "avocado", "chia seeds", "flaxseed",
        ],
        "min_requirements": {"FiberContent": (">=", 5)},
        "max_servings": 1,
        "source": "ADA Standards of Care 2025 — professional.diabetes.org",
        "note":   "Carbohydrate Counting Method — 3–4 servings (45–60 g) per main meal",
    },

    # ── ارتفاع ضغط الدم — DASH / NHLBI / WHO ─────────────
    "hypertension": {
        "numeric_rules": {
            "SodiumContent":       ("<=", 600),
            "SaturatedFatContent": ("<=", 5),
            "CholesterolContent":  ("<=", 100),
        },
        "warning_rules": {
            "SodiumContent":       (">=", 450),
            "FatContent":          (">=", 20),
            "SaturatedFatContent": (">=", 3),
        },
        "strict_block": [
            "salt", "table salt", "sea salt", "kosher salt",
            "soy sauce", "fish sauce", "worcestershire sauce",
            "pickle", "pickled", "canned soup", "bouillon",
            "lard", "shortening", "palm oil",
        ],
        "soft_block": [
            "cheese", "processed cheese", "deli meat",
            "ham", "hot dog",
            "ketchup", "mustard", "bbq sauce",
        ],
        "preferred_ingredients": [
            "banana", "orange", "berries", "apple",
            "spinach", "broccoli", "sweet potato", "tomato",
            "oats", "whole grain", "brown rice",
            "low-fat milk", "low-fat yogurt",
            "salmon", "tuna", "chicken breast",
            "walnuts", "almonds", "flaxseed",
            "olive oil", "avocado",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "DASH Diet — NHLBI + WHO 2023 + StatPearls NBK482514",
        "note":   "1500 mg sodium/day for hypertension; DASH diet priority",
    },

    # ── أمراض القلب — AHA/ACC Guidelines 2025 ─────────────
    "heart_disease": {
        "numeric_rules": {
            "SodiumContent":       ("<=", 500),
            "SaturatedFatContent": ("<=", 4),
            "CholesterolContent":  ("<=", 67),
            "FatContent":          ("<=", 20),
            "SugarContent":        ("<=", 10),
        },
        "warning_rules": {
            "SaturatedFatContent": (">=", 3),
            "CholesterolContent":  (">=", 50),
            "FatContent":          (">=", 15),
            "SugarContent":        (">=", 7),
        },
        "strict_block": [
            "partially hydrogenated oil", "trans fat",
            "shortening", "margarine",
            "hot dog", "bologna",
            "lard", "palm oil",
        ],
        "soft_block": [
            "butter", "full-fat cheese", "whole milk",
            "red meat", "egg yolk", "coconut oil",
            "salt", "pickle",
        ],
        "preferred_ingredients": [
            "salmon", "mackerel", "sardine", "tuna",
            "oats", "barley", "flaxseed", "chia seeds",
            "blueberries", "strawberries", "pomegranate",
            "spinach", "kale", "broccoli", "tomato",
            "olive oil", "avocado", "walnuts",
            "garlic", "onion",
            "lentils", "beans", "chickpeas", "tofu",
        ],
        "min_requirements": {"FiberContent": (">=", 4)},
        "max_servings": 1,
        "source": "AHA/ACC Cardiovascular Guidelines 2025",
        "note":   "Stricter than hypertension — elevated cardiac event risk",
    },

    # ── الكلى المزمنة — NKF / KDIGO 2024 ──────────────────
    "chronic_kidney_disease": {
        "numeric_rules": {
            "SodiumContent":      ("<=", 500),
            "ProteinContent":     ("<=", 20),
            "CholesterolContent": ("<=", 67),
            "FatContent":         ("<=", 20),
        },
        "warning_rules": {
            "SodiumContent":  (">=", 350),
            "ProteinContent": (">=", 15),
            "FatContent":     (">=", 15),
        },
        "strict_block": [
            "banana", "orange juice", "tomato juice",
            "potato", "sweet potato",
            "dairy", "whole milk", "cola", "dark soda",
            "chocolate", "nuts", "peanut butter",
            "salt substitute",
            "canned food", "processed meat",
        ],
        "soft_block": [
            "red meat", "cheese", "whole grain",
            "beans", "lentils",
        ],
        "preferred_ingredients": [
            "cabbage", "cauliflower", "green beans", "cucumber",
            "apple", "berries", "grapes", "pineapple",
            "white rice", "white bread", "egg white",
            "chicken breast", "fish",
            "olive oil",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "National Kidney Foundation + KDIGO Guidelines 2024",
        "note":   "Low protein slows CKD progression; restrict K, P, Na",
    },

    # ── السمنة — AHA/ACC/TOS + DGA 2025 ───────────────────
    "obesity": {
        "numeric_rules": {
            "Calories":            ("<=", 400),
            "SugarContent":        ("<=", 10),
            "SaturatedFatContent": ("<=", 5),
            "FatContent":          ("<=", 15),
        },
        "warning_rules": {
            "Calories":   (">=", 350),
            "FatContent": (">=", 12),
            "SugarContent": (">=", 7),
        },
        "strict_block": [
            "deep fried", "fried chicken", "french fries",
            "chips", "potato chips",
            "candy", "chocolate", "cake", "doughnut",
            "soda", "cola", "energy drink", "fruit punch",
            "ice cream",
        ],
        "soft_block": [
            "white bread", "white rice", "pasta",
            "butter", "mayonnaise", "cream", "juice",
        ],
        "preferred_ingredients": [
            "broccoli", "spinach", "zucchini", "cucumber",
            "lettuce", "celery", "cabbage", "mushroom",
            "chicken breast", "turkey breast", "egg white",
            "lentils", "chickpeas",
            "oats", "quinoa",
            "apple", "berries", "grapefruit",
            "Greek yogurt", "cottage cheese",
        ],
        "min_requirements": {
            "ProteinContent": (">=", 15),
            "FiberContent":   (">=", 4),
        },
        "max_servings": 1,
        "source": "AHA/ACC/TOS 2013 + Joslin 2018 + DGA 2025-2030",
        "note":   "500–750 kcal deficit/day → 0.5–1 kg/week weight loss",
    },

    # ── ارتجاع المريء — ACG Guidelines 2022 ───────────────
    "gerd": {
        "numeric_rules": {
            "Calories":   ("<=", 500),
            "FatContent": ("<=", 15),
        },
        "warning_rules": {
            "FatContent": (">=", 10),
            "Calories":   (">=", 400),
        },
        "strict_block": [
            "coffee", "espresso", "caffeine", "decaf coffee",
            "tomato sauce", "marinara", "ketchup", "salsa",
            "citrus", "lemon juice", "orange juice", "grapefruit",
            "chocolate", "cocoa", "mint", "peppermint",
            "chili", "hot sauce", "jalapeño", "cayenne",
            "garlic", "raw onion",
            "fried food", "deep fried",
        ],
        "soft_block": [
            "butter", "cream", "whole milk",
            "carbonated water", "sparkling water",
        ],
        "preferred_ingredients": [
            "oatmeal", "oats", "banana", "melon", "apple",
            "broccoli", "green beans", "asparagus", "cauliflower",
            "ginger", "aloe vera juice",
            "chicken breast", "turkey", "fish",
            "egg white", "tofu",
            "whole grain bread", "brown rice",
            "low-fat yogurt",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "ACG GERD Guidelines 2022 + NIDDK",
        "note":   "Small frequent meals; avoid lying down within 3 hours of eating",
    },

    # ── الكوليسترول المرتفع — ACC/AHA 2018 ────────────────
    "high_cholesterol": {
        "numeric_rules": {
            "CholesterolContent":  ("<=", 67),
            "SaturatedFatContent": ("<=", 4),
            "FatContent":          ("<=", 20),
        },
        "warning_rules": {
            "CholesterolContent":  (">=", 50),
            "SaturatedFatContent": (">=", 3),
            "FatContent":          (">=", 15),
        },
        "strict_block": [
            "lard", "hot dog", "ghee",
            "partially hydrogenated oil", "trans fat",
            "full-fat cheese",
            "organ meat", "liver",
        ],
        "soft_block": [
            "egg yolk", "whole milk", "cream",
            "coconut oil", "palm oil", "red meat",
            "shrimp",
        ],
        "preferred_ingredients": [
            "oats", "oatmeal", "barley", "psyllium",
            "flaxseed", "chia seeds",
            "salmon", "mackerel", "sardine",
            "olive oil", "avocado",
            "almonds", "walnuts",
            "beans", "lentils", "chickpeas",
            "garlic", "onion",
            "berries", "apple", "orange",
            "broccoli", "spinach", "kale",
        ],
        "min_requirements": {"FiberContent": (">=", 4)},
        "max_servings": 1,
        "source": "ACC/AHA Cholesterol Guidelines 2018",
        "note":   "Soluble fiber reduces LDL; avoid saturated & trans fats",
    },

    # ── هشاشة العظام — NOF + AACE/ACE 2020 ───────────────
    "osteoporosis": {
        "numeric_rules": {
            "Calories": (">=", 300),
        },
        "warning_rules": {
            "Calories": ("<=", 250),
        },
        "strict_block": ["cola", "dark soda"],
        "soft_block": ["coffee", "caffeine"],
        "preferred_ingredients": [
            "milk", "yogurt", "cheese", "sardine", "salmon",
            "broccoli", "kale", "bok choy", "spinach",
            "tofu", "fortified orange juice", "fortified cereal",
            "tuna", "mackerel", "egg yolk", "mushroom",
            "chicken", "turkey", "lentils", "beans",
        ],
        "min_requirements": {"ProteinContent": (">=", 15)},
        "max_servings": 1,
        "source": "National Osteoporosis Foundation + AACE/ACE 2020",
        "note":   "Ca 1000–1200 mg/day + Vitamin D 800–1000 IU/day",
    },

    # ── فقر الدم — WHO + ASH ──────────────────────────────
    "anemia": {
        "numeric_rules": {
            "Calories":       (">=", 300),
            "ProteinContent": (">=", 15),
        },
        "warning_rules": {
            "ProteinContent": ("<=", 10),
            "SugarContent":   (">=", 25),
        },
        "strict_block": ["coffee", "tea", "black tea", "green tea"],
        "soft_block": ["whole grain", "bran"],
        "preferred_ingredients": [
            "red meat", "beef", "lamb",
            "chicken", "turkey",
            "oyster", "clam", "sardine", "tuna",
            "spinach", "lentils", "beans", "chickpeas",
            "tofu", "pumpkin seeds",
            "fortified cereal",
            "bell pepper", "broccoli", "citrus", "tomato",
            "leafy greens", "asparagus", "avocado",
            "fortified bread",
        ],
        "min_requirements": {
            "ProteinContent": (">=", 15),
            "Calories":       (">=", 300),
        },
        "max_servings": 1,
        "source": "WHO Nutritional Anaemia Guidelines + ASH",
        "note":   "Vitamin C triples non-heme iron absorption",
    },

    # ── الحمل — ACOG 2021 + WHO Antenatal ─────────────────
    "pregnancy": {
        "numeric_rules": {
            "Calories":       (">=", 400),
            "ProteinContent": (">=", 20),
        },
        "warning_rules": {
            "Calories":      ("<=", 350),
            "SodiumContent": (">=", 600),
        },
        "strict_block": [
            "raw fish", "sushi", "sashimi",
            "raw oyster", "raw clam",
            "unpasteurized cheese", "soft cheese", "brie",
            "unpasteurized juice", "raw milk",
            "deli meat", "cold cuts",
            "swordfish", "shark", "tilefish", "king mackerel",
            "raw egg", "raw sprouts",
        ],
        "soft_block": ["coffee", "caffeine", "tuna", "processed meat"],
        "preferred_ingredients": [
            "spinach", "kale", "lentils", "beans", "asparagus",
            "fortified cereal", "fortified bread",
            "red meat", "chicken", "turkey",
            "milk", "yogurt", "cheese", "sardine", "broccoli",
            "salmon", "chia seeds", "flaxseed",
            "egg yolk", "fortified milk",
            "chicken breast", "tofu", "quinoa",
        ],
        "min_requirements": {
            "Calories":       (">=", 400),
            "ProteinContent": (">=", 20),
        },
        "max_servings": 1,
        "source": "ACOG 2021 + WHO Antenatal Care Guidelines",
        "note":   (
            "+300 kcal/day in 2nd/3rd trimester; folate 400–800 mcg/day. "
            "Low-mercury seafood (shrimp, salmon, cooked fish) limited to "
            "2–3 servings/week per ACOG/FDA guidance."
        ),
    },

    # ── الأطفال (4–12) — AAP + DGA 2025 ──────────────────
    "children": {
        "numeric_rules": {
            "Calories":      ("<=", 500),
            "SugarContent":  ("<=", 12),
            "SodiumContent": ("<=", 500),
        },
        "warning_rules": {
            "SugarContent": (">=", 8),
            "FatContent":   (">=", 20),
        },
        "strict_block": [
            "whole grapes", "hard candy", "chewing gum",
            "energy drink", "soda", "cola",
        ],
        "soft_block": ["fried food", "fast food", "processed meat", "hot dog"],
        "preferred_ingredients": [
            "milk", "yogurt", "cheese",
            "chicken", "turkey", "eggs", "fish",
            "lentils", "beans",
            "broccoli", "carrot", "sweet potato", "spinach",
            "apple", "banana", "berries", "orange",
            "whole grain", "oats", "brown rice", "quinoa",
        ],
        "min_requirements": {
            "ProteinContent": (">=", 10),
            "Calories":       (">=", 250),
        },
        "max_servings": 1,
        "source": "AAP Pediatric Nutrition + DGA 2025-2030 (Ages 4–12)",
        "note":   "Excess sugar & sodium in childhood predisposes to chronic disease",
    },

    # ── كبار السن (65+) — ESPEN + AGS + DGA 2025 ─────────
    "elderly": {
        "numeric_rules": {
            "ProteinContent": (">=", 20),
            "Calories":       (">=", 300),
            "SodiumContent":  ("<=", 600),
        },
        "warning_rules": {
            "ProteinContent":      ("<=", 15),
            "Calories":            ("<=", 250),
            "SaturatedFatContent": (">=", 5),
        },
        "strict_block": [],
        "soft_block": ["fried food", "processed meat", "high sodium foods"],
        "preferred_ingredients": [
            "chicken", "turkey", "fish", "eggs",
            "Greek yogurt", "cottage cheese",
            "lentils", "beans",
            "milk", "yogurt", "sardine", "salmon",
            "oatmeal", "mashed cauliflower",
            "banana", "berries", "apple",
            "blueberries", "spinach", "broccoli", "tomato",
        ],
        "min_requirements": {
            "ProteinContent": (">=", 20),
            "Calories":       (">=", 300),
        },
        "max_servings": 1,
        "source": "ESPEN Guidelines 2019 + AGS + DGA 2025 (Adults 65+)",
        "note":   "Sarcopenia prevention: 1.0–1.2 g protein/kg/day",
    },

    # ── عدم تحمل اللاكتوز — ACG + NIDDK ─────────────────
    "lactose_intolerance": {
        "numeric_rules": {},
        "warning_rules": {},
        "strict_block": [
            "milk", "whole milk", "skim milk", "low-fat milk",
            "2% milk", "evaporated milk", "condensed milk",
            "cream", "heavy cream", "sour cream", "half and half",
            "butter", "ghee",
            "cheese", "cheddar", "mozzarella", "parmesan",
            "ricotta", "cottage cheese", "cream cheese",
            "swiss cheese", "brie", "feta",
            "yogurt", "kefir", "ice cream", "whipped cream",
            "dairy", "whey", "whey protein", "lactose",
            "casein", "lactalbumin", "milk solids",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "lactose-free milk", "almond milk", "oat milk",
            "soy milk", "coconut milk", "rice milk",
            "dairy-free yogurt", "vegan cheese",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "ACG + NIDDK Clinical Guidelines",
        "note":   "Hard aged cheese (parmesan, cheddar) may be tolerated in small amounts",
    },

    # ── حساسية الجلوتين / سيلياك — ACG 2023 ──────────────
    "gluten_intolerance": {
        "numeric_rules": {},
        "warning_rules": {},
        "strict_block": [
            "wheat", "whole wheat", "wheat flour", "white flour",
            "all-purpose flour", "bread flour", "cake flour",
            "self-rising flour", "durum", "semolina", "farro",
            "spelt", "kamut", "einkorn", "emmer", "bulgur",
            "wheat bran", "wheat germ", "wheat starch",
            "barley", "rye", "triticale",
            "oat", "oats", "oatmeal", "rolled oats",
            "pasta", "noodles", "spaghetti", "penne", "lasagna",
            "bread", "breadcrumbs", "panko", "croutons",
            "couscous", "malt", "malt extract", "malt vinegar",
            "soy sauce", "flour tortilla", "pita",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "rice", "brown rice", "white rice", "rice flour",
            "quinoa", "buckwheat", "millet", "amaranth", "teff",
            "corn", "cornmeal", "cornstarch", "polenta",
            "potato", "sweet potato", "tapioca", "arrowroot",
            "almond flour", "coconut flour", "chickpea flour",
            "gluten-free oats", "gluten-free pasta", "gluten-free bread",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "ACG Guidelines 2023 + Celiac Disease Foundation",
        "note":   "FDA: gluten-free = < 20 ppm",
    },

    # ── حساسية المكسرات — FARE + ACAAI ───────────────────
    "nut_allergy": {
        "numeric_rules": {},
        "warning_rules": {},
        "strict_block": [
            "peanut", "peanuts", "peanut butter", "peanut oil",
            # 🛠️ إصلاح: "groundnut" مرادف بريطاني/هندي لـ peanut،
            # كان مفقوداً منذ دمج القوائم فكانت وصفات حقيقية تحتوي
            # "groundnuts" تتسرب (RecipeId 9878، 521164). النمط يضيف
            # "s?" تلقائياً فيغطي المفرد والجمع. لم نضف "nutty"/
            # "peanutty" عمداً — أسماء مثل "Nutty Chocolate Mint Fudge"
            # مكوّناتها نظيفة، وإضافتها ستُفرط في الحجب (RecipeId
            # 332170 "nutty rice" غامض وبقي كما هو).
            "groundnut",
            "almond", "almonds", "almond flour", "almond milk",
            "walnut", "walnuts", "pecan", "pecans",
            "cashew", "cashews", "pistachio", "pistachios",
            "hazelnut", "hazelnuts", "nutella",
            "macadamia", "brazil nut", "pine nut", "pine nuts",
            "mixed nuts", "tree nut", "nut butter",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "sunflower seeds", "pumpkin seeds", "sesame seeds",
            "flaxseed", "hemp seeds",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "FARE + ACAAI",
        "note":   "Anaphylaxis risk from trace amounts — carry EpiPen",
    },

    # ── حساسية المأكولات البحرية — FARE + FDA Top 9 ───────
    "seafood_allergy": {
        "numeric_rules": {},
        "warning_rules": {},
        "strict_block": [
            "fish", "salmon", "tuna", "cod", "tilapia",
            "halibut", "bass", "trout", "sardine", "anchovy",
            "mahi", "swordfish", "catfish", "flounder",
            "shrimp", "crab", "lobster", "clam", "oyster",
            "scallop", "mussel", "squid", "octopus",
            "crawfish", "prawn", "seafood",
            "fish sauce", "worcestershire", "anchovy paste",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "chicken", "turkey", "beef", "lamb",
            "lentils", "beans", "tofu", "tempeh", "eggs",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "FARE + FDA Top 9 Allergens List 2023",
        "note":   "Both finfish and shellfish excluded — anaphylaxis risk",
    },

    # ── حساسية البيض — FARE + ACAAI ──────────────────────
    "egg_allergy": {
        "numeric_rules": {},
        "warning_rules": {},
        "strict_block": [
            "egg", "eggs", "egg white", "egg yolk",
            "whole egg", "scrambled egg", "hard boiled egg",
            "egg powder", "dried egg", "egg solids",
            "albumin", "globulin", "lysozyme", "mayonnaise",
            "meringue", "surimi",
            # 🛠️ إصلاح: "eggnog" و"eggshell" — فحص قراءة-فقط وجد وصفات
            # تحتوي "eggnog ice cream" (منتج تجاري فيه بيض) أو
            # "eggshells" تتسرب لأن \begg\b لا يطابق الصيغ المركّبة
            # (RecipeId 77596، 77790، 185837، 200057، 262370، 420103،
            # 90358، 537278). لا يوجد عمود HasEggs في CSV، فالفحص نصي
            # فقط لهذه الفئة — الحماية هنا أولوية قصوى.
            "eggnog", "eggshell",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "flaxseed", "chia seeds", "applesauce", "tofu",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "FARE + ACAAI Guidelines",
        "note":   "Common in children; many outgrow by adolescence",
    },

    # ── حساسية السمسم — FDA FASTER Act 2023 (المسبب التاسع رسمياً) ──
    "sesame_allergy": {
        "numeric_rules": {},
        "warning_rules": {},
        "strict_block": [
            "sesame", "sesame seed", "sesame seeds", "sesame oil",
            "sesame paste", "tahini", "benne", "benne seed",
            "gomashio", "halva", "za'atar", "zaatar",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "sunflower seeds", "pumpkin seeds", "poppy seeds",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "FDA FASTER Act 2023 — 9th major allergen",
        "note":   "Effective Jan 1, 2023; hidden often in buns, sauces, hummus",
    },

    # ── النقرس — ACR 2020 ─────────────────────────────────
    "gout": {
        "numeric_rules": {
            "Calories":   ("<=", 500),
            "FatContent": ("<=", 20),
        },
        "warning_rules": {
            "ProteinContent": (">=", 30),
            "FatContent":     (">=", 15),
        },
        "strict_block": [
            "organ meat", "liver", "kidney", "sweetbread",
            "anchovies", "sardines", "herring", "mackerel",
            "high fructose corn syrup", "corn syrup",
            "soda", "cola", "energy drink", "fruit punch",
        ],
        "soft_block": [
            "red meat", "beef", "pork", "lamb",
            "shrimp", "crab", "lobster",
            "spinach", "asparagus", "mushroom",
        ],
        "preferred_ingredients": [
            "chicken breast", "turkey",
            "egg", "low-fat milk", "yogurt",
            "cherry", "tart cherry",
            "coffee",
            "broccoli", "cucumber", "lettuce", "tomato",
            "apple", "banana", "pineapple",
            "whole grain", "oats",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "ACR Gout Clinical Practice Guidelines 2020",
        "note":   "8–12 cups water/day significantly reduces flares",
    },

    # ── القولون العصبي — ACG 2021 + Low-FODMAP ────────────
    "irritable_bowel_syndrome": {
        "numeric_rules": {
            "Calories":   ("<=", 500),
            "FatContent": ("<=", 15),
        },
        "warning_rules": {
            "FatContent":   (">=", 12),
            "FiberContent": (">=", 10),
        },
        "strict_block": [
            "onion", "garlic", "leek",
            "wheat", "rye", "barley",
            "apple", "pear", "mango", "cherry",
            "milk", "yogurt", "soft cheese",
            "legumes", "beans", "chickpeas", "lentils",
            "cashew", "pistachio",
            "honey", "agave",
            "artificial sweetener",
            "sorbitol", "mannitol", "xylitol",
            "cauliflower", "mushroom",
        ],
        "soft_block": ["fried food", "spicy food", "carbonated drinks"],
        "preferred_ingredients": [
            "chicken", "turkey", "fish", "egg", "tofu",
            "rice", "oats", "quinoa", "potato",
            "carrot", "zucchini", "cucumber", "bell pepper",
            "spinach", "kale", "green beans",
            "strawberries", "blueberries", "grapes", "orange",
            "banana", "kiwi",
            "lactose-free milk", "hard cheese", "parmesan",
            "olive oil", "maple syrup",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "ACG IBS Guidelines 2021 + Monash University Low-FODMAP",
        "note":   "Low-FODMAP effective in 75% of cases; work with a dietitian",
    },

    # ── التهاب الكبد — AASLD + EASL 2023 ─────────────────
    "hepatitis": {
        "numeric_rules": {
            "Calories":            (">=", 350),
            "FatContent":          ("<=", 20),
            "SaturatedFatContent": ("<=", 5),
            "SugarContent":        ("<=", 15),
        },
        "warning_rules": {
            "FatContent":    (">=", 15),
            "SugarContent":  (">=", 10),
            "SodiumContent": (">=", 600),
        },
        "strict_block": [
            "partially hydrogenated oil", "trans fat", "shortening",
            "high fructose corn syrup", "corn syrup",
            "raw oyster", "raw clam", "raw fish", "sushi",
        ],
        "soft_block": [
            "salt",
            "butter", "lard", "ghee", "coconut oil",
            "red meat", "liver", "organ meat",
        ],
        "preferred_ingredients": [
            "broccoli", "cauliflower", "brussels sprouts",
            "spinach", "kale", "artichoke",
            "berries", "blueberries", "grapes",
            "green tea", "turmeric", "garlic",
            "chicken breast", "turkey", "egg white",
            "tofu", "lentils", "beans",
            "olive oil", "avocado", "salmon", "chia seeds",
            "oats", "brown rice", "sweet potato", "quinoa",
            "coffee",
        ],
        "min_requirements": {
            "Calories":       (">=", 350),
            "ProteinContent": (">=", 15),
        },
        "max_servings": 1,
        "source": "AASLD Practice Guidelines + EASL Clinical Practice Guidelines 2023",
        "note":   "Alcohol absolutely forbidden even in mild hepatitis",
    },

    # ── الربو — GINA 2024 + ERS/ATS ──────────────────────
    "asthma": {
        "numeric_rules": {
            "Calories":            ("<=", 600),
            "SaturatedFatContent": ("<=", 5),
            "SodiumContent":       ("<=", 600),
        },
        "warning_rules": {
            "SaturatedFatContent": (">=", 3),
            "SodiumContent":       (">=", 450),
            "FatContent":          (">=", 20),
        },
        "strict_block": [
            "dried fruit",
            "pickled food", "vinegar",
            "food coloring", "artificial color",
        ],
        "soft_block": [
            "processed meat", "hot dog",
            "fast food", "fried food",
            "margarine",
            "dairy",
        ],
        "preferred_ingredients": [
            "salmon", "mackerel", "sardine", "tuna",
            "flaxseed", "chia seeds", "walnuts",
            "bell pepper", "broccoli", "kiwi", "strawberries",
            "orange", "citrus",
            "salmon", "egg yolk", "fortified milk",
            "spinach", "pumpkin seeds", "dark chocolate",
            "avocado", "banana",
            "apple", "tomato",
            "garlic", "ginger", "turmeric",
            "oats", "quinoa", "brown rice",
            "chicken breast", "turkey", "lentils", "beans",
        ],
        "min_requirements": {},
        "max_servings": 1,
        "source": "GINA Global Strategy for Asthma Management 2024 + ERS/ATS",
        "note":   "Sulfites in preserved foods are a documented asthma trigger — check labels",
    },

    # ── قصور الغدة الدرقية — ATA 2014 + ETA 2023 ─────────
    "hypothyroidism": {
        "numeric_rules": {
            "Calories":     ("<=", 500),
            "FatContent":   ("<=", 18),
            "FiberContent": (">=", 3),
        },
        "warning_rules": {
            "Calories":     (">=", 450),
            "FatContent":   (">=", 15),
            "SugarContent": (">=", 15),
        },
        "strict_block": [
            "raw cabbage", "raw kale", "raw broccoli",
            "raw cauliflower", "raw brussels sprouts",
            "raw bok choy", "raw turnip", "raw rutabaga",
            "soy", "soy milk", "tofu", "tempeh", "edamame",
            "soy sauce", "miso", "soy protein",
            "millet",
        ],
        "soft_block": [
            "cabbage", "kale", "broccoli", "cauliflower",
            "brussels sprouts",
            "coffee",
            "milk", "yogurt", "cheese",
        ],
        "preferred_ingredients": [
            "seaweed", "iodized salt", "cod", "tuna", "shrimp",
            "egg", "yogurt", "milk",
            "brazil nut", "sardine", "salmon", "turkey", "chicken",
            "beef", "pumpkin seeds", "chickpeas", "lentils",
            "berries", "spinach", "sweet potato",
            "olive oil", "avocado",
            "chicken breast", "turkey", "eggs", "fish",
            "oats", "brown rice", "quinoa",
        ],
        "min_requirements": {"ProteinContent": (">=", 15)},
        "max_servings": 1,
        "source": "American Thyroid Association Guidelines 2014 + ETA 2023",
        "note":   "Take Levothyroxine on empty stomach; wait 30–60 min before eating.",
    },

    # ── فرط نشاط الغدة الدرقية — ATA/AACE 2016 ──────────
    "hyperthyroidism": {
        "numeric_rules": {
            "Calories":       (">=", 450),
            "ProteinContent": (">=", 20),
        },
        "warning_rules": {
            "Calories":       ("<=", 400),
            "ProteinContent": ("<=", 15),
            "SodiumContent":  (">=", 600),
        },
        "strict_block": [
            "seaweed", "kelp", "nori", "kombu", "wakame",
            "iodized salt",
        ],
        "soft_block": [
            "tuna", "cod", "shrimp",
            "milk", "yogurt",
            "chocolate", "cola", "soda",
        ],
        "preferred_ingredients": [
            "kale", "broccoli", "cabbage", "cauliflower",
            "brussels sprouts", "bok choy",
            "tofu", "soy milk",
            "milk", "yogurt", "cheese", "sardine",
            "broccoli", "fortified orange juice",
            "chicken breast", "turkey", "eggs",
            "lentils", "beans", "quinoa",
            "oats", "brown rice", "whole grain bread",
            "sweet potato", "banana",
            "spinach", "avocado", "pumpkin seeds",
        ],
        "min_requirements": {
            "Calories":       (">=", 450),
            "ProteinContent": (">=", 20),
        },
        "max_servings": 1,
        "source": "ATA/AACE Hyperthyroidism Guidelines 2016 + ETA 2022",
        "note":   "Hyperthyroidism is the mirror image of hypothyroidism — goitrogens are HELPFUL here",
    },

    # ── داء كرون — ECCO 2023 + ACG 2021 ──────────────────
    "crohns_disease": {
        "numeric_rules": {
            "FatContent":   ("<=", 15),
            "Calories":     ("<=", 500),
            "FiberContent": ("<=", 8),
        },
        "warning_rules": {
            "FatContent":    (">=", 12),
            "FiberContent":  (">=", 6),
            "SodiumContent": (">=", 700),
        },
        "strict_block": [
            "fried food", "deep fried",
            "spicy food", "chili", "hot sauce", "cayenne",
            "popcorn", "raw nuts", "seeds",
            "raw bran", "wheat bran",
            "corn on the cob", "raw corn",
            "high fructose corn syrup", "corn syrup",
            "sorbitol", "mannitol",
        ],
        "soft_block": [
            "raw broccoli", "raw cauliflower", "raw cabbage",
            "raw onion", "raw garlic",
            "beans", "lentils", "chickpeas",
            "milk", "cream", "ice cream",
        ],
        "preferred_ingredients": [
            "white rice", "white bread", "cream of wheat", "oatmeal",
            "banana", "applesauce", "melon",
            "cooked carrots", "cooked zucchini",
            "cooked spinach", "cooked sweet potato",
            "chicken breast", "turkey", "fish",
            "egg", "tofu",
            "olive oil", "avocado",
            "broth", "bone broth",
            "lactose-free yogurt",
        ],
        "min_requirements": {
            "Calories":       (">=", 300),
            "ProteinContent": (">=", 15),
        },
        "max_servings": 1,
        "source": "ECCO European Crohn's and Colitis Guidelines 2023 + ACG 2021",
        "note":   "Rules differ between flare (stricter) and remission. Monitor Fe, B12, Zn.",
    },

    # ── الإمساك المزمن — ACG 2021 + WGO ──────────────────
    "constipation": {
        "numeric_rules": {
            "FiberContent": (">=", 7),
            "FatContent":   ("<=", 20),
        },
        "warning_rules": {
            "FiberContent": ("<=", 4),
            "FatContent":   (">=", 18),
        },
        "strict_block": [
            "white bread", "white rice",
            "fried food", "fast food",
            "processed cheese",
        ],
        "soft_block": ["dairy", "red meat"],
        "preferred_ingredients": [
            "oats", "oatmeal", "psyllium husk",
            "apple", "pear", "berries",
            "lentils", "beans", "chickpeas",
            "whole wheat", "whole grain bread", "bran",
            "broccoli", "carrot", "leafy greens",
            "prune", "dried fig", "kiwi",
            "flaxseed",
            "olive oil", "avocado",
        ],
        "min_requirements": {"FiberContent": (">=", 7)},
        "max_servings": 1,
        "source": "ACG Chronic Constipation Guidelines 2021 + WGO",
        "note":   "8 cups water/day essential alongside fiber",
    },

    # ── نقص الوزن — ASPEN + AND + DGA 2025 ───────────────
    "underweight": {
        "numeric_rules": {
            "Calories":       (">=", 550),
            "ProteinContent": (">=", 25),
            "FatContent":     (">=", 15),
        },
        "warning_rules": {
            "Calories":       ("<=", 500),
            "ProteinContent": ("<=", 20),
        },
        "strict_block": [
            "diet soda", "zero calorie drink",
            "sugar-free syrup", "fat-free dressing",
        ],
        "soft_block": ["lettuce", "cucumber", "celery"],
        "preferred_ingredients": [
            "avocado", "olive oil", "coconut oil",
            "peanut butter", "almond butter",
            "nuts", "seeds", "trail mix",
            "whole egg", "chicken thigh", "salmon",
            "tuna", "beef",
            "full-fat yogurt", "whole milk", "cheese",
            "oats", "brown rice", "quinoa",
            "sweet potato", "banana",
            "whole grain bread", "pasta",
            "honey", "granola", "dried fruit",
            "dark chocolate",
        ],
        "min_requirements": {
            "Calories":       (">=", 550),
            "ProteinContent": (">=", 25),
        },
        "max_servings": 2,
        "source": "ASPEN Nutrition Support Guidelines + AND + DGA 2025-2030",
        "note":   "+500 kcal/day + resistance training for lean mass gain",
    },

    # ── تكيس المبايض — Endocrine Society 2023 ─────────────
    "pcos": {
        "numeric_rules": {
            "CarbohydrateContent": ("<=", 45),
            "SugarContent":        ("<=", 10),
            "FiberContent":        (">=", 5),
        },
        "warning_rules": {"CarbohydrateContent": (">=", 30)},
        "strict_block": [
            "white bread", "white rice", "sugar",
            "sugary drinks", "processed snacks",
        ],
        "soft_block": [],
        "preferred_ingredients": [
            "whole grain", "oats", "brown rice", "quinoa",
            "lentils", "beans", "chickpeas",
            "chicken breast", "turkey", "fish", "tofu",
            "spinach", "broccoli", "cucumber", "bell pepper",
            "berries", "apple", "avocado", "olive oil",
        ],
        "min_requirements": {"FiberContent": (">=", 5)},
        "max_servings": 1,
        "source": "Endocrine Society PCOS Guidelines 2023",
        "note":   "Low glycemic index diet; fiber + lean protein priority",
    },
}
