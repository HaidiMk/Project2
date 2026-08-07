"""
taste_concepts.py — خريطة مفاهيم/مرادفات الذوق (Concept → Synonyms)
===================================================================
تحوّل كلمات عامة لا توجد حرفياً في مفردات Word2Vec (مثل "spicy" أو
"italian") إلى قائمة رموز مكوّنات حقيقية موجودة فعلياً في النموذج
(word2vec_ingredients.model) — فيتمدد المفهوم إلى إشارات مكوّنات متعددة.

كل رمز هدف تحقق من وجوده في مفردات النموذج عند كتابة هذا الملف
(فحص membership عبر wv): أي رمز غير موجود حُذف أو استُبدل من القائمة.

تُستخدم هذه الخريطة داخل taste_inference.py بالترتيب التالي لحل كل
رمز مرشّح:
    1. مطابقة تامة في المفردات (السلوك الحالي — يبقى كما هو)
    2. تمدد مفهوم (إن كان الرمز مفتاحاً في CONCEPT_MAP — كل رموز
       الهدف تضاف كنواقل، وليس واحداً فقط)
    3. مطابقة تقريبية (fuzzy) كملاذ أخير فقط
وينطبق نفس الترتيب داخل جمل DISLIKED تماماً.

ملاحظة المفاهيم المتداخلة: اختيرت القوائم لتغطية اتجاهات تذوقية
مميزة؛ التداخل البسيط (مثل lemon في sour و mediterranean) مقصود —
المفهوم الواحد يدفع بعدة نواقل اتجاهاً واحداً.

الإثراء بالرموز المركّبة (من فحص المفردات الناقصة):
    فحص سابق للبيانات أظهر أن الكثير من المكونات الشائعة لا تظهر
    كنص مفرد إطلاقاً — اسمها الحقيقي في مفردات النموذج مركّب بشرطة
    سفلية (مثل "sour cream" -> sour_cream، "parmesan cheese" ->
    parmesan_cheese). أُثرت قوائم الأهداف برموز مركّبة مُحقَّقة
    الوجود في المفردات الحالية (كل رمز أُضيف فُحص membership عليه
    في وقت التعديل — "ribs" و "steak" المفردان غير موجودين).

    ⚠️ فخّ دلالي موثق — لا تُعد هذه الأخطاء:
      - "ribs": الرمز الأكثر شيوعاً الذي يحوي "ribs" هو celery_ribs
        (3,496 ظهور) ويعني سيقان الكرفس لا الضلوع! إشارات الشواء
        تستخدم المركّبات المحددة: country-style_pork_ribs و beef_ribs.
      - "steak": لا يوجد رمز عام للستيك — فقط قطع محددة (مثل
        beef_flank_steak و beef_round_steak). لا تضيف "steak" المجردة.
      - "fish_sauce" (2,054) صلصة توابل لا بروتين — تصلح في asian/salty
        ولا توضع في seafood أو healthy.

مفاهيم الفئات العامة (من تدقيق المطابقة التقريبية):
    تدقيق لاحق وجد أن كلمات فئات عامة/جمع شائعة (مثل "vegetables")
    لا يوجد لها تطابق تام ولا مفتاح مفهوم، فتسقط في المطابقة التقريبية
    وتُربط خطأً برموز متشابهة إملائياً لكنها خاطئة دلالياً:
      - "vegetables" كان يُربط بـ vegetable_suet (دهن خبز، غير خضروات!)
      - "sweets" كان يُربط بـ swedes (لفت سويدي!)
    أُضيفت مفاتيح مفاهيم لهذه الكلمات (vegetables/veggies/fruits/meat/
    nuts/beans/grains/herb/spices/sweets/legumes/poultry/fish/carbs/
    berry/whole_grains...) بكل أشكالها المفرد والجمع، كل رمز هدف منها
    محقَّق الوجود في مفردات النموذج. "green" وحدها تُركت بلا مفتاح:
    غامضة جداً (لون؟ زيتون أخضر؟ بصل أخضر؟) — السلوك الحالي
    (مطابقة تقريبية إلى "greens") مقبول ولا يحمل معنى خاطئاً،
    والخطأ في التخمين أسوأ من عدم المطابقة.
"""

CONCEPT_MAP = {
    "spicy": [
        "cayenne", "cayenne_pepper", "chili_powder", "chili_pepper",
        "red_pepper_flakes", "habanero", "paprika", "curry_powder",
        "garam_masala",
        "jalapeno_chile", "jalapeno_chiles", "red_chilies",
        "chipotle_pepper",
    ],
    "sweet": [
        "sugar", "brown_sugar", "honey", "maple_syrup", "chocolate",
        "vanilla", "molasses", "powdered_sugar", "granulated_sugar",
        "dates", "raisins", "vanilla_extract", "sweet_potato",
        "corn_syrup", "light_corn_syrup",
    ],
    "salty": [
        "salt", "sea_salt", "kosher_salt", "soy_sauce", "anchovy",
        "capers", "bacon",
        "black_olives", "green_olives", "kalamata_olives", "fish_sauce",
    ],
    "sour": [
        "lemon", "lime", "lemon_juice", "lime_juice", "vinegar",
        "white_vinegar", "apple_cider_vinegar", "buttermilk", "yogurt",
        "pickles", "sauerkraut",
        "tamarind_paste", "tamarind_pulp", "tamarind_juice",
    ],
    "healthy": [
        "spinach", "kale", "broccoli", "cauliflower", "quinoa", "oats",
        "oatmeal", "avocado", "blueberries", "salmon", "lentils",
        "chickpeas",
    ],
    "italian": [
        "pasta", "spaghetti", "penne", "basil", "oregano", "tomatoes",
        "garlic", "olive_oil", "prosciutto",
        "parmesan_cheese", "parmigiano-reggiano_cheese", "mozzarella_cheese",
    ],
    "asian": [
        "soy_sauce", "ginger", "rice", "tofu", "bok_choy", "wasabi",
        "scallions", "rice_vinegar", "miso", "sesame_seeds",
        "bean_sprouts", "oyster_sauce", "fish_sauce", "curry",
        "coconut_milk", "wonton",
        "sriracha_sauce", "udon_noodles", "cellophane_noodles",
    ],
    "mexican": [
        "tortilla", "corn_tortillas", "salsa", "cumin", "cilantro",
        "black_beans", "avocado", "corn", "lime",
        "jalapeno_chile", "jalapeno_chiles", "jalapeno_jack_cheese",
        "chipotle_chile_in_adobo", "chipotle_pepper",
        "chili_powder", "green_chilies", "red_chilies",
        "queso_fresco", "queso_blanco",
    ],
    "mediterranean": [
        "olive_oil", "feta", "chickpeas", "tahini", "lemon", "oregano",
        "cucumber", "tomatoes", "yogurt", "zucchini", "eggplant",
        "garlic", "parsley", "mint", "cinnamon",
        "black_olives", "green_olives", "kalamata_olives",
    ],
    "vegetarian": [
        "lentils", "chickpeas", "tofu", "tempeh", "quinoa", "mushrooms",
        "eggplant", "black_beans", "kidney_beans", "pinto_beans",
        "sweet_potatoes",
    ],
    "seafood": [
        "shrimp", "salmon", "tuna", "cod", "scallops", "mussels",
        "clams", "anchovies", "trout", "lobster", "squid",
        "lump_crabmeat", "fresh_crabmeat", "white_crab_meat",
    ],
    "dairy": [
        "milk", "butter", "cheese", "yogurt", "sour_cream",
        "cream_cheese", "heavy_cream", "buttermilk",
        "cheddar_cheese", "sharp_cheddar_cheese", "mozzarella_cheese",
        "parmesan_cheese",
    ],
    "breakfast": [
        "eggs", "egg", "bacon", "oatmeal", "sausage", "hash_browns",
        "bagel", "grits",
        "granola_cereal", "maple_syrup",
    ],
    "dessert": [
        "chocolate", "sugar", "butter", "flour", "vanilla",
        "ice_cream", "chocolate_chips", "cream_cheese", "marshmallows",
        "cocoa_powder", "corn_syrup", "light_corn_syrup",
    ],
    "grilled": [
        "barbecue_sauce", "chicken", "ketchup", "mustard", "pork",
        "beef", "hamburger", "sausage", "chicken_wings",
        "country-style_pork_ribs", "beef_ribs",
        "beef_flank_steak", "beef_round_steak",
    ],
    # ── فئات عامة (مفاهيم كلمات الجمع/الفئات) — أُضيفت بعد تدقيق ──
    "vegetables": [
        "carrots", "spinach", "broccoli", "zucchini", "mushrooms",
        "bell_pepper", "bell_peppers", "tomatoes", "onions", "celery",
        "cauliflower", "cabbage", "lettuce", "kale", "green_beans",
        "peas", "corn", "squash", "eggplant", "cucumber", "pumpkin",
        "asparagus", "sweet_potatoes", "potatoes", "garlic",
    ],
    "fruits": [
        "blueberries", "strawberries", "apples", "bananas", "oranges",
        "grapes", "watermelon", "melon", "pears", "pineapple", "mango",
        "lemons", "limes", "raspberries", "cranberries", "cherries",
        "plums", "apricots", "kiwi", "figs", "papaya", "pomegranate",
    ],
    "meat": [
        "beef", "ground_beef", "pork", "lamb", "duck", "ham", "bacon",
        "sausage", "chicken", "chicken_breasts", "turkey",
        "ground_turkey",
    ],
    "nuts": [
        "walnuts", "pecans", "peanuts", "cashews", "hazelnuts",
        "pine_nuts", "pistachios",
    ],
    "beans": [
        "black_beans", "kidney_beans", "pinto_beans", "green_beans",
        "navy_beans", "lima_beans", "soybeans", "edamame", "chickpeas",
    ],
    "grains": [
        "rice", "brown_rice", "white_rice", "oats", "oatmeal", "quinoa",
        "barley", "couscous", "bulgur", "cornmeal",
    ],
    "whole_grains": [
        "oats", "oatmeal", "quinoa", "barley", "brown_rice", "couscous",
        "bulgur",
    ],
    "herb": [
        "basil", "oregano", "parsley", "cilantro", "mint", "rosemary",
        "thyme", "dill", "sage", "chives", "bay_leaves", "tarragon",
    ],
    "spices": [
        "cinnamon", "cumin", "paprika", "turmeric", "ginger", "nutmeg",
        "cloves", "cardamom", "fennel", "allspice", "cayenne",
        "chili_powder", "black_pepper", "garlic_powder", "onion_powder",
        "curry_powder", "garam_masala",
    ],
    "sweets": [
        "sugar", "brown_sugar", "honey", "chocolate", "vanilla",
        "molasses", "powdered_sugar", "granulated_sugar", "cocoa_powder",
        "dates", "raisins",
    ],
    "legumes": [
        "lentils", "chickpeas", "black_beans", "kidney_beans",
        "pinto_beans", "navy_beans", "lima_beans", "soybeans", "edamame",
    ],
    "poultry": [
        "chicken", "chicken_breasts", "turkey", "ground_turkey", "duck",
    ],
    "fish": [
        "salmon", "tuna", "cod", "trout", "anchovies", "salmon_fillets",
        "salmon_steaks", "tuna_steaks",
    ],
    "shellfish": [
        "shrimp", "scallops", "mussels", "clams", "lobster", "squid",
    ],
    "carbs": [
        "pasta", "spaghetti", "rice", "brown_rice", "white_rice",
        "potatoes", "sweet_potatoes", "cornmeal", "flour",
    ],
    "berry": [
        "blueberries", "strawberries", "raspberries", "cranberries",
    ],
}

# ── أشكال المفرد/الجمع لنفس المفهوم (alias بأسماء متعددة) ──────────
for _alias, _canonical in {
    "veggie": "vegetables",
    "veggies": "vegetables",
    "vegetable": "vegetables",
    "veg": "vegetables",
    "fruit": "fruits",
    "meats": "meat",
    "nut": "nuts",
    "bean": "beans",
    "grain": "grains",
    "spice": "spices",
    "legume": "legumes",
    "carb": "carbs",
}.items():
    CONCEPT_MAP[_alias] = CONCEPT_MAP[_canonical]
