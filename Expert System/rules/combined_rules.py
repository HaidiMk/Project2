from typing import Dict, Any
from rules.medical_rules import MEDICAL_RULES

COMBINED_RULES: Dict[tuple, Dict[str, Any]] = {


    ("diabetes", "hypertension"): {
        "SugarContent":        ("<=", 15),
        "CarbohydrateContent": ("<=", 60),
        "FiberContent":        (">=", 5),
        "SodiumContent":       ("<=", 500),   
        "SaturatedFatContent": ("<=", 5),
        "Calories":            ("<=", 600),
        "blocked_ingredients": [],
        "conflict_warning": None,
        "note": "Sodium limit 500 (stricter than hypertension alone 600)",
    },

    ("diabetes", "heart_disease"): {
        "SugarContent":        ("<=", 15),
        "CarbohydrateContent": ("<=", 60),
        "FiberContent":        (">=", 5),
        "SodiumContent":       ("<=", 500),
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 67),
        "FatContent":          ("<=", 20),
        "Calories":            ("<=", 600),
        "blocked_ingredients": [],
        "conflict_warning": None,
    },

    ("diabetes", "obesity"): {
        "SugarContent":        ("<=", 10),
        "CarbohydrateContent": ("<=", 45),
        "FiberContent":        (">=", 5),
        "SodiumContent":       ("<=", 600),
        "SaturatedFatContent": ("<=", 5),
        "FatContent":          ("<=", 15),
        "Calories":            ("<=", 400),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Diabetes + Obesity: lower calorie target applied. "
            "Fewer options but safer and more effective."
        ),
    },

    ("chronic_kidney_disease", "diabetes"): {
        "SugarContent":        ("<=", 15),
        "CarbohydrateContent": ("<=", 60),
        "FiberContent":        (">=", 3),
        "SodiumContent":       ("<=", 500),
        "ProteinContent":      ("<=", 20),
        "CholesterolContent":  ("<=", 67),
        "Calories":            ("<=", 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Diabetes requires moderate protein but CKD limits it. "
            "Combined restrictions narrow options — best available recipes shown."
        ),
    },

    ("heart_disease", "hypertension"): {
        "SodiumContent":       ("<=", 500),
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 67),
        "FatContent":          ("<=", 20),
        "SugarContent":        ("<=", 10),
        "blocked_ingredients": [],
        "conflict_warning": None,
    },

    ("chronic_kidney_disease", "hypertension"): {
        "SodiumContent":       ("<=", 450),   
        "ProteinContent":      ("<=", 20),
        "CholesterolContent":  ("<=", 67),
        "FatContent":          ("<=", 20),
        "SaturatedFatContent": ("<=", 4),
        "blocked_ingredients": [],
        "conflict_warning": None,
        "note": "Sodium 450 mg (strictest — kidneys directly affected by hypertension)",
    },

    ("heart_disease", "high_cholesterol"): {
        "CholesterolContent":  ("<=", 50),
        "SaturatedFatContent": ("<=", 3),
        "FatContent":          ("<=", 15),
        "SodiumContent":       ("<=", 500),
        "SugarContent":        ("<=", 10),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": None,
        "note": "Strictest cholesterol + fat limits — highest cardiac risk combination",
    },

    ("diabetes", "high_cholesterol"): {
        "SugarContent":        ("<=", 10),
        "CarbohydrateContent": ("<=", 55),
        "FiberContent":        (">=", 5),
        "CholesterolContent":  ("<=", 67),
        "SaturatedFatContent": ("<=", 4),
        "FatContent":          ("<=", 20),
        "blocked_ingredients": [],
        "conflict_warning": None,
    },

    ("diabetes", "hypertension", "obesity"): {
        "SugarContent":        ("<=", 8),
        "CarbohydrateContent": ("<=", 40),
        "FiberContent":        (">=", 6),
        "SodiumContent":       ("<=", 450),
        "SaturatedFatContent": ("<=", 4),
        "FatContent":          ("<=", 12),
        "Calories":            ("<=", 350),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Metabolic Syndrome (3 diagnoses). Very strict limits — fewer recipes but safest. "
            "Strongly recommend consulting a registered dietitian."
        ),
    },

   

    ("gluten_intolerance", "lactose_intolerance"): {
        "blocked_ingredients": list(set(
            MEDICAL_RULES["lactose_intolerance"]["strict_block"] +
            MEDICAL_RULES["gluten_intolerance"]["strict_block"]
        )),
        "conflict_warning": (
            "Two intolerances — all dairy and wheat/gluten removed. "
            "Rice, quinoa, plant-based milks are your staples."
        ),
    },

    ("egg_allergy", "nut_allergy"): {
        "blocked_ingredients": list(set(
            MEDICAL_RULES["egg_allergy"]["strict_block"] +
            MEDICAL_RULES["nut_allergy"]["strict_block"]
        )),
        "conflict_warning": (
            "Egg + Nut allergy — both common allergens. "
            "Always check labels for hidden ingredients."
        ),
    },

    ("nut_allergy", "seafood_allergy"): {
        "blocked_ingredients": list(set(
            MEDICAL_RULES["nut_allergy"]["strict_block"] +
            MEDICAL_RULES["seafood_allergy"]["strict_block"]
        )),
        "conflict_warning": (
            "Two Top-9 allergens. Safe protein sources: chicken, meat, legumes, eggs (if tolerated)."
        ),
    },

    

    ("diabetes", "lactose_intolerance"): {
        "SugarContent":        ("<=", 15),
        "CarbohydrateContent": ("<=", 60),
        "FiberContent":        (">=", 5),
        "Calories":            ("<=", 600),
        "blocked_ingredients": MEDICAL_RULES["lactose_intolerance"]["strict_block"],
        "conflict_warning": (
            "Diabetic + lactose intolerant — avoid dairy and sugary foods. "
            "Oats and legumes are excellent choices."
        ),
    },

    ("diabetes", "gluten_intolerance"): {
        "SugarContent":        ("<=", 15),
        "CarbohydrateContent": ("<=", 60),
        "FiberContent":        (">=", 5),
        "Calories":            ("<=", 600),
        "blocked_ingredients": MEDICAL_RULES["gluten_intolerance"]["strict_block"],
        "conflict_warning": (
            "Diabetes + Celiac — brown rice, quinoa, buckwheat are ideal: "
            "gluten-free AND low glycemic index."
        ),
    },

    ("diabetes", "heart_disease", "lactose_intolerance"): {
        "SugarContent":        ("<=", 10),
        "CarbohydrateContent": ("<=", 55),
        "FiberContent":        (">=", 5),
        "SodiumContent":       ("<=", 500),
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 67),
        "FatContent":          ("<=", 20),
        "Calories":            ("<=", 550),
        "blocked_ingredients": MEDICAL_RULES["lactose_intolerance"]["strict_block"],
        "conflict_warning": (
            "Triple condition. Safe core: fish + vegetables + dairy-free whole grains."
        ),
    },

    
    
    ("diabetes", "pregnancy"): {
        "SugarContent":        ("<=", 10),
        "CarbohydrateContent": ("<=", 45),
        "FiberContent":        (">=", 5),
        "Calories":            ("between", 400, 550),
        "ProteinContent":      (">=", 20),
        "blocked_ingredients": MEDICAL_RULES["pregnancy"]["strict_block"],
        "conflict_warning": (
            "Gestational diabetes — calories must be adequate for fetal growth (≥400) "
            "while controlling carbs. Monitor blood glucose after each meal."
        ),
    },

    ("hypertension", "pregnancy"): {
        "SodiumContent":       ("<=", 500),
        "ProteinContent":      (">=", 20),
        "Calories":            (">=", 400),
        "blocked_ingredients": MEDICAL_RULES["pregnancy"]["strict_block"],
        "conflict_warning": (
            "Hypertension in pregnancy — Preeclampsia risk. "
            "Sodium control is critical. Consult your OB immediately."
        ),
    },

    ("gerd", "pregnancy"): {
        "FatContent":          ("<=", 15),
        "Calories":            (">=", 400),
        "ProteinContent":      (">=", 20),
        "SodiumContent":       ("<=", 550),
        "blocked_ingredients": MEDICAL_RULES["pregnancy"]["strict_block"],
        "conflict_warning": (
            "GERD in pregnancy — small frequent meals + upright sleeping position."
        ),
    },

    ("pregnancy", "seafood_allergy"): {
        "Calories":            (">=", 400),
        "ProteinContent":      (">=", 20),
        "blocked_ingredients": MEDICAL_RULES["seafood_allergy"]["strict_block"],
        "conflict_warning": (
            "Seafood allergy bans fish entirely — best source of Omega-3 for fetal brain. "
            "Safe alternatives: flaxseed, chia seeds, algal oil Omega-3 supplements."
        ),
    },

    ("constipation", "pregnancy"): {
        "FiberContent":        (">=", 7),
        "Calories":            (">=", 400),
        "ProteinContent":      (">=", 20),
        "blocked_ingredients": MEDICAL_RULES["pregnancy"]["strict_block"],
        "conflict_warning": (
            "Constipation in pregnancy — Fiber and water are completely safe. "
            "Avoid stimulant laxatives."
        ),
    },

    

    ("high_cholesterol", "obesity"): {
        "Calories":            ("<=", 400),
        "SugarContent":        ("<=", 10),
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 67),
        "FatContent":          ("<=", 12),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": None,
    },

    ("gerd", "obesity"): {
        "Calories":            ("<=", 400),
        "FatContent":          ("<=", 12),
        "SugarContent":        ("<=", 10),
        "SaturatedFatContent": ("<=", 4),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Excess weight increases abdominal pressure and worsens reflux. "
            "Weight loss is the most effective GERD treatment."
        ),
    },

    ("heart_disease", "obesity"): {
        "Calories":            ("<=", 400),
        "SodiumContent":       ("<=", 450),
        "SaturatedFatContent": ("<=", 3),
        "CholesterolContent":  ("<=", 60),
        "FatContent":          ("<=", 12),
        "SugarContent":        ("<=", 8),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Weak heart + obesity — sodium is the top danger: "
            "excess sodium can cause pulmonary congestion."
        ),
    },

    ("asthma", "obesity"): {
        "Calories":            ("<=", 400),
        "FatContent":          ("<=", 12),
        "SaturatedFatContent": ("<=", 4),
        "SugarContent":        ("<=", 8),
        "SodiumContent":       ("<=", 500),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Obesity worsens asthma mechanically and inflammatorily. "
            "Weight loss reduces asthma attacks."
        ),
    },


    ("chronic_kidney_disease", "heart_disease"): {
        "SodiumContent":       ("<=", 400),
        "ProteinContent":      ("<=", 18),
        "SaturatedFatContent": ("<=", 3),
        "CholesterolContent":  ("<=", 60),
        "FatContent":          ("<=", 15),
        "SugarContent":        ("<=", 10),
        "FiberContent":        (">=", 4),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Cardiorenal Syndrome — heart and kidneys directly affect each other. "
            "Sodium is enemy #1."
        ),
    },

    ("chronic_kidney_disease", "obesity"): {
        "Calories":            ("<=", 400),
        "ProteinContent":      ("<=", 18),
        "SodiumContent":       ("<=", 450),
        "FatContent":          ("<=", 12),
        "SaturatedFatContent": ("<=", 4),
        "SugarContent":        ("<=", 10),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Obesity increases glomerular pressure and accelerates CKD progression."
        ),
    },

    ("anemia", "chronic_kidney_disease"): {
        "ProteinContent":      ("<=", 20),
        "SodiumContent":       ("<=", 450),
        "FatContent":          ("<=", 18),
        "Calories":            (">=", 300),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Real conflict — renal anemia needs protein but kidneys limit it. "
            "EPO therapy may be needed — consult hematologist."
        ),
    },

    ("chronic_kidney_disease", "gout"): {
        "ProteinContent":      ("<=", 15),
        "SodiumContent":       ("<=", 450),
        "FatContent":          ("<=", 18),
        "Calories":            ("<=", 450),
        "blocked_ingredients": [],
        "conflict_warning": (
            "CKD + Gout — strict protein and purine limits. "
            "Eggs and low-fat dairy are the safest protein sources."
        ),
    },

    ("chronic_kidney_disease", "hepatitis"): {
        "ProteinContent":      ("<=", 15),
        "SodiumContent":       ("<=", 400),
        "FatContent":          ("<=", 15),
        "SugarContent":        ("<=", 12),
        "Calories":            (">=", 350),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Kidney + liver failure (Hepatorenal) — strictest protein and sodium limits. "
            "Requires intensive medical supervision."
        ),
    },

    
    
    ("diabetes", "gout"): {
        "SugarContent":        ("<=", 8),
        "CarbohydrateContent": ("<=", 50),
        "FiberContent":        (">=", 5),
        "FatContent":          ("<=", 18),
        "Calories":            ("<=", 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Partial conflict — diabetes controls carbs, gout bans fructose. "
            "Avoid all fruit juices and sweeteners."
        ),
    },

    ("gout", "hypertension"): {
        "SodiumContent":       ("<=", 500),
        "SaturatedFatContent": ("<=", 5),
        "FatContent":          ("<=", 18),
        "SugarContent":        ("<=", 10),
        "Calories":            ("<=", 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Drug warning — Thiazide diuretics (common BP drugs) raise uric acid and trigger gout. "
            "Inform your doctor of both conditions."
        ),
    },

    ("gout", "obesity"): {
        "Calories":            ("<=", 400),
        "FatContent":          ("<=", 12),
        "SugarContent":        ("<=", 6),
        "SaturatedFatContent": ("<=", 4),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Gout-obesity vicious cycle — gradual weight loss (not crash dieting) breaks the cycle."
        ),
    },

    ("gout", "heart_disease"): {
        "SodiumContent":       ("<=", 450),
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 60),
        "FatContent":          ("<=", 15),
        "SugarContent":        ("<=", 8),
        "Calories":            ("<=", 450),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Gout and heart disease share chronic inflammation as root cause. "
            "Alcohol and fructose banned for both."
        ),
    },

    

    ("irritable_bowel_syndrome", "lactose_intolerance"): {
        "FatContent":          ("<=", 12),
        "Calories":            ("<=", 450),
        "blocked_ingredients": list(set(
            MEDICAL_RULES["lactose_intolerance"]["strict_block"] +
            MEDICAL_RULES["irritable_bowel_syndrome"]["strict_block"]
        )),
        "conflict_warning": (
            "IBS + Lactose intolerance — Low-FODMAP + dairy-free. "
            "Rice, chicken, cooked vegetables are safest."
        ),
    },

    ("gerd", "irritable_bowel_syndrome"): {
        "FatContent":          ("<=", 12),
        "Calories":            ("<=", 450),
        "blocked_ingredients": [],
        "conflict_warning": (
            "GERD bans citrus, garlic, onion; IBS also bans wheat, legumes, dairy. "
            "Safe overlap: chicken, rice, cooked carrot, banana, oatmeal."
        ),
    },

    ("diabetes", "irritable_bowel_syndrome"): {
        "SugarContent":        ("<=", 12),
        "CarbohydrateContent": ("<=", 55),
        "FiberContent":        (">=", 3),
        "FatContent":          ("<=", 15),
        "Calories":            ("<=", 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Conflict — diabetes needs high fiber (≥5g) but IBS cannot tolerate excess. "
            "Soluble fiber (oats, cooked carrot) is safe for both."
        ),
    },

    ("gluten_intolerance", "irritable_bowel_syndrome"): {
        "FatContent":          ("<=", 12),
        "Calories":            ("<=", 500),
        "blocked_ingredients": MEDICAL_RULES["gluten_intolerance"]["strict_block"],
        "conflict_warning": (
            "IBS + Celiac — both ban wheat but for different reasons. "
            "Rice, quinoa, low-FODMAP vegetables are the safest core."
        ),
    },

    ("constipation", "irritable_bowel_syndrome"): {
        "FiberContent":        (">=", 5),
        "FatContent":          ("<=", 12),
        "Calories":            ("<=", 450),
        "blocked_ingredients": [],
        "conflict_warning": (
            "IBS-C (constipation type) — Soluble fiber (oats, psyllium) is safe for both. "
            "Avoid insoluble fiber (wheat bran)."
        ),
    },

    
    ("diabetes", "hepatitis"): {
        "SugarContent":        ("<=", 10),
        "CarbohydrateContent": ("<=", 50),
        "FiberContent":        (">=", 5),
        "FatContent":          ("<=", 15),
        "SaturatedFatContent": ("<=", 4),
        "Calories":            ("<=", 500),
        "ProteinContent":      (">=", 15),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Diabetes + hepatitis — excess sugar converts to hepatic fat and accelerates fibrosis."
        ),
    },

    ("hepatitis", "hypertension"): {
        "SodiumContent":       ("<=", 450),
        "FatContent":          ("<=", 18),
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 80),
        "SugarContent":        ("<=", 15),
        "Calories":            (">=", 350),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Hepatitis + hypertension — sodium increases ascites risk. "
            "Salt restriction is top priority."
        ),
    },

    ("hepatitis", "obesity"): {
        "Calories":            ("<=", 400),
        "FatContent":          ("<=", 12),
        "SaturatedFatContent": ("<=", 4),
        "SugarContent":        ("<=", 8),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Obesity is the #1 cause of NAFLD. "
            "Losing 5–10% of body weight significantly improves liver inflammation."
        ),
    },

    ("hepatitis", "high_cholesterol"): {
        "FatContent":          ("<=", 15),
        "SaturatedFatContent": ("<=", 3),
        "CholesterolContent":  ("<=", 60),
        "SugarContent":        ("<=", 10),
        "FiberContent":        (">=", 5),
        "Calories":            ("<=", 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Damaged liver is less efficient at processing cholesterol."
        ),
    },

    

    ("hypothyroidism", "obesity"): {
        "Calories":            ("<=", 450),
        "FatContent":          ("<=", 15),
        "SaturatedFatContent": ("<=", 4),
        "SugarContent":        ("<=", 8),
        "FiberContent":        (">=", 5),
        "blocked_ingredients": MEDICAL_RULES["hypothyroidism"]["strict_block"],
        "conflict_warning": (
            "Hypothyroidism slows metabolism making weight loss harder. "
            "Treating the thyroid with medication is prerequisite."
        ),
    },

    ("diabetes", "hypothyroidism"): {
        "SugarContent":        ("<=", 12),
        "CarbohydrateContent": ("<=", 55),
        "FiberContent":        (">=", 4),
        "Calories":            ("<=", 500),
        "FatContent":          ("<=", 16),
        "blocked_ingredients": MEDICAL_RULES["hypothyroidism"]["strict_block"],
        "conflict_warning": (
            "Hypothyroidism worsens blood sugar control and increases insulin resistance. "
            "Soy is banned for both."
        ),
    },

    ("high_cholesterol", "hypothyroidism"): {
        "SaturatedFatContent": ("<=", 4),
        "CholesterolContent":  ("<=", 60),
        "FatContent":          ("<=", 15),
        "FiberContent":        (">=", 5),
        "Calories":            ("<=", 500),
        "blocked_ingredients": MEDICAL_RULES["hypothyroidism"]["strict_block"],
        "conflict_warning": (
            "Hypothyroidism directly raises LDL cholesterol. "
            "Treating the thyroid usually lowers cholesterol automatically."
        ),
    },

    ("diabetes", "hyperthyroidism"): {
        "SugarContent":        ("<=", 10),
        "CarbohydrateContent": ("<=", 50),
        "Calories":            (">=", 450),
        "ProteinContent":      (">=", 20),
        "blocked_ingredients": MEDICAL_RULES["hyperthyroidism"]["strict_block"],
        "conflict_warning": (
            "Critical conflict — hyperthyroidism raises blood sugar by speeding metabolism. "
            "Treating the thyroid first improves glucose control automatically."
        ),
    },

    ("hyperthyroidism", "osteoporosis"): {
        "Calories":            (">=", 400),
        "ProteinContent":      (">=", 20),
        "FatContent":          ("<=", 18),
        "blocked_ingredients": MEDICAL_RULES["hyperthyroidism"]["strict_block"],
        "conflict_warning": (
            "Hyperthyroidism leaches calcium from bones — very high fracture risk. "
            "Calcium + Vitamin D essential alongside thyroid treatment."
        ),
    },

    ("hyperthyroidism", "underweight"): {
        "Calories":            (">=", 550),
        "ProteinContent":      (">=", 25),
        "FatContent":          (">=", 15),
        "blocked_ingredients": MEDICAL_RULES["hyperthyroidism"]["strict_block"],
        "conflict_warning": (
            "Hyperthyroidism is likely causing the underweight. "
            "High calories needed until hormone levels stabilize."
        ),
    },

   
    ("hyperthyroidism", "hypothyroidism"): {
        "blocked_ingredients": [],
        "conflict_warning": (
            "Data error — a patient cannot have both hypothyroidism and hyperthyroidism simultaneously. "
            "Please verify the diagnosis and correct the input."
        ),
    },

    
    ("anemia", "crohns_disease"): {
        "Calories":            (">=", 350),
        "ProteinContent":      (">=", 15),
        "FatContent":          ("<=", 15),
        "FiberContent":        ("<=", 8),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Crohn's-related anemia — inflamed intestines absorb iron and B12 poorly. "
            "Heme iron (chicken, fish) is preferable. B12 injections may be needed."
        ),
    },

    ("crohns_disease", "underweight"): {
        "Calories":            (">=", 400),
        "ProteinContent":      (">=", 20),
        "FatContent":          ("<=", 15),
        "FiberContent":        ("<=", 6),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Critical conflict — Crohn's restricts many foods but underweight needs more calories. "
            "Tube feeding or TPN may be needed in severe cases."
        ),
    },

    ("crohns_disease", "gluten_intolerance"): {
        "FatContent":          ("<=", 12),
        "FiberContent":        ("<=", 6),
        "Calories":            ("between", 300, 500),
        "ProteinContent":      (">=", 15),
        "blocked_ingredients": MEDICAL_RULES["gluten_intolerance"]["strict_block"],
        "conflict_warning": (
            "Crohn's + Celiac — very narrow food choices. "
            "Core: white rice, potato, chicken, eggs, cooked vegetables."
        ),
    },

    ("constipation", "crohns_disease"): {
        "FiberContent":        ("between", 4, 7),
        "FatContent":          ("<=", 15),
        "Calories":            ("between", 300, 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Real conflict — Crohn's bans coarse fiber but constipation needs it. "
            "Solution: soluble fiber only (oats, cooked carrot, psyllium)."
        ),
    },

    
    ("constipation", "hypothyroidism"): {
        "FiberContent":        (">=", 7),
        "FatContent":          ("<=", 18),
        "Calories":            ("<=", 500),
        "blocked_ingredients": MEDICAL_RULES["hypothyroidism"]["strict_block"],
        "conflict_warning": (
            "Hypothyroidism directly causes constipation by slowing bowel motility. "
            "High fiber + adequate water while waiting for medication to take effect."
        ),
    },

    

    ("diabetes", "underweight"): {
        "Calories":            (">=", 450),
        "ProteinContent":      (">=", 20),
        "CarbohydrateContent": ("<=", 60),
        "SugarContent":        ("<=", 15),
        "FiberContent":        (">=", 4),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Dangerous conflict — diabetes limits calories and carbs but underweight needs more. "
            "Priority: calories from protein and healthy fats (not sugar). "
            "Continuous glucose monitoring is essential."
        ),
    },

    ("elderly", "underweight"): {
        "Calories":            (">=", 450),
        "ProteinContent":      (">=", 25),
        "FatContent":          (">=", 12),
        "SodiumContent":       ("<=", 600),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Elderly malnutrition is silent and dangerous. "
            "Small frequent nutrient-dense meals > large rare meals."
        ),
    },

    

    ("diabetes", "heart_disease", "hypertension"): {
        "SugarContent":        ("<=", 8),
        "CarbohydrateContent": ("<=", 45),
        "FiberContent":        (">=", 6),
        "SodiumContent":       ("<=", 400),
        "SaturatedFatContent": ("<=", 3),
        "CholesterolContent":  ("<=", 55),
        "FatContent":          ("<=", 12),
        "Calories":            ("<=", 500),
        "ProteinContent":      (">=", 15),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Cardiac Triple — most common combination worldwide. "
            "Strictest possible dietary limits. Cardiac dietitian consultation strongly recommended."
        ),
    },

    
    ("diabetes", "elderly"): {
        "SugarContent":        ("<=", 12),
        "CarbohydrateContent": ("<=", 55),
        "FiberContent":        (">=", 4),
        "Calories":            ("between", 350, 550),
        "ProteinContent":      (">=", 20),
        "SodiumContent":       ("<=", 550),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Elderly diabetic — protein is critical to prevent muscle wasting alongside carb control. "
            "Small frequent meals are ideal."
        ),
    },

    ("elderly", "heart_disease"): {
        "SodiumContent":       ("<=", 450),
        "SaturatedFatContent": ("<=", 3),
        "CholesterolContent":  ("<=", 60),
        "FatContent":          ("<=", 15),
        "ProteinContent":      (">=", 20),
        "Calories":            ("between", 300, 500),
        "FiberContent":        (">=", 4),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Elderly heart failure — sodium is the most dangerous element: "
            "causes fluid retention and respiratory distress."
        ),
    },

    ("chronic_kidney_disease", "elderly"): {
        "ProteinContent":      ("<=", 18),
        "SodiumContent":       ("<=", 400),
        "FatContent":          ("<=", 18),
        "CholesterolContent":  ("<=", 67),
        "Calories":            ("between", 300, 500),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Important conflict — elderly need high protein (≥20g) to prevent sarcopenia, "
            "but CKD limits it (≤18g). "
            "High-quality protein (eggs, fish, chicken) is better than high quantity."
        ),
    },

    
    
    ("asthma", "egg_allergy"): {
        "SaturatedFatContent": ("<=", 5),
        "SodiumContent":       ("<=", 550),
        "blocked_ingredients": MEDICAL_RULES["egg_allergy"]["strict_block"],
        "conflict_warning": (
            "Egg allergy + asthma — food allergies can trigger asthma attacks. "
            "Always check labels carefully."
        ),
    },

    ("asthma", "nut_allergy"): {
        "SaturatedFatContent": ("<=", 5),
        "SodiumContent":       ("<=", 550),
        "blocked_ingredients": MEDICAL_RULES["nut_allergy"]["strict_block"],
        "conflict_warning": (
            "Asthma + nut allergy — both are atopic conditions (Atopic Triad). "
            "Always carry an EpiPen."
        ),
    },

    ("asthma", "hypertension"): {
        "SodiumContent":       ("<=", 550),
        "SaturatedFatContent": ("<=", 4),
        "FatContent":          ("<=", 18),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Drug alert — Beta-blockers (most common antihypertensives) can worsen asthma. "
            "Tell your doctor about both conditions."
        ),
    },

    ("asthma", "diabetes"): {
        "SugarContent":        ("<=", 12),
        "CarbohydrateContent": ("<=", 55),
        "FiberContent":        (">=", 5),
        "SaturatedFatContent": ("<=", 5),
        "SodiumContent":       ("<=", 550),
        "Calories":            ("<=", 550),
        "blocked_ingredients": [],
        "conflict_warning": (
            "Asthma and diabetes both involve chronic inflammation. "
            "Anti-inflammatory diet (Omega-3, vegetables, fiber) benefits both."
        ),
    },


    ("gluten_intolerance", "osteoporosis"): {
        "Calories":            (">=", 300),
        "ProteinContent":      (">=", 15),
        "blocked_ingredients": MEDICAL_RULES["gluten_intolerance"]["strict_block"],
        "conflict_warning": (
            "Real conflict — Celiac disease damages intestinal villi, preventing calcium "
            "and Vitamin D absorption, which accelerates osteoporosis. "
            "Strict gluten-free diet is the primary treatment for bone health here."
        ),
    },
}
