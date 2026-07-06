

from typing import Dict, List, Tuple


AGE_RANGES: Dict[str, Tuple[int, int]] = {
    "child":   (1,  12),
    "teen":    (13, 17),
    "adult":   (18, 64),
    "elderly": (65, 110),
}

AGE_RANGE_EN: Dict[str, str] = {
    "child":   "Child (1-12 years)",
    "teen":    "Teen (13-17 years)",
    "adult":   "Adult (18-64 years)",
    "elderly": "Elderly (65+ years)",
}

DISEASES_BY_STAGE: Dict[str, List[str]] = {
    "child": [
        "anemia",
        "asthma",
        "gluten_intolerance",
        "lactose_intolerance",
        "gerd",
        "obesity",
        "constipation",
    ],
    "teen": [
        "anemia",
        "obesity",
        "gluten_intolerance",
        "lactose_intolerance",
        "gerd",
        "asthma",
        "irritable_bowel_syndrome",
        "diabetes",
        "constipation",
        "underweight",
        "crohns_disease",
    ],
    "adult": [
        "diabetes",
        "hypertension",
        "obesity",
        "high_cholesterol",
        "gluten_intolerance",
        "lactose_intolerance",
        "anemia",
        "gerd",
        "gout",
        "chronic_kidney_disease",
        "osteoporosis",
        "irritable_bowel_syndrome",
        "hepatitis",
        "asthma",
        "heart_disease",
        "hypothyroidism",
        "hyperthyroidism",
        "crohns_disease",
        "constipation",
        "underweight",
    ],
    "elderly": [
        "diabetes",
        "hypertension",
        "high_cholesterol",
        "heart_disease",
        "osteoporosis",
        "chronic_kidney_disease",
        "obesity",
        "anemia",
        "irritable_bowel_syndrome",
        "asthma",
        "gerd",
        "gout",
        "hypothyroidism",
        "constipation",
        "underweight",
        "hepatitis",
    ],
}


DISEASES_FEMALE_ONLY: Dict[str, List[str]] = {
    "child":   [],
    "teen":    ["pcos"],
    "adult":   ["pcos"],
    "elderly": [],
}


ALLERGIES_BY_STAGE: Dict[str, List[str]] = {
   
    "child":   ["peanuts", "milk", "eggs", "seafood", "soy", "gluten", "sesame"],
    "teen":    ["peanuts", "milk", "eggs", "seafood", "soy", "gluten", "sesame"],
    "adult":   ["peanuts", "milk", "eggs", "seafood", "soy", "gluten", "sesame"],
    "elderly": ["peanuts", "milk", "eggs", "seafood", "soy", "gluten", "sesame"],
}


MEAL_TYPE_EN: Dict[str, str] = {
    "breakfast": "Breakfast",
    "lunch":     "Lunch",
    "dinner":    "Dinner",
    "any":       "No preference — show all",
}


DISEASE_EN: Dict[str, str] = {
    "diabetes":                 "Diabetes",
    "hypertension":             "Hypertension (High Blood Pressure)",
    "obesity":                  "Obesity",
    "high_cholesterol":         "High Cholesterol (Hypercholesterolemia)",
    "gluten_intolerance":       "Gluten Intolerance / Celiac Disease",
    "lactose_intolerance":      "Lactose Intolerance",
    "anemia":                   "Anemia (Iron Deficiency)",
    "gerd":                     "GERD / Acid Reflux",
    "gout":                     "Gout (High Uric Acid)",
    "chronic_kidney_disease":   "Chronic Kidney Disease (CKD)",
    "osteoporosis":             "Osteoporosis",
    "irritable_bowel_syndrome": "Irritable Bowel Syndrome (IBS)",
    "hepatitis":                "Hepatitis (Liver Disease)",
    "asthma":                   "Asthma",
    "heart_disease":            "Heart Disease",
    "pcos":                     "Polycystic Ovary Syndrome / PCOS (Females Only)",
    "hypothyroidism":           "Hypothyroidism (Underactive Thyroid)",
    "hyperthyroidism":          "Hyperthyroidism (Overactive Thyroid)",
    "crohns_disease":           "Crohn's Disease",
    "constipation":             "Chronic Constipation",
    "underweight":              "Underweight (BMI < 18.5)",
    "pregnancy":                "Pregnancy",
}

ALLERGY_EN: Dict[str, str] = {
    "peanuts": "Peanuts / Tree Nuts",
    "milk":    "Milk and Dairy",
    "eggs":    "Eggs",
    "seafood": "Seafood (Fish & Shellfish)",
    "soy":     "Soy",
    "gluten":  "Gluten / Wheat",
    "sesame":  "Sesame",
}

PREFERENCE_EN: Dict[str, str] = {
    "vegetarian":    "Vegetarian - No meat or seafood",
    "vegan":         "Vegan - No animal products",
    "seafood_lover": "Prefers seafood",
    "meat_lover":    "Prefers red meat",
    "chicken_lover": "Prefers chicken & poultry",
    "low_carb":      "Low carbohydrate (< 30 g/meal)",
    "mediterranean": "Mediterranean diet",
    "no_preference": "No specific preference",
}

GOALS_BY_CONDITION: Dict[str, List[str]] = {
    "pregnancy": ["pregnancy_diet", "maintenance"],
    "elderly":   ["elderly_diet", "weight_loss", "maintenance", "muscle_gain", "heart_health"],
    "child":     ["maintenance", "weight_loss", "weight_gain"],
    "teen":      ["weight_loss", "weight_gain", "muscle_gain", "maintenance"],
    "adult":     ["weight_loss", "weight_gain", "muscle_gain", "maintenance",
                  "heart_health", "healthy_eating"],
    "healthy":   ["weight_loss", "weight_gain", "muscle_gain", "maintenance",
                  "heart_health", "healthy_eating"],
}

GOAL_EN: Dict[str, str] = {
    "weight_loss":    "Weight Loss",
    "weight_gain":    "Weight Gain",
    "muscle_gain":    "Muscle Gain",
    "maintenance":    "Weight Maintenance",
    "heart_health":   "Heart Health",
    "pregnancy_diet": "Pregnancy Diet",
    "elderly_diet":   "Elderly Nutrition",
    "healthy_eating": "Healthy Eating",
}

NUTRIENT_COLS: List[str] = [
    "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
    "SodiumContent", "CarbohydrateContent", "FiberContent",
    "SugarContent", "ProteinContent",
]

COLUMN_NAMES: List[str] = [
    "RecipeId", "Name", "AuthorId", "AuthorName",
    "CookTime", "PrepTime", "TotalTime", "DatePublished",
    "Description", "RecipeCategory", "MainCategory", "DietType", "MealType",
    "Rating", "NumReviews", "Servings", "IsValidServing",
    "Calories", "FatContent", "SaturatedFatContent",
    "CholesterolContent", "SodiumContent", "CarbohydrateContent",
    "FiberContent", "SugarContent", "ProteinContent",
    "IngredientsList", "InstructionsList", "KeywordsList",
    "ImageUrl", "ImagesCount", "ImagesJson",
    "HasLactose", "HasGluten", "HasNuts", "HasSoy", "HasSeafood",
    "HealthScore", "RiskLevel", "DifficultyScore", "TimeCategory",
]