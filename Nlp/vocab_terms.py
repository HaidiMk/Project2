# Mirrored from Expert System/core/constants.py — keep in sync
# manually if the canonical lists change there.

# Canonical vocabulary lists used for semantic matching. This local, static
# copy is what makes the Nlp/ package fully standalone: it has zero import
# dependency on Expert System/ (or anything else) at runtime.


DISEASE_EN = {
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


MEAL_TYPE_EN = {
    "breakfast": "Breakfast",
    "lunch":     "Lunch",
    "dinner":    "Dinner",
    "any":       "No preference — show all",
}


ALLERGY_EN = {
    "peanuts": "Peanuts / Tree Nuts",
    "milk":    "Milk and Dairy",
    "eggs":    "Eggs",
    "seafood": "Seafood (Fish & Shellfish)",
    "soy":     "Soy",
    "gluten":  "Gluten / Wheat",
    "sesame":  "Sesame",
}


PREFERENCE_EN = {
    "vegetarian":    "Vegetarian - No meat or seafood",
    "vegan":         "Vegan - No animal products",
    "seafood_lover": "Prefers seafood",
    "meat_lover":    "Prefers red meat",
    "chicken_lover": "Prefers chicken & poultry",
    "low_carb":      "Low carbohydrate (< 30 g/meal)",
    "mediterranean": "Mediterranean diet",
    "no_preference": "No specific preference",
}
