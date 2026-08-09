"""
test_robustness_all.py — false-positive regression suite for all four
semantic categories (condition, meal_type, diet_preference,
main_ingredient) plus the required sample queries.

Groups:
  G1 condition true positives     — genuine condition mentions must be found.
  G2 condition false positives    — generic words / misleading phrases must
                                    NEVER claim a condition (in particular,
                                    "something for my blood sugar problem"
                                    must not become "anemia").
  G3 meal_type                    — margin gate: genuine meals found,
                                    ambiguous words ("snack", "quick meal")
                                    rejected.
  G4 diet_preference              — 0.70 threshold: true preferences found,
                                    bare generic words rejected.
  G5 main_ingredient regression   — the original 20/20 suite's logic,
                                    spot-checked here.
  G6 required sample queries      — exact-field regression of the three
                                    canonical parse examples.

Run from the project root:
    python Nlp/test_robustness_all.py
"""

try:
    from Nlp.query_parser import parse_query
except ModuleNotFoundError:  # executed standalone with Nlp/ on sys.path
    from query_parser import parse_query

G1 = [  # (query, expected condition)
    ("i have diabetes", "diabetes"),
    ("diabetic meal plan", "diabetes"),
    ("food for high blood pressure", "hypertension"),
    ("something for blood pressure", "hypertension"),
    ("kidney disease diet", "chronic_kidney_disease"),
    ("food for kidney problems", "chronic_kidney_disease"),
    ("heart disease", "heart_disease"),
    ("iron deficiency food", "anemia"),
    ("I have anemia, need iron rich food", "anemia"),
    ("gluten intolerant", "gluten_intolerance"),
    ("lactose intolerant", "lactose_intolerance"),
    ("chronic kidney disease", "chronic_kidney_disease"),
    ("blood sugar", "diabetes"),
    ("high blood sugar", "diabetes"),
]

G2 = [  # (query, forbidden condition) — must NOT claim the forbidden one
    ("something for my blood sugar problem", "anemia"),
    ("i have a blood problem", "anemia"),
    ("heart healthy meal", "heart_disease"),
    ("i want to lose weight", "obesity"),
    ("i want weight loss", "obesity"),
    ("i am overweight", "underweight"),
    ("gluten free breakfast", "gluten_intolerance"),
    ("dairy free lunch", "lactose_intolerance"),
    ("sensitive stomach", "irritable_bowel_syndrome"),
    ("chronic disease", "chronic_kidney_disease"),
    ("high sugar breakfast", "diabetes"),
    ("low sugar dinner", "diabetes"),
    ("cut sugar from my diet", "diabetes"),
    ("sugar cake recipe", "diabetes"),
    ("sugar milk", "lactose_intolerance"),
    ("cholesterol", "high_cholesterol"),
    ("weight issue", "obesity"),
]

G3 = [  # (query, expected meal_type or None)
    ("breakfast", "breakfast"),
    ("morning meal", "breakfast"),
    ("lunch", "lunch"),
    ("afternoon food", "lunch"),
    ("dinner", "dinner"),
    ("supper", "dinner"),
    ("snack", None),
    ("quick meal", None),
    ("brunch", None),
]

G4 = [  # (query, expected diet_preference or None)
    ("vegetarian", "vegetarian"),
    ("vegan", "vegan"),
    ("chicken lover", "chicken_lover"),
    ("vegetable", None),
    ("diet", None),
    ("fish", None),
    ("chicken", None),
]

G5 = [  # (query, expected main_ingredient or None)
    ("chicken lunch", "chicken"),
    ("salmon dinner", "salmon"),
    ("tofu stir fry", "tofu"),
    ("refreshing salad", None),
    ("healthy snack", None),
    ("something light for dinner", None),
]

G6 = [  # (query, {field: expected})
    ("I want low sugar dinner for diabetic", {
        "condition": "diabetes", "meal_type": "dinner",
        "sugar_max": 15, "main_ingredient": None}),
    ("breakfast with 20 grams protein and no dairy", {
        "protein_min": 20, "meal_type": "breakfast",
        "allergy": ["milk"], "main_ingredient": None}),
    ("high protein chicken lunch, low sodium", {
        "protein_min": 20, "sodium_max": 600,
        "main_ingredient": "chicken", "meal_type": "lunch"}),
]


def main() -> None:
    passed = 0
    total = 0

    def check(q, field, expected, desc):
        nonlocal passed, total
        total += 1
        got = parse_query(q)[field]
        ok = got == expected
        passed += int(ok)
        print(f"  {q!r:36} {desc:16} got={got!r:14} "
              f"expected={expected!r:14} {'PASS' if ok else 'FAIL'}")

    print("=== G1: condition true positives ===")
    for q, expected in G1:
        check(q, "condition", expected, "condition")

    print("\n=== G2: condition false positives (forbidden) ===")
    for q, forbidden in G2:
        total += 1
        got = parse_query(q)["condition"]
        ok = got != forbidden
        passed += int(ok)
        print(f"  {q!r:36} condition={got!r:20} "
              f"must NOT be {forbidden!r:14} {'PASS' if ok else 'FAIL'}")

    print("\n=== G3: meal_type ===")
    for q, expected in G3:
        check(q, "meal_type", expected, "meal_type")

    print("\n=== G4: diet_preference ===")
    for q, expected in G4:
        check(q, "diet_preference", expected, "diet_preference")

    print("\n=== G5: main_ingredient regression ===")
    for q, expected in G5:
        check(q, "main_ingredient", expected, "main_ingredient")

    print("\n=== G6: required sample queries ===")
    for q, fields in G6:
        for field, expected in fields.items():
            check(q, field, expected, field)

    print(f"\n{passed}/{total} passed")
    if passed != total:
        print("(failures above are honest: the matcher still misjudges those "
              "cases; do not hide them)")


if __name__ == "__main__":
    main()
