"""
test_robustness.py — main_ingredient false-positive regression suite.

Verifies the semantic matcher's main_ingredient precision:

  Group A — generic queries with NO ingredient mentioned must NOT produce a
            main_ingredient (a generic word like "salad" / "soup" / "dish"
            must never turn into an inferred ingredient such as "cheese" /
            "pasta", which used to corrupt recipe filtering).
  Group B — queries that DO state an ingredient must still detect it.

Run from the project root:
    python Nlp/test_robustness.py
"""

try:
    from Nlp.query_parser import parse_query
except ModuleNotFoundError:  # executed standalone with Nlp/ on sys.path
    from query_parser import parse_query

GROUP_A = [  # expected main_ingredient: None
    "refreshing salad",
    "something light for dinner",
    "healthy snack",
    "quick breakfast idea",
    "comfort food for lunch",
    "warm soup",
    "spicy dish",
    "sweet treat",
    "low calorie meal",
    "filling dinner",
]

GROUP_B = [  # (query, expected main_ingredient)
    ("chicken lunch", "chicken"),
    ("salmon dinner", "salmon"),
    ("tofu stir fry", "tofu"),
    ("egg breakfast", "eggs"),
    ("rice bowl", "rice"),
    ("beef stew", "beef"),
    ("shrimp pasta", "shrimp"),
    ("lentil soup", "lentils"),
    ("avocado toast", "avocado"),
    ("quinoa salad", "quinoa"),
]


def main() -> None:
    passed = 0
    total = 0

    print("=== Group A: no ingredient stated -> main_ingredient must be None ===")
    for q in GROUP_A:
        total += 1
        got = parse_query(q)["main_ingredient"]
        ok = got is None
        passed += int(ok)
        print(f"  {q!r:32} -> main_ingredient={got!r:12} "
              f"{'PASS' if ok else 'FAIL'}")

    print("\n=== Group B: stated ingredient -> main_ingredient must be found ===")
    for q, expected in GROUP_B:
        total += 1
        got = parse_query(q)["main_ingredient"]
        ok = got == expected
        passed += int(ok)
        print(f"  {q!r:32} -> main_ingredient={got!r:12} "
              f"expected={expected!r:10} {'PASS' if ok else 'FAIL'}")

    print(f"\n{passed}/{total} passed")
    if passed != total:
        print("(failures above are honest: the matcher still misjudges those "
              "cases; do not hide them)")


if __name__ == "__main__":
    main()
