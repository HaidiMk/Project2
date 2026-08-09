"""
Runnable smoke tests for the standalone Nlp/ query parser.

Run from the project root:
    python Nlp/test_query_parser.py
(or as a package module: python -m Nlp.test_query_parser)
"""

try:
    from Nlp.query_parser import extract_numeric_constraints, parse_query
except ModuleNotFoundError:
    from query_parser import extract_numeric_constraints, parse_query

TEST_QUERIES = [
    "I want low sugar dinner for diabetic",
    "breakfast with 20 grams protein and no dairy",
    "high protein chicken lunch, low sodium",
]

EXTRA_QUERIES = [
    "under 600 mg sodium dinner",
    "vegan lunch with 30 grams protein",
    "no nuts and no seafood for breakfast",
    "high fiber meal, at least 7 grams fiber",
    "I have high blood pressure, low calorie dinner",
]


def main() -> None:
    print("== required sample queries ==")
    for q in TEST_QUERIES:
        print(q, "->", parse_query(q))

    print("\n== extra checks ==")
    for q in EXTRA_QUERIES:
        print(q, "->", parse_query(q))


if __name__ == "__main__":
    main()
