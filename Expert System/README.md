# Expert System

This folder is the core of the Smart Dietary Advisor: the rule engine that
decides which recipes are medically safe for a user, the two trained models
that score and personalize the safe ones, and a meal-plan builder on top.

## What's in here, at a glance

- **A rule-based safety filter** — hand-written medical, allergy, halal, and
  diet-preference rules. This is the hard gate: nothing unsafe for a user's
  stated conditions or allergies gets past it.
- **Two trained models** — one learns how well a recipe's nutrition fits a
  user's medical conditions, the other learns ingredient meaning so it can
  match recipes to a free-text taste description ("I love garlic, dislike
  seafood"). Both feed into the final recipe ranking alongside the rules.
- **A meal planner** — builds a full day (or week) of breakfast/lunch/dinner
  out of the same safe, ranked recipes, without repeats and within a calorie
  target.

The recipe database behind all of this is the cleaned Food.com dataset:
**384,541 recipes** with full nutritional info.

## Requirements

```bash
pip install pandas numpy torch scikit-learn
```

## Running it

```bash
# interactive command-line demo
python main.py

# a ready-made example (diabetes + hypertension)
python main.py --demo

# a quick look at the dataset
python eda_report.py
```

## Folder layout

```
Expert System/
├── main.py              ← entry point
├── eda_report.py         ← dataset overview
├── core/                 ← user profile + shared constants
├── rules/                ← medical, allergy/halal, and preference rules
├── engine/                ← the filtering + scoring engine
├── ui/                    ← command-line interface
├── data/                  ← the recipe dataset and trained model files
├── ml/
│   ├── health_classifier/  ← Model 1 — nutrition/condition suitability scorer
│   └── word2vec/            ← Model 2 — ingredient embeddings + taste matching
└── planner/               ← daily/weekly meal-plan builder
```

## What it covers

The rule engine currently has **28 medical rule entries** spanning children,
teens, adults, and elderly users (things like diabetes, hypertension, kidney
disease, pregnancy, and more), plus **7 allergy categories** (peanuts, milk,
eggs, seafood, soy, gluten, sesame) and an always-on halal filter.

## Checking it works

```bash
cd ../TOPSIS
python sanity_check.py                                    # full pipeline regression suite
cd ..
python "Expert System/ml/word2vec/test_alternatives.py"   # taste-based alternatives smoke test
python "Expert System/planner/test_meal_planner.py"        # meal planner smoke test
python Nlp/test_query_parser.py                            # smart-search parser checks
```

## More detail

For the full technical write-up — model training, evaluation numbers, API
contracts, and known limitations — see the `README.md` at the repository
root.
