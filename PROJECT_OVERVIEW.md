# Smart Dietary Advisor — Project Overview

Technical engineering record for the Smart Dietary Advisor system: a recipe
recommendation pipeline that filters a large recipe corpus for medical safety,
ranks it for user goals, and personalizes it with two trained models.

This document is written for someone unfamiliar with the project (a committee
member or new teammate). It describes what the system does, why it is built the
way it is, how both models were trained and evaluated, the bugs found during
manual testing and how they were fixed, and how to verify the system yourself.

---

## 1. Introduction

**What it is.** Smart Dietary Advisor takes a user's profile (age, height,
weight, gender, medical conditions, allergies, dietary goal) plus an optional
free-text taste statement ("I love garlic and spicy food, dislike seafood") and
returns a ranked list of safe, goal-aligned, taste-matched recipes.

**The pipeline** (four stages, all applied to every candidate recipe):

```
User profile + taste text
        │
        ▼
1. Expert System (rules)        — hard safety filtering: medical rules,
        │                         allergies, diet preferences, halal checks
        ▼
2. TOPSIS (goal scoring)        — multi-criteria ranking toward the user's
        │                         goal (weight loss, muscle gain, ...)
        ▼
3. Model 1 — health_classifier  — neural multi-label classifier; scores each
        │                         recipe's nutritional suitability for the
        │                         user's specific medical conditions
        ▼
4. Model 2 — word2vec taste     — ingredient embeddings + user free-text
        │                         vector; scores how well a recipe matches
        │                         the user's stated likes/dislikes
        ▼
Final blended ranking           — weighted combination of TOPSIS, AI health
                                  score, expert score, and taste score
```

The final blend is:

- without taste text:  `final = 0.4·TOPSIS + 0.4·AI_health + 0.2·expert`
- with taste text:     `final = 0.35·TOPSIS + 0.35·AI_health + 0.15·expert + 0.15·taste`

**Data.** Recipes come from the Food.com dataset (Kaggle — *Food.com Recipes
and Interactions*, `RAW_recipes.csv`, ~672 MB raw). After cleaning the corpus
contains **384,541 recipes** with 9 normalized nutritional features per recipe
(`cleaned_recipes.csv`).

---

## 2. Why the original architecture was rejected

The first version of the system was a **rule-based expert system + TOPSIS
multi-criteria ranking**, optionally blended with a weak collaborative-filtering
component (NCF) trained on recipe ratings.

The committee's verdict was that this did not constitute **real trained AI**:

- The expert system is hand-written rules (medical thresholds, allergy
  blacklists) — deterministic and explainable, but not learned.
- TOPSIS is a classical decision-science ranking method with fixed weights.
- The NCF component was weak, trained on sparse interaction data, and added
  little measurable value.

**Strategic decision:** keep the rule layer (it is clinically justified and
explainable), but add **two genuinely trained models** that give the system a
defensible machine-learning core:

1. **Model 1 — health_classifier**: a neural network that *learns* what
   "nutritionally safe for condition X" means from data.
2. **Model 2 — word2vec**: a neural distributional-semantics model (skip-gram
   Word2Vec) that *learns ingredient meaning* from the recipe corpus and powers
   the taste-personalization layer.

The rules were kept as a hard safety gate (filtering), and the two models were
integrated as scoring layers blended into the final ranking.

---

## 3. Model 1 — health_classifier (multi-label neural classifier)

**File locations:** `Expert System/ml/health_classifier/` (code),
`Expert System/data/health_classifier.pt` + `health_scaler.pkl` +
`health_classifier_labels.json` + `health_classifier_thresholds.json` (artifacts).

### Problem it solves

Given a recipe's 9 nutritional features, predict which of 22 medical conditions
the recipe is suitable for. The expert system already filters hard violations
via rules; Model 1 provides a *learned, graded* suitability score that captures
the rating-quality signal the rules cannot.

### Why multi-label classification

A recipe is not "good" or "bad" in one dimension — a single recipe may be
suitable for a diabetic while being unsuitable for someone with hypertension.
Each condition is therefore a separate binary problem, and the network outputs
**22 independent probabilities** (multi-label), not a single class.

### Soft-label design (rule × rating)

Binary 0/1 labels would throw away quality information. Instead, labels are
**soft targets** built by `build_labels.py` from the medical rules combined with
community ratings:

```
rule_label        ∈ {0, 1}      (numeric rule thresholds, applied literally)
normalized_rating = (Rating − 1) / 4            (1..5 → 0..1)
final_label       = rule_label · (0.5 + 0.5 · normalized_rating)
```

So an unsafe recipe is always 0 (hard safety gate preserved), while a safe
recipe ranges from 0.5 (poorly rated) to 1.0 (top rated). The final labels live
in `{0} ∪ [0.5, 1.0]` and are trained with BCE, which supports fractional
targets. 24 labels are generated; 2 (`label_lactose_intolerance`,
`label_gluten_intolerance`) are excluded from training because they are
near-constant (~99.8% positive) and carry no signal — those two conditions
remain handled by text filtering in the expert system. The trained label set is
**22 conditions**.

### Network architecture

```
9 nutritional features
  → Linear(64) + ReLU + Dropout(0.2)
  → Linear(32) + ReLU + Dropout(0.2)
  → Linear(16) + ReLU
  → Linear(22)                 (raw logits)
```

- Loss: `BCEWithLogitsLoss` with per-condition `pos_weight = n_neg / n_pos`
  (capped at 10) to counter class imbalance.
- Inputs standardized with `StandardScaler` **fit on the training split only**
  (no data leakage), saved as `health_scaler.pkl` for inference.
- Training uses early stopping and saves the best weights to
  `health_classifier.pt`.

### Evaluation methodology

- **Split:** 70/15/15 train/validation/test with a fixed seed. sklearn has no
  multi-label stratified split, so a practical proxy is used: stratified
  splitting on the binarized `label_diabetes` (≥ 0.5) — the most
  representationally important condition — which keeps rare conditions
  balanced across all three splits. Split balance is verified by printing each
  condition's distribution across splits.
- **Why high AUC is expected:** the labels are derived from the same medical
  rules the system already uses, so the classification problem is nearly
  deterministic from the features. AUC should therefore be high; this is a
  property of the label construction, not an artifact of evaluation.
- **Threshold tuning per condition** (`tune_thresholds.py`): for each of the
  22 conditions, a grid of thresholds 0.10–0.95 (step 0.05) is scanned on the
  **validation set only** (the test set is never touched for tuning). The
  threshold with the highest F1 is chosen; ties resolve to the lower
  (more conservative) threshold — recall is preferred, "doubt favors the
  patient". Results saved to `health_classifier_thresholds.json`.
- **Test-set reports** (`evaluate_classifier.py`, results in
  `ml/health_classifier/results/metrics*.md|json`): per-condition accuracy,
  precision, recall, F1, and AUC, both at the flat 0.5 threshold and at the
  tuned per-condition thresholds.

**Final metric (after threshold tuning): macro-F1 = 0.9785 on the held-out
test set.** At the flat 0.5 threshold the classifier is already strong; tuning
recovers further F1 for the rarer conditions.

### Integration into the pipeline

`inference.py` loads the model, scaler, and label keys once (singleton) and
computes `ai_health_score`:

```
ai_health_score(recipe) = mean( sigmoid(logits) over the user's condition keys )
```

If no condition keys match the trained set, the mean over all 22 conditions is
used as a neutral fallback. The score feeds the blend as the `0.4`/`0.35` AI
component.

---

## 4. Model 2 — word2vec (ingredient embeddings + taste personalization)

**File locations:** `Expert System/ml/word2vec/` (code),
`Expert System/data/word2vec_ingredients.model` +
`recipe_taste_embeddings.pkl` (artifacts).

### The personalization gap it fills

The rule layer and Model 1 answer "is this recipe safe and appropriate?" — but
not "will this user like it?". Taste is personal, and the project needed a
learned representation of ingredient meaning to compare user preferences
against recipe content. This is the personalization layer.

### Why Word2Vec (distributional semantics)

The distributional hypothesis states that words appearing in similar contexts
have similar meanings. Ingredient tokens that co-occur in recipes (e.g.
`soy_sauce` and `ginger`) should end up close in embedding space — exactly what
a taste comparison needs. Word2Vec is a shallow neural model (a self-supervised
predictive embedding method) that trains in minutes on modest hardware, and the
trained model is small enough (~5.3 MB) to commit to the repository.

### Training setup

`build_taste_embeddings.py` cleans each recipe's ingredient list (lowercase,
strip parenthetical notes, join multi-word ingredients with underscores, e.g.
`olive oil` → `olive_oil`), drops the 1,232 recipes with empty ingredient lists,
and trains gensim Word2Vec:

| Hyperparameter | Value |
|---|---|
| Algorithm | skip-gram (`sg=1`) |
| Vector size | 150 |
| Window | 6 |
| Min count | 5 |
| Epochs | 15 |
| Seed | 42 |
| Workers | `os.cpu_count()` |

Result: **4,504-token vocabulary** (from 6,896 distinct cleaned tokens; tokens
below min_count are dropped). Full pipeline runtime ≈ 61 s (14 s cleaning +
26 s training + 15 s embedding pass). Training sanity check: `most_similar`
top-5 for known ingredients (`chicken_breast`, `olive_oil`, `garlic`, ...).

### Recipe embeddings

Each recipe's embedding = **mean of its ingredient vectors** (only tokens in
vocabulary). Recipes with no in-vocabulary ingredient get no embedding. Result:
`recipe_taste_embeddings.pkl` — a dict mapping RecipeId → 150-dim float32
vector, covering **383,293 recipes**.

### User free-text → vector

`taste_inference.py` turns a sentence like "I love garlic and spicy food,
dislike seafood" into a vector through three layers:

1. **Tokenization** — lowercase, strip punctuation, keep single words *and*
   adjacent pairs joined with underscores (`sour cream` → `sour` + `cream` +
   `sour_cream`).
2. **Clause splitting and polarity** — text is split on `;`, `.`, and
   standalone `but`; each clause is classified as liked or disliked via trigger
   phrases (`like`, `love`, ... vs `dislike`, `don't like`, `hate`, `avoid`,
   `not a fan of`, ...).
3. **Token resolution, in order:**
   - *Exact* match in the Word2Vec vocabulary (`garlic` → `garlic` vector);
   - *Concept map* — 43 curated food concepts (`spicy`, `italian`,
     `vegetables`, `sweets`, ...) each mapping to verified vocab tokens;
   - *Fuzzy* match (rapidfuzz) as a last resort, only for tokens ≥ 5 chars
     with similarity ≥ 0.82.

The user vector is the **mean of liked-token vectors minus the mean of
disliked-token vectors**. Each recipe's taste score is the cosine similarity
between the recipe embedding and this user vector, normalized to [0, 1]
(neutral = 0.5 when no tokens are recognized). The score feeds the blend as the
`0.15` taste component.

---

## 5. Iterative fixes and lessons learned

Four real bugs were found during manual testing, each fixed and verified before
the next was pursued. This section records what was observed, the root cause,
and the verification that confirmed the fix.

### (a) Dislike/negation bug — disliked ingredients treated as liked

**Observed:** A user text "dislike garlic and ginger" still produced
garlic/ginger-named recipes at the top of the ranking with high taste scores
(e.g. *Orange-Ginger Glazed Cornish Game Hens* scored 0.7582).

**Root cause:** The whole sentence was treated as one bag of tokens with a
single positive polarity; negation ("dislike") did not flip the polarity of its
clause, so disliked ingredients were *added* to the user vector like liked ones.

**Fix:** Clause-level polarity parsing: split on `;`, `.`, and standalone
`but`; detect trigger phrases (`dislike`, `don't like`, `do not like`, `hate`,
`avoid`, `not a fan of`); build the user vector as **liked-clause vectors minus
disliked-clause vectors** (pure-avoidance text with no liked terms becomes the
negative of the disliked vector; a degenerate zero vector falls back to neutral
0.5).

**Verification:**
- The same game-hen recipe dropped from **0.7582 → 0.4224** under the dislike
  text; no garlic/ginger-named recipe remained in the top-15 (regression check
  added to `sanity_check.py`).
- Cosine anchors established: `cos("I love garlic", wv["garlic_cloves"]) =
  **+0.9192**` and `cos("I dislike garlic", ...) = **−0.9192**` — symmetric
  around zero, confirming negation flips the vector exactly.
- Bonus check: "I dislike seafood" suppressed tuna recipes (Pico De Gallo
  Seared Ahi taste **0.8417 → 0.4427**).

### (b) rapidfuzz WRatio false positives — "love" → "cloves"

**Observed:** "I love garlic" boosted *cloves* (as in cloves of garlic? no — as
in the spice): "love" fuzzy-matched `clove`/`garlic_cloves`, and "food"
matched `best_foods_mayonnaise`.

**Root cause:** The fuzzy matcher used rapidfuzz's default **WRatio** scorer,
which applies partial-ratio matching — short common English words match as
contiguous substrings of longer tokens (`love ⊂ cloves`, `food ⊂
best_foods_mayonnaise`).

**Fix:** Use plain `fuzz.ratio` (full-string Levenshtein similarity) as the
scorer and exclude tokens shorter than 5 characters from fuzzy matching
entirely (`FUZZY_MIN_TOKEN_LEN = 5`). Fuzzy is used only as a last-resort
fallback after exact match and concept map.

**Verification:**
- The ±0.9192 anchors were unchanged (garlic still resolves exactly).
- Smoke tests: `garlik` → `garlic` (83.3% similarity), `chiken` → `chicken`
  (92.3%); no false matches for "love" or "food"; byte-identical results for
  exact-match-only inputs.
- Cost measured: resolution cost rose from ~0.03 ms to ~4.64 ms per call
  (fuzzy scoring is more expensive) — acceptable at ranking-time scale.

### (c) Concept-map enrichment with compound tokens — and the `celery_ribs` trap

**Observed:** Free-text concepts like `italian` or `mexican` expanded to
concept maps built from *bare* words that mostly do not exist in the vocab
(parmesan, jalapeno, chili, mozzarella, ...), because the data cleaning joins
multi-word ingredients with underscores. A read-only vocabulary audit showed
**all 31 missing bare words had exact count 0** in the corpus — their signals
live in compound tokens (`parmesan_cheese`, `jalapeno_chile`, `chili_powder`,
...), not in bare forms. This was data sparsity + compound joining, not a
cleaning bug.

**Fix:** Enriched the concept map with **membership-verified compound tokens**
(italian += `parmesan_cheese`, `parmigiano-reggiano_cheese`,
`mozzarella_cheese`; mexican += `jalapeno_chile`, `chipotle_chile_in_adobo`,
...; dessert += `cocoa_powder`; dairy += `cheddar_cheese`, ...). Every token was
checked against the live model vocabulary before being added.

**Semantic trap documented:** `celery_ribs` (3,496 occurrences) means *celery
stalks*, not BBQ ribs — so bare `ribs` must never be added; grilled-rib signals
come only from `country-style_pork_ribs` / `beef_ribs`. Similarly there is no
generic `steak` token, only specific cuts (`beef_flank_steak`, `beef_round_steak`,
`ham_steak`, ...). And `fish_sauce` (2,054) is a *seasoning*, not a protein — it
belongs in asian/salty, never in seafood/healthy.

**Verification:** concept vector density grew (italian 9→12 tokens, mexican
9→19, dessert 9→12, asian 16→19, ...); the full `sanity_check.py` suite passed
with all outputs identical to the pre-change baseline.

### (d) General/plural category-word gap — "vegetables" → "vegetable_suet"

**Observed:** "I love vegetables" boosted *Vegetable Suet* (a niche baking
fat) — fuzzy matching returned `vegetable_suet` (high spelling similarity,
wrong meaning), and in a manual ranking it pushed *Zone Diet - Perfect
Pancakes* (a pancake recipe) into the top-2 with taste 0.6746. Similarly,
"sweets" fuzzy-matched `swedes` (a root vegetable).

**Root cause:** Broad category words (`vegetables`, `fruits`, `meat`, `nuts`,
...) have no exact vocab token and were not concept keys, so they fell through
to the spelling-only fuzzy fallback with no semantic guardrail.

**Fix:** Added 28 new concept keys (43 total) mapping each category word —
including singular/plural aliases — to curated, membership-verified token lists:
`vegetables`/`veggie`/`veggies`/`vegetable`/`veg` (25 tokens), `fruits`/`fruit`
(22), `meat`/`meats` (12), `nuts`/`nut` (7), `beans`/`bean` (9),
`grains`/`grain` (10), `herb` (12), `spices`/`spice` (17), `sweets` (11),
`legumes`/`legume` (9), `poultry` (5), `fish` (8), `shellfish` (6),
`carbs`/`carb` (9), `berry` (4), `whole_grains` (7). Words left deliberately
unmapped: `green` (too ambiguous — color vs green onions vs green olives; the
existing fuzzy match to `greens` is directionally sensible), and items with no
clean token set (`almonds`, `bread`, `noodles`, ...) where returning "no match"
is safer than a wrong guess.

**Verification:**
- The audit table (85 candidate words) confirmed every previous false positive
  now resolves via CONCEPT_MAP, not fuzzy; `potatoes` stayed an exact match.
- Manual before/after on the profile (22yo female, seafood_allergy,
  weight_gain, `user_taste_text="I love vegetables"`): before, the pancake
  recipe ranked #2 (taste 0.6746 via suet); after, vegetable-forward recipes
  led with sensible scores (Iron Soup taste 0.9012, White Chili 0.8918).
- Full `sanity_check.py` suite passed with outputs byte-identical to baseline;
  the ±0.9192 anchors were unchanged.

---

## 6. Repository structure (models)

`Expert System/ml/` was split into two clearly-named model folders (the model
code was previously mixed together in one flat folder):

```
Expert System/
├── data/                                  # model artifacts (committed, except large ones)
│   ├── health_classifier.pt               # Model 1 weights
│   ├── health_scaler.pkl                  # Model 1 input standardizer
│   ├── health_classifier_labels.json      # 22 condition keys (output order)
│   ├── health_classifier_thresholds.json  # per-condition tuned thresholds
│   ├── word2vec_ingredients.model         # Model 2 trained model (~5.3 MB)
│   └── recipe_taste_embeddings.pkl        # 383,293 recipe vectors (~230 MB, gitignored)
├── ml/
│   ├── setup_artifacts.md                 # artifact inventory + regeneration guide (both models)
│   ├── data/labeled_recipes.csv           # Model 1 training labels (~55 MB, gitignored)
│   ├── health_classifier/                 # ⭐ Model 1 — multi-label neural classifier
│   │   ├── build_labels.py                #   soft-label generation (rule × rating)
│   │   ├── health_classifier.py           #   model definition + training + 70/15/15 split
│   │   ├── inference.py                   #   ai_health_score (production singleton)
│   │   ├── tune_thresholds.py             #   per-condition threshold calibration (val only)
│   │   ├── evaluate_classifier.py         #   test-set reports (flat 0.5 / --tuned)
│   │   ├── results/                       #   metrics.md / metrics_tuned.md (+ json)
│   │   └── MODEL_DOCUMENTATION.md         #   detailed Arabic model documentation
│   └── word2vec/                          # ⭐ Model 2 — ingredient embeddings + taste
│       ├── build_taste_embeddings.py      #   corpus cleaning + Word2Vec training
│       ├── taste_concepts.py              #   concept map (43 concepts, vocab-verified)
│       └── taste_inference.py             #   user text → vector → recipe taste scores
└── engine/, rules/, ui/, core/            # expert system (unchanged)
```

---

## 7. Known current limitations

Honest account of what the system does not do yet:

- **English-only taste input.** The free-text taste parser operates on
  English. There is no Arabic (or other-language) support yet, even though the
  expert-system CLI itself is Arabic.
- **Fuzzy matching is English-lexical and intentionally conservative.** It
  matches on spelling similarity only and deliberately rejects cross-language
  near-matches (e.g. `patata` vs `potato` fall below the 0.82 threshold and
  resolve to nothing) — this is by design; a wrong guess is worse than no
  match.
- **No coverage for ingredients rare in this dataset era.** Words like
  `hummus`, `chia`, `agave`, `smoked_paprika`, `tamarind` (bare), `almonds`
  (bare), and generic `steak` have no vocabulary token — they are absent or too
  rare in this Food.com snapshot to pass `min_count=5`. Their concepts are
  mapped via related tokens where defensible, and left unmapped where not.
- **Taste text is not required.** If no text is provided, the system falls
  back to the 0.4/0.4/0.2 blend; the taste layer simply does not participate.
- **Model 1's high AUC is a property of label construction.** The labels are
  derived from the same clinical rules used for filtering, so the learned task
  is nearly deterministic; the classifier is a learned, graded re-ranking
  signal, not an independent medical oracle.
- **Ranking is corpus-bound.** Recommendations are limited to the 384,541
  recipes in the cleaned Food.com corpus; the vocabulary (4,504 tokens) is
  fixed at training time.

---

## 8. How to verify this yourself

**End-to-end regression suite** — run from the `TOPSIS/` folder
(~15–20 minutes; requires the data artifacts):

```powershell
cd TOPSIS
$env:PYTHONIOENCODING='utf-8'
python sanity_check.py
```

It verifies, among other things: all three profiles (diabetic 5,665 safe /
muscle_gain 294,548 / elderly 3,019 safe recipes) run without errors and
without NaN scores; blend backward-compatibility (`final == 0.4·TOPSIS +
0.4·AI + 0.2·expert` with no taste text); taste-score integration (column in
[0,1], ranking changes, nonsense text → neutral 0.5); the dislike-parsing
regression (no garlic/ginger-named recipe in the top-15 under a dislike text);
and the concept/fuzzy smoke checks. Expected result: **ALL CHECKS PASSED**.

**Key regression anchors** (also inside the suite):

```
cos("I love garlic",    wv["garlic_cloves"]) == +0.9192
cos("I dislike garlic", wv["garlic_cloves"]) == -0.9192
```

**Artifact regeneration** — see `Expert System/ml/setup_artifacts.md` for the
full inventory (what is committed vs regenerated locally) and the exact
commands, in order: `cleaner.py` (cleaned CSVs), then
`ml/health_classifier/build_labels.py` (training labels), then
`ml/word2vec/build_taste_embeddings.py` (word2vec + recipe embeddings, ~61 s),
and `ml/health_classifier/health_classifier.py` + `tune_thresholds.py` +
`evaluate_classifier.py` for Model 1 (detailed steps in
`ml/health_classifier/MODEL_DOCUMENTATION.md`).
