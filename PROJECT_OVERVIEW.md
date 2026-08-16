# Smart Dietary Advisor — Project Overview

Technical engineering record for the Smart Dietary Advisor system: a recipe
recommendation pipeline that filters a large recipe corpus for medical safety,
ranks it for user goals, and personalizes it with two trained models — exposed
to real clients through a Django REST API.

This document is written for someone unfamiliar with the project (a committee
member or new teammate). It describes what the system does, why it is built the
way it is, how both models were trained and evaluated, the bugs found during
manual testing and how they were fixed, how the backend API is put together,
and how to verify the system yourself.

---

## 1. Introduction

**What it is.** Smart Dietary Advisor takes a user's profile (age, height,
weight, gender, medical conditions, allergies, dietary goal) plus an optional
free-text taste statement ("I love garlic and spicy food, dislike seafood") and
returns a ranked list of safe, goal-aligned, taste-matched recipes. A mobile
app is the primary end-user client of the backend API described in §9; a
separate read-only admin dashboard (§10) is planned as a second consumer.

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
(`cleaned_recipes.csv`), plus recipe images (§9) and a larger, less-cleaned
435,009-row superset (`cleaned_recipes_full.csv`) kept alongside it but not
used by the live pipeline.

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

A separate SHAP-based explainability layer (`ml/health_classifier/explain.py`)
computes per-condition, per-feature reasons for a recipe's score on demand
(`explain_health_score(recipe, condition_keys, top_n)`); it is used by the
backend's recipe-explanation endpoint (§9) and carries a one-time ~8-second
warmup cost the first time it runs in a process.

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

## 5. Smart Search (Nlp/) — a UX feature, not a trained model

**What it is.** `Nlp/` is a search-bar convenience layer added at the repo
root, next to `Expert System/` and `TOPSIS/`. It wraps a **pretrained
Sentence-BERT model** (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim
embeddings) to translate a free-text query such as *"low sugar dinner for
diabetic"* into structured filters, then hands those filters to the
**existing, unmodified** `DietaryExpertSystem.filter_recipes()`. It is the
engine behind the backend's `GET /api/recipes/search/` endpoint (§9).

**Architecturally separate from the two trained models.** This module is NOT a
third trained model and should not be mistaken for one:

- The project's two trained AI components are **Model 1 (health_classifier)**
  and **Model 2 (word2vec)** — both trained on this project's own data, both
  contributing *learned* scores to the final ranking blend.
- `Nlp/` uses a **general-purpose pretrained model with zero fine-tuning on
  this project's data**. It makes **no safety and no ranking decision of its
  own**: it only assembles the pieces of a `UserProfile` (conditions,
  allergies, meal type, macro limits, ...) from the query text and calls the
  rule engine exactly as any other caller would. Every safety filter, every
  TOPSIS score, and every learned score comes from the unchanged existing
  pipeline. Its purpose is purely to make the search bar usable with natural
  language.

**How it works.** `Nlp/query_parser.py` maps query phrases onto the project's
own term vocabulary (`Nlp/vocab_terms.py` — disease, allergy, preference, and
meal-type term lists, kept in sync with `Expert System/core/constants.py`);
the sentence embeddings are used to resolve free-text fragments to those known
terms. `Nlp/pipeline.py` exposes `search_recipes(query_text, base_profile,
expert_system, top_n)`, which merges the parsed filters into the caller's base
`UserProfile` and calls the engine. It never modifies the engine's inputs,
rules, or output ordering.

**Integration fixes (both entirely inside `Nlp/`).** The module was originally
written against an older `DietaryExpertSystem` API from before the NCF
component was replaced by health_classifier. Two mismatches surfaced during
integration testing, both fixed purely within `Nlp/`:

1. `DietaryExpertSystem(df, train_ncf=False)` — the `train_ncf` parameter no
   longer exists after NCF removal → fixed to `DietaryExpertSystem(df)`.
2. `result["ncf_active"]` — no longer present in the engine's result dict →
   replaced with a **live-derived** `"ai_health_scoring"` flag defined as
   `"_ai_health_score" in result["safe_recipes"].columns` — i.e. reported from
   the actual engine output rather than a hardcoded value.

Neither fix required any change inside `Expert System/` — the engine API
stayed untouched and the module was adapted to it.

**Files:** `pipeline.py`, `query_parser.py`, `vocab_terms.py` (core),
`check_nlp.py` (internal consistency diagnostic), and five test files
(`test_query_parser.py`, `test_pipeline.py`, `test_medical_safety.py`,
`test_robustness.py`, `test_robustness_all.py`). Test commands are in §13.

---

## 6. Iterative fixes and lessons learned

Five real issues were found — four bugs during manual testing and one
allergen-coverage gap found by a systematic read-only audit — each fixed and
verified before the next was pursued. This section records what was observed,
the root cause, and the verification that confirmed the fix.

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

### (e) Allergen name-matching gaps found via systematic investigation

**Observed:** A systematic read-only scan of all **6 allergy categories**
(peanuts, milk, eggs, seafood, soy, gluten) over the full 384,541-recipe
corpus found a small set of recipes that genuinely contained an allergen in
their ingredient text yet slipped past every safety net.

**Methodology:** for each category, recipes were checked against (1) the
column-based check (`HasNuts`, `HasLactose`, `HasSeafood`, `HasSoy`,
`HasGluten`), (2) the ingredient-text regex, and (3) the name regex. Every
name-flagged recipe was then classified as **TRUE GAP** (allergen present in
ingredient text, missed by all three nets), **NAME-ONLY** (allergen-adjacent
name, ingredients genuinely clean — safe to show), or **AMBIGUOUS**
(unverifiable data). The classification was deliberately designed to avoid
over-blocking legitimately safe recipes.

**Root cause — the word-boundary regex.** The engine matches each blocklist
term as `\bterm\bs?\b` (word boundaries on both sides, optional trailing "s"
— the documented Scallops fix). This cannot match an allergen word embedded
inside a larger word or compound: `\bpeanut\b` misses `peanutty`, `\begg\b`
misses `eggnog`/`eggshell`, and so on.

**Scale and causes — 10 TRUE GAP recipes out of 384,541** (2 peanut, 8 egg),
concentrated in three distinct word-form causes:

1. **Regional English synonyms** — `groundnut(s)`, the British/Indian term
   for peanuts (RecipeIds 9878, 521164).
2. **Compound commercial ingredient names** — `eggnog ice cream` as a single
   ingredient token (RecipeIds 77596, 77790, 185837, 200057, 262370, 420103).
3. **Egg-derived ingredient words** — `eggshells` (RecipeIds 90358, 537278).

A structural finding from the same investigation: `cleaned_recipes.csv` has
**no `HasEggs` column at all** — unlike the other five categories, egg-allergy
protection relies entirely on text matching, with no column-based safety net.
(The column map in `halal_and_allergies.py` references `HasEggs`, but no such
column exists in the data, so the engine's column check for eggs is a no-op.)

**Fix:** three terms added to the single source of truth in
`Expert System/rules/medical_rules.py` — `groundnut` in
`MEDICAL_RULES["nut_allergy"]["strict_block"]`, and `eggnog` + `eggshell` in
`MEDICAL_RULES["egg_allergy"]["strict_block"]`. Both the medical-condition
filters and the allergy filters (`ALLERGY_RULES` in `halal_and_allergies.py`)
reuse these exact lists, so no duplication was needed. The pattern builder's
automatic trailing `s?` covers the plural forms (`groundnuts`, `eggshells`).
**Deliberate exclusions:** `nutty` and `peanutty` were NOT added — "Nutty
Chocolate Mint Fudge" and similar NAME-ONLY recipes have genuinely clean
ingredients, and a prefix/adjacent-form rule would have over-blocked roughly
2,900 recipes flagged under the audit model (≈2,150 once the engine's
unconditional halal/pet/non-food filters are accounted for). This mirrors the
project's conservative-fix philosophy — the same spirit as the Scallops `s?`
fix and the concept-map membership verification. RecipeId 332170 ("Tandoori
Pork Saute", whose ingredient text contains "nutty rice") was left untouched
as AMBIGUOUS.

**Verification:**
- All 10 target RecipeIds confirmed **blocked** through the real
  `DietaryExpertSystem` for the relevant allergic profile. (Honest note: five
  of the ten — the alcoholic eggnog recipes — were in practice already dropped
  by the engine's unconditional halal ingredient filter via brandy/rum/bourbon
  terms; the fix guarantees all ten are now blocked through the allergy path
  itself. The genuinely new protection is five recipes: 9878, 521164, 420103,
  90358, 537278.)
- The previously-safe NAME-ONLY recipes — "Peanutty Oatmeal Cookies", "Nutty
  Chocolate Mint Fudge", "Wild Rice Almondine", "Catfish Almondine",
  "Eggless Homemade Ice Cream", "Flourless Chocolate Cake", "Wheatberry
  Salad", ... — confirmed **still available** to the relevant allergic users:
  no over-blocking.
- Full 6-category re-scan after the fix: **TRUE GAP = 0** in every category.
  The `eggnog` term additionally blocks ~200 eggnog-named recipes whose
  ingredient text shows no egg term (mostly pudding-mix/nutmeg shortcut
  recipes and incomplete data) — an intended consequence of treating eggnog
  as egg-containing.
- `TOPSIS/sanity_check.py` regression suite re-run end-to-end:
  **byte-identical to baseline** — same safe counts (5,665 / 294,548 /
  3,019), same score ranges and anchors, **ALL CHECKS PASSED**.

---

## 7. suggest_alternatives() — taste-based safe alternatives (standalone, opt-in)

**What it is.** `Expert System/ml/word2vec/alternatives.py` closes the
"blocked recipe, then what?" gap: until now, a recipe blocked by an allergen
or medical-rule violation simply disappeared from the user's results — no
substitute was offered. Given a blocked recipe and a user's own already
computed safe set (the output of
`DietaryExpertSystem.filter_recipes()['safe_recipes']`), it ranks that safe
set by taste similarity to the blocked recipe and returns the top-N as
suggested alternatives.

> **Not the same thing as the backend's `alternatives/` endpoint.** The
> live `GET /api/recipes/<id>/alternatives/` endpoint documented in §9 does
> not currently call this function — it re-ranks the user's safe set by
> TOPSIS with the requested recipe excluded, a simpler goal-based approach.
> `suggest_alternatives()` is the taste-similarity-driven module described
> below; it exists in the codebase, is fully tested standalone, but is not
> yet wired into that endpoint.

**Medical safety is inherited, not re-implemented.** Candidates are drawn
ONLY from the caller-supplied safe set — no recipe outside the user's safe
frame can ever be returned. The module adds no safety logic of its own; it
can only recommend recipes the existing pipeline already cleared.

**Method.** Cosine similarity between the blocked recipe's L2-normalized
Word2Vec taste embedding and each candidate's embedding, vectorized against
one process-wide aligned matrix of all 383,293 available recipe embeddings
(`recipe_taste_embeddings.pkl`). When taste similarity cannot be trusted,
the module automatically switches to a nutrition-only fallback: z-scored
Euclidean distance over the same 9 `NUTRITION_COLS` used by
health_classifier (corpus mean/std cached from `cleaned_recipes.csv`),
scored as `1 / (1 + distance)` so that higher = more similar.

**The three safeguards:**

1. **Sparse-ingredient recipes bypass taste similarity automatically.** The
   exploration PoC found that recipes with very few in-vocab ingredient
   tokens (e.g. a 2-ingredient list) produce flat, near-tied taste rankings
   that are nutritionally nonsensical — the scores differ only in the noise
   digits. A blocked recipe with no embedding, or with fewer than
   `min_vocab_tokens` (default 4) in-vocab tokens, therefore takes the
   nutrition fallback instead, and the returned `"method"` field tells the
   caller which path fired. (The fallback also fires if the taste path finds
   zero eligible candidates, i.e. none of the safe recipes have embeddings.)
2. **Deterministic tie-breaking.** Real corpus embeddings produce near-tied
   cosine scores; scores within 1e-4 of each other are sub-ordered by
   ascending z-scored nutrition distance, so the output order never depends
   on dict/array iteration order.
3. **Self-exclusion.** The blocked recipe's own RecipeId is always removed
   from its alternatives, and safe-set rows with unparseable RecipeIds are
   dropped before ranking.

Input validation follows the project's explain.py discipline: missing
required columns or non-numeric nutrition values raise `ValueError` —
nothing is silently imputed.

**API shape.** `suggest_alternatives(blocked_recipe, safe_recipes_df,
top_n=3, min_vocab_tokens=4) -> dict` — `blocked_recipe` is a pandas Series,
one-row DataFrame, or plain dict supplying `RecipeId`, `Name`,
`IngredientsList`, and the 9 nutrition columns; `safe_recipes_df` must
contain `RecipeId`, `Name`, and the same 9 nutrition columns. Returns a
JSON-serializable dict: `"method"` (`"taste_similarity"` or
`"nutrition_fallback"`), `"blocked_recipe"` (RecipeId + Name),
`"requested_top_n"`, `"returned_count"` (may be under `top_n` when the safe
set is small), `"reason"` (non-empty whenever `returned_count < top_n`, and
a clear explanation — never an exception — when the safe set yields no
candidates at all), and `"alternatives"`: up to `top_n` entries of
`{RecipeId, Name, score, Calories, ProteinContent}`.

**Standalone and opt-in.** Nothing in the scoring/ranking pipeline imports
this file; a caller invokes `suggest_alternatives()` explicitly as a
separate step — the same integration pattern as `explain_health_score()`.

**Verification.** `test_alternatives.py` runs three real blocked-recipe
cases against a peanut-allergic profile's safe set: two well-populated
peanut recipes ("Creamy Peanut Dessert", RecipeId 508281, and "Super Easy
Peanut Noodles", RecipeId 220974) that exercise the `taste_similarity` path,
and RecipeId 218 — a deliberately ingredient-poor recipe (a 2-ingredient
list) that must automatically come back `"method": "nutrition_fallback"`.
Ranking is fully vectorized (no per-element Python loop, tie-break
included), so querying the whole safe set is fast.

---

## 8. meal_planner — daily & weekly meal-plan builder (standalone, opt-in)

**What it is.** `Expert System/planner/meal_planner.py` closes the "one
recipe at a time" gap: the pipeline returns a single ranked list, but users
want a full balanced day (and week) that hits their calorie target without
repeated recipes or conflicting meals. The planner assembles breakfast +
lunch + dinner — one recipe per slot, no repeats, summed calories inside a
tolerance band of the day target — and chains days into a full week with
global no-repeat.

**Composed entirely from existing pipeline pieces.** It reuses
`UserProfile`'s calorie math, `DietaryExpertSystem.filter_recipes()`
(medical safety), and `topsis_model.rank_with_topsis()` (TOPSIS/AI/expert
ranking). No new model, no training, and no modification to the
scoring/ranking pipeline; nothing in the pipeline imports it (same opt-in
integration pattern as `explain_health_score()` and `suggest_alternatives()`).
It is not currently wired to any backend endpoint (§9) — it is available for
integration but not yet exposed over HTTP.

**Key design decisions** (each validated by the meal-planner feasibility
investigation that preceded the build):

1. **Day target = 3 × the engine's per-meal target, NOT raw
   `UserProfile.daily_calories`.** `filter_recipes()` returns
   `target_meal_calories` (TDEE/3 plus the goal's calorie offset) — already
   the achievable per-meal number. Raw TDEE is structurally unreachable for
   medically-capped profiles: the diabetic test profile (45yo/172cm/88kg
   male, light activity) has TDEE 2386 kcal, but medical per-recipe calorie
   caps mean a 3-meal day can never exceed about 1797 kcal. Building the day
   against raw TDEE would force an impossible number.
2. **Calorie-stratified candidate selection instead of plain top-K ranked.**
   Each slot's candidate pool is the top-ranked recipe per calorie band —
   3 bands spanning the pool's own calorie range, up to `top_k_per_band`
   (default 40) per band, i.e. up to 120 candidates per slot. Plain top-K
   ranked was proven to fail: under calorie-reducing goals (weight_loss
   pulls low-calorie recipes to the top of every slot) it discards every
   high-calorie candidate, so the investigation found ZERO valid
   combinations from a 3-slot top-K pool.
3. **Graceful degradation instead of errors or forced numbers.** Profiles
   that cannot reach the target even with a relaxed tolerance get the
   closest achievable day with `"target_reached": false` and an honest,
   specific `"warning"` that names the binding rule (e.g. the per-recipe
   calories cap) and the actual shortfall. Neither planner function ever
   raises on an unachievable plan.
4. **Global no-repeat — and the lunch/dinner pool finding.** lunch and
   dinner draw from the IDENTICAL underlying MealType pool ("MainDish" per
   `filtering_engine.MEAL_TYPE_DATA_MAP`), so "no repeats" cannot be
   enforced per meal type — it is enforced globally across all three slots
   of every day, and across the entire week in weekly plans. A slot whose
   pool is exhausted by exclusion falls back to its full pool and is
   reported in `"reused_slots"`/`"note"` — never silently.
5. **Tolerance, relaxation, and in-band selection.** The default day
   tolerance is ±15% (±10% was found to support too few valid combinations
   to fill a 7-day week; ±15% supports 7 no-repeat days for moderately
   constrained profiles). If no combination fits, the planner first retries
   at tolerance + 0.10 (up to the hard cap of 0.40) and only then degrades
   to the closest achievable day. Inside a valid band, combinations are
   preferred 70% by mean `final_score` (ranking quality) and 30% by
   closeness of the summed calories to the target — pure best-rank selection
   clusters at the low end of the band, while pure calorie fit would ignore
   ranking quality.
6. **The 4th "snack" slot — investigated and rejected on evidence.** For
   the constrained test profile (diabetes + high_cholesterol + muscle_gain),
   per-recipe caps limit a 3-meal day to about 1500 kcal against a roughly
   2900 kcal target. A tested 4th "snack" slot closes at most about 480 kcal
   of that gap — the day would still land roughly 900 kcal short. Because
   the gap is structural (medical per-recipe caps, not a selection
   shortcoming), the planner does not pretend a 4th slot solves it; it
   returns the closest achievable 3-meal day with an honest warning.

**API shape.**

- `build_daily_plan(profile, system=None, tolerance=0.15,
  top_k_per_band=40, exclude_recipe_ids=None, user_taste_text=None) -> dict`
  — the profile's `meal_type` attribute is IGNORED (a copy is made and set
  per slot internally); the planner always builds breakfast + lunch +
  dinner. Returns a JSON-serializable dict: `day_target_calories`,
  `actual_calories`, `target_reached`, `tolerance_requested`,
  `tolerance_used` (None when only the closest achievable day exists),
  `warning`, `note`, `meals` (one entry per slot with RecipeId, Name, the
  full nutrition columns, and `final_score`), `totals`
  (Calories/Protein/Sugar/Fiber), `candidate_pools`, `reused_slots`.
- `build_weekly_plan(profile, days=7, tolerance=0.15, system=None,
  top_k_per_band=40, user_taste_text=None) -> dict` — same parameters; the
  three slot pipelines run ONCE and the pools are reused across all days.
  Each day carries a `day_index`; the `summary` block reports
  `days_requested`, `days_returned`, `distinct_recipes_used`,
  `forced_repeat_slots`, `days_target_reached`,
  `avg_target_achievement_pct`, and `avg_actual_calories`.

Constructing the `DietaryExpertSystem` once (loading the 384k-row dataset
takes several seconds) and passing it via `system` avoids reloading the
dataset on every call; with `system=None` the module reloads it each time.

**Verification.** `test_meal_planner.py` exercises both sides of the design:
the reachable-target profile (45yo diabetic male, weight_loss — TDEE 2386
but capped at about 1797 kcal/day) whose daily plan returns
`"target_reached": true`, and the constrained profile (30yo male,
diabetes + high_cholesterol, muscle_gain — about 1500 kcal achievable vs
roughly 2900 kcal target) whose daily plan is expected to return
`"target_reached": false` with the honest warning. For the reachable
profile's 7-day weekly plan, the module's own documented reference summary
reports 21 distinct recipes used (7 days × 3 slots), 0 forced-repeat slots,
7/7 days reaching the target, about 96.2 % average target achievement, and
about 1840 kcal average daily calories. The combination search itself is
fully vectorized (per-dinner-row 2-D block math over the paired
breakfast/lunch matrices), so even the exhaustive closest-mode search over
120³ candidates runs in well under a second (as documented in the code).

---

## 9. Backend REST API (`Backend/`)

**What it is.** A Django 6 + Django REST Framework project that exposes the
pipeline in §1–§8 over HTTP for real clients (primarily the mobile app).
It does not reimplement any AI/ML logic — every recommendation, search,
score, and explanation is computed by calling straight into `Expert System/`,
`TOPSIS/`, and `Nlp/` functions. There is no separate "API interface" module
inside the AI projects: the backend imports and calls
`DietaryExpertSystem.filter_recipes()`, `topsis_model.rank_with_topsis()`,
`Nlp.pipeline.search_recipes()`, and
`ml.health_classifier.explain.explain_health_score()` directly.

### 9.1 How the two codebases connect

`Backend/config/ai_bridge.py` is imported once, at the top of `settings.py`,
and extends `sys.path` with three folders relative to the repository root:
the repo root itself (for `import Nlp`), `Expert System/` (for `core`,
`rules`, `engine`, `ml`), and `TOPSIS/` (for `topsis_model`). This lets
ordinary Django view code write `from engine.filtering_engine import ...`
or `from topsis_model import rank_with_topsis` as if those packages were
part of the Django project, without moving or duplicating any AI project
file.

`Backend/recipes/services/` holds four small modules that bridge Django and
the AI layer:

- **`ai_runtime.py`** — process-wide, thread-safe lazy singletons for the
  expensive AI resources: `get_expert_system()` builds a `DietaryExpertSystem`
  from the 384,541-row CSV on its first call (several seconds) and returns
  the cached instance afterward; `warm_up_explainer()` and
  `warm_up_nlp_search()` do the same for the SHAP explainer (~8 s numba JIT)
  and the Sentence-BERT search model. Three opt-in environment flags
  (`AI_EAGER_WARMUP`, `AI_EAGER_EXPLAIN_WARMUP`, `AI_EAGER_NLP_WARMUP`) move
  each warmup to server startup, in a background thread, instead of paying
  the cost on the first real request.
- **`filter_cache.py`** — a thread-safe, bounded (32-entry) LRU cache in
  front of `filter_recipes()`. That call runs two row-wise pandas
  computations over the full corpus inside `Expert System/engine/scorer.py`
  and costs roughly ten to fifty seconds depending on profile complexity;
  its result depends only on the user's profile fields plus the per-request
  `meal_type`, never on free-text search terms, so it is safe to cache and
  reuse across repeat requests with the same signature. A duck-typed proxy
  (`_CachingExpertSystem`) exposes the identical `.filter_recipes(profile)`
  interface so it can be handed to `Nlp.pipeline.search_recipes()` as its
  `expert_system` argument transparently — search benefits from the same
  cache with no changes to `Nlp/`. `get_cached_expert_system()` is the
  process-wide entry point every view uses in place of calling
  `ai_runtime.get_expert_system()` directly.
- **`profile_translator.py`** — converts a stored Django `UserProfile` into
  the AI project's `core.user_profile.UserProfile` dataclass. The two sides
  use different vocabularies for goals (Django's four choices map onto the
  AI engine's eight, with two eager medical overrides — pregnancy always
  becomes `pregnancy_diet`, and age 65+ with a general weight goal becomes
  `elderly_diet`) and for conditions/allergies/preferences (free-form JSON
  lists on the Django side). The AI engine silently drops any condition or
  preference string it does not recognize, so this layer normalizes common
  near-miss aliases (e.g. `"diabetic"` → `diabetes`, `"keto"` → `low_carb`,
  `"pescatarian"` → `seafood_lover`) and then raises a `ProfileTranslationError`
  naming the offending field and value for anything still unrecognized,
  rather than letting it fail silently inside the engine. `meal_type` (which
  meal, right now) is deliberately left untouched by this layer — it is a
  per-request query parameter, not a stored profile attribute.
- **`serialization.py`** — a single `json_safe(value)` helper that converts
  numpy/pandas scalar types to native Python types and any NaN/`pd.NA` to
  `None`, so every endpoint's JSON response is always valid regardless of
  which columns happen to be missing for a given recipe.

### 9.2 Django apps

Three apps are installed beyond Django's own defaults and DRF/CORS:
`accounts`, `profiles`, `recipes`. Authentication is DRF's built-in
`TokenAuthentication` (`IsAuthenticated` is the default permission class
project-wide, applied explicitly on every view below). Recipe data itself
has no Django ORM model at all — the entire recipe corpus lives in the
in-memory DataFrame built by the AI layer, not the SQL database; only user
accounts and health profiles are stored in Django's SQLite database.

**`accounts`** — registration and token issuance:

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/api/accounts/register/` | POST | public | Creates a `User`, issues a token immediately so the client doesn't need a separate login call right after signup. |
| `/api/accounts/login/` | POST | public | Accepts a username or an email address plus password; resolves email to the matching username before calling Django's `authenticate()`. |
| `/api/accounts/logout/` | POST | token required | Deletes the caller's token row — that key stops working immediately, with no separate blacklist needed. |

**`profiles`** — one health/dietary profile per user, all on a single
endpoint (`/api/profiles/me/`, always the caller's own profile — there is no
`<id>` in the URL, and so no way to look up another user's profile by
guessing an id):

| Method | Behavior |
|---|---|
| GET | Retrieve the caller's profile (404 if not created yet). |
| POST | Create it — 400 if one already exists (use PUT/PATCH to update). |
| PUT | Replace it entirely. |
| PATCH | Partially update it. |

Stored fields: `age` (1–120), `height` (cm, 50–250), `weight` (kg, 20–400),
`gender` (male/female), `pregnant` (bool — only valid when `gender=female`),
`conditions`/`allergies`/`preferences` (each a free-form JSON list of
strings), `taste_text`, `goal` (lose_weight / maintain_weight / gain_weight
/ gain_muscle), `activity_level` (sedentary / light / moderate / active /
very_active), `meal_type` (standard / vegetarian / vegan / keto / halal /
other — a stored diet-preference choice, distinct from the per-request
breakfast/lunch/dinner concept used by the recipe endpoints below).

**`recipes`** — the AI-backed endpoints. All five require
`IsAuthenticated` and a completed profile (404 if the caller has none);
all translate the stored profile via `profile_translator` first (400 with
the translator's exact message on any unrecognized stored value); all
convert unexpected AI-layer exceptions into a logged, generic 500 rather
than leaking a stack trace to the client.

- **`GET /api/recipes/recommendations/`** — the base recommendation feed.
  Query params: `meal_type` (breakfast/lunch/dinner/any, default `any`),
  `taste_text` (optional free text for Model 2), `limit` (default 20,
  capped at 100). Runs the profile through `filter_recipes()` (via the
  cache) then `rank_with_topsis()`. Response: `count`, `total_safe`
  (recipes surviving Expert System filtering), `total_original` (full
  corpus size), `meal_type` (the value actually applied), and `results` —
  each row carrying `RecipeId`, `Name`, `ImageUrl`, `Calories`,
  `ProteinContent`, `CarbohydrateContent`, `SugarContent`, `FiberContent`,
  `final_score`, and the four component scores (`_topsis_score`,
  `_ai_health_score`, `_expert_score`, and `_taste_score` when
  `taste_text` was given).

- **`GET /api/recipes/search/`** — free-text search. Query params: `query`
  (required), `meal_type` (optional — wins over whatever the query text
  itself implies, except in the rare case the parsed text also implies a
  conflicting meal type, in which case the parsed value wins and the
  discrepancy is visible by comparing the response's top-level `meal_type`
  against `filters_applied.meal_type`), `taste_text`, `limit`. Calls
  `Nlp.pipeline.search_recipes()` with a deliberately large internal
  pre-rank limit so the full NLP-filtered safe set — not just its own
  internal top slice — is what TOPSIS re-ranks; truncating before TOPSIS
  would silently drop recipes the final ranking should have surfaced.
  Response shape matches `recommendations/` plus `total_safe` narrowed by
  the query's own numeric/ingredient filters, and a `filters_applied` object
  exposing the raw parsed filters (condition, meal_type, protein_min,
  sugar_max, sodium_max, fiber_min, calories_max, allergy, diet_preference,
  main_ingredient) for transparency.

- **`GET /api/recipes/<id>/alternatives/`** — up to three alternative
  recipes for a given `RecipeId`. Takes the caller's safe set (via
  `filter_recipes()`), excludes the given id, and ranks the rest with
  `rank_with_topsis()` on the caller's goal (no taste text). Response:
  `original_recipe_id`, `alternatives_count`, `results` (same per-row shape
  as the other list endpoints, without the component-score breakdown).

- **`GET /api/recipes/<id>/explanation/`** — plain-language reasons for a
  recipe's medical suitability. Looks the recipe up directly in the cached
  expert system's DataFrame (404 if not found), and if the caller's profile
  has no stored medical conditions, returns a simple "no chronic conditions,
  this recipe is fine" message without invoking the classifier. Otherwise
  calls `explain_health_score()` for the caller's condition keys and
  flattens its per-condition SHAP reasons into a single `reasons` list.
  Response: `recipe_id`, `is_safe` (always `true` — the endpoint currently
  only explains recipes the caller can already see, never evaluates recipes
  the Expert System would reject), `reasons`.

- **`GET /api/recipes/<id>/`** — full recipe detail. Looks the recipe up by
  `RecipeId` directly in the cached expert system's DataFrame (404 if the
  id isn't present in the live corpus, whether it never existed there or
  was dropped during data cleaning), and separately runs the caller's
  profile through `filter_recipes()` to report whether this specific recipe
  is currently safe for this specific user. Response: `recipe_id`,
  `is_safe`, and `recipe` — every column the live corpus carries for that
  row (all 41 columns of `cleaned_recipes.csv`, native JSON types, missing
  values as `null`), not a curated subset.

### 9.3 Recipe images

`cleaned_recipes.csv` already carries three image-related columns for every
recipe, populated during data cleaning from the raw dataset's R-style
`c("url1", "url2", ...)` image-list format: `ImageUrl` (a single
representative URL, or absent when a recipe has no photos), `ImagesCount`
(how many photos exist), and `ImagesJson` (the full list, as a JSON array
string). `ImageUrl` is included in every list endpoint's result rows
(`recommendations/`, `search/`, `alternatives/`) for card thumbnails, and
the full trio is naturally present in the detail endpoint's response since
it returns every column. No second dataset or additional load is needed for
this — the columns were already part of the DataFrame the expert system
loads at startup. A separate, larger `cleaned_recipes_full.csv` also exists
on disk (435,009 rows, including some rows dropped from the live corpus
during cleaning, plus a handful of columns not carried into the live file);
the live pipeline does not load it, and the recipe-detail endpoint does not
need to, since every field it currently exposes is already in
`cleaned_recipes.csv`.

### 9.4 Testing

The `recipes` app has an automated Django test suite (`python manage.py test
recipes`) covering every endpoint's error paths (401/403, 404, 400, 500),
happy-path response shape, the meal_type precedence rules, the NLP
anti-truncation-bias fix (asserting the full filtered set — not just the
requested `limit` — reaches TOPSIS), the filter-recipes cache's hit/miss/
eviction/proxy behavior in isolation, the preference-alias translation
table, and per-user safety checks on the detail endpoint verified against
real corpus data (e.g. confirming a peanut-allergy profile marks a recipe
actually flagged `HasNuts=True` as unsafe). Because most of these tests
exercise the real AI pipeline end-to-end rather than mocking it, the full
suite takes several minutes to run — most of that time is the ~10–50 second
`filter_recipes()` cost, paid once per distinct test profile. The
`accounts` and `profiles` apps currently have no test coverage beyond
Django's default project scaffolding.

---

## 10. Dashboard (`dashboard/`)

**What it is.** A standalone, read-only React admin/viewer dashboard (Vite +
Recharts) living outside `Backend/`, `Expert System/`, `TOPSIS/`, and `Nlp/`.
It is a separate consumer of a Django REST API, in the same spirit as the
mobile app, but scoped only to an administrator/viewer role: aggregate
statistics, a results table, and a chart — no personal recommendations, no
profile fields, no login flow shared with the end-user mobile app, and no
recipe or user management.

**Current status: front end built, backend not yet implemented.** The
dedicated Django "Dashboard API" this project is designed against does not
exist yet among the endpoints in §9. Until it does, the dashboard is
intentionally built to run and be previewed against no backend at all: stat
cards show a neutral placeholder, the results table reads "No results
available yet," the chart reads "Chart data will appear when Dashboard API
data is available," and a single banner states the API is pending. No
placeholder or sample numbers are hard-coded anywhere in the project.

**Provisional contract.** The dashboard's own documentation records a
proposed (not yet implemented, not yet agreed) response shape it expects to
consume once a Dashboard API exists — aggregate stats (`total_recipes`,
`supported_conditions`, `supported_allergies`, `recommendations_count`), a
`results` list of scored recipe rows, and a `chart_data` list of
label/value pairs. Every field name is expected to be finalized when the
Dashboard API is actually implemented; the frontend centralizes all of the
API base URL and endpoint path configuration in one file
(`src/config/api.js`, base URL from a `.env` value) and one data-fetching
function (`src/services/api.js`), so updating the real contract later is a
small, localized change.

---

## 11. Repository structure

The repository root holds `Expert System/` (rule engine + both trained
models + data + the `planner/` meal-plan layer), `TOPSIS/` (goal scoring +
the `sanity_check.py` regression suite), `Nlp/` (smart search, §5),
`Backend/` (the Django REST API, §9), and `dashboard/` (the standalone
admin frontend, §10):

```
Smart-Dietary-Advisor/                    # repo root
├── Expert System/                        # rule engine + both trained models
│   ├── data/                             # model artifacts + recipe corpus
│   │   ├── cleaned_recipes.csv           # 384,541-row corpus the live pipeline loads
│   │   ├── cleaned_recipes_full.csv      # 435,009-row superset, not loaded by the live pipeline
│   │   ├── health_classifier.pt          # Model 1 weights
│   │   ├── health_scaler.pkl             # Model 1 input standardizer
│   │   ├── health_classifier_labels.json # 22 condition keys (output order)
│   │   ├── health_classifier_thresholds.json  # per-condition tuned thresholds
│   │   ├── word2vec_ingredients.model    # Model 2 trained model (~5.3 MB)
│   │   └── recipe_taste_embeddings.pkl   # 383,293 recipe vectors (~230 MB, gitignored)
│   ├── ml/
│   │   ├── setup_artifacts.md            # artifact inventory + regeneration guide (both models)
│   │   ├── data/labeled_recipes.csv      # Model 1 training labels (~55 MB, gitignored)
│   │   ├── health_classifier/            # Model 1 — multi-label neural classifier
│   │   │   ├── build_labels.py           #   soft-label generation (rule × rating)
│   │   │   ├── health_classifier.py      #   model definition + training + 70/15/15 split
│   │   │   ├── inference.py              #   ai_health_score (production singleton)
│   │   │   ├── explain.py                #   explain_health_score — SHAP per-condition reasons
│   │   │   ├── tune_thresholds.py        #   per-condition threshold calibration (val only)
│   │   │   ├── evaluate_classifier.py    #   test-set reports (flat 0.5 / --tuned)
│   │   │   └── results/                  #   metrics.md / metrics_tuned.md (+ json)
│   │   └── word2vec/                     # Model 2 — ingredient embeddings + taste
│   │       ├── build_taste_embeddings.py #   corpus cleaning + Word2Vec training
│   │       ├── taste_concepts.py         #   concept map (43 concepts, vocab-verified)
│   │       ├── taste_inference.py        #   user text → vector → recipe taste scores
│   │       └── alternatives.py           #   suggest_alternatives() — taste-similar safe alternatives (opt-in)
│   ├── planner/                          # meal-plan builder — standalone, opt-in (§8)
│   │   └── meal_planner.py               #   build_daily_plan / build_weekly_plan
│   ├── core/                             # UserProfile dataclass, shared constants
│   ├── rules/                            # medical rules, allergy/halal rules, goal & preference vectors
│   ├── engine/                           # filtering_engine.py, scorer.py, rule_builder.py
│   └── ui/                               # command-line demo interface
├── TOPSIS/                               # goal scoring + sanity_check.py suite
├── Nlp/                                  # smart search — UX feature, NOT a trained model (§5)
│   ├── pipeline.py                       #   search_recipes(): query → filters → engine
│   ├── query_parser.py                   #   free-text → structured filters (embeddings)
│   ├── vocab_terms.py                    #   term lists (synced with core/constants.py)
│   ├── check_nlp.py                      #   internal consistency diagnostic
│   └── test_*.py                         #   five test files (see §13)
├── Backend/                               # Django REST API (§9)
│   ├── config/                           # settings, urls, ai_bridge.py sys.path shim
│   ├── accounts/                         # register / login / logout
│   ├── profiles/                         # stored health/dietary profile (one per user)
│   └── recipes/                          # AI-backed endpoints
│       └── services/                     # ai_runtime, filter_cache, profile_translator, serialization
└── dashboard/                            # standalone React admin dashboard (§10)
    └── src/                              # components, hooks, config, services
```

---

## 12. Known current limitations

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
- **The classifier's per-feature attributions can mislead off the production
  distribution.** The diabetes head shows counter-intuitive SHAP signs for
  ProteinContent (strong negative) and Calories (positive) when scored on
  arbitrary recipes. Investigation traced these to (i) a learned
  protein↔calorie-density collinearity (+0.58 corpus correlation; high-protein
  recipes violate the Calories≤600 gate 5.5× more often), not to anything in the
  training labels (label–protein correlation ≈ 0), and (ii) non-monotone tail
  artifacts of the model's extreme decision margins. Within the 24,679-recipe
  safe-for-diabetes population that health_classifier actually ranks, the same
  audit shows clinically sensible ordering (standardized coefficients: Calories
  −0.92, Carbohydrates −0.57, Sugar −0.32; protein slightly favorable at +0.19
  once confounders are controlled), so real rankings are unaffected and the hard
  Expert System rules remain the actual safety gate. The SHAP explanation layer
  therefore uses deliberately neutral wording for ProteinContent and Calories
  instead of asserting medical harm.
- **Ranking is corpus-bound.** Recommendations are limited to the 384,541
  recipes in the cleaned Food.com corpus; the vocabulary (4,504 tokens) is
  fixed at training time.
- **Smart search needs a pretrained model download on first use.** `Nlp/`
  loads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, which
  requires internet access the first time it runs (the model is then cached
  locally). The embedding model is generic and zero fine-tuned, so query
  understanding is limited to the fixed term lists in `vocab_terms.py` — it is
  a UX convenience, not a learned understanding layer.
- **The eggs allergy category has no protective dataset column.** There is no
  `HasEggs` column in `cleaned_recipes.csv`, so egg-allergy protection relies
  entirely on text matching — inherently more fragile than the column-backed
  categories (`HasNuts`, `HasLactose`, ...), even after the
  groundnut/eggnog/eggshell fix. A future data pass should add and populate
  the column, as `halal_and_allergies.py` already assumes it exists.
- **One known test/engine semantic gap remains (by design).**
  `test_medical_safety.py` Scenario 2 asserts a strict substring rule
  (ingredient or name containing "peanut") that is deliberately stricter than
  the engine's word-boundary regex. Exactly one recipe — "Peanutty Oatmeal
  Cookies", whose ingredients are genuinely clean (verified) — trips this
  assertion. It is a NAME-ONLY case, not a real leak, and is documented as
  known behavior rather than "fixed" by over-broad pattern changes.
- **`suggest_alternatives()` quality depends on the blocked recipe's
  ingredient detail**, and it is not yet wired into the live
  `alternatives/` endpoint (§9, §7) — that endpoint currently uses a
  simpler TOPSIS-based re-rank of the safe set instead.
- **`meal_planner` cannot reach the calculated daily target for
  medically-constrained profiles**, and it is not yet exposed over any
  backend endpoint (§8, §9) — where per-recipe calorie caps (e.g. diabetes)
  create a hard ceiling below the day target, no 3-meal combination can hit
  it, and the planner returns the closest achievable day with
  `"target_reached": false` and a warning naming the binding rule and the
  shortfall. This is known, expected, and honestly reported — the
  alternative would be ignoring the medical caps — not a bug.
- **The backend's filter-recipes cache mitigates repeat cost, not first-call
  cost.** Every new, distinct profile signature (age/height/weight/gender/
  pregnant/conditions/allergies/preferences/goal/activity_level/meal_type)
  still pays the full ~10–50 second row-wise scoring cost in
  `Expert System/engine/scorer.py` on its first request; only repeat requests
  with the same signature are fast. The cache is bounded to 32 entries to
  keep memory use predictable, so a high-traffic deployment with many
  distinct profiles would still see frequent cold-cache latency.
- **The Django project is configured for local development, not
  production.** `DEBUG=True`, `ALLOWED_HOSTS=['*']`, `CORS_ALLOW_ALL_ORIGINS
  =True`, a hardcoded `SECRET_KEY`, and SQLite as the database are all
  appropriate for local/mobile-testing use but would need to be tightened
  and replaced before any real deployment.
- **Only the `recipes` app has automated test coverage.** `accounts` and
  `profiles` currently rely on manual verification; their test files are
  still the default empty Django scaffolding.
- **The recipe-explanation endpoint's `is_safe` field is not a real safety
  check.** `GET /api/recipes/<id>/explanation/` always returns `is_safe:
  true` — it explains suitability for recipes the caller can already see,
  it does not independently verify the recipe against the Expert System's
  rules for that user. The recipe-detail endpoint's `is_safe` field (§9),
  by contrast, is a genuine per-user safety check against `filter_recipes()`.
- **The admin dashboard has no backend yet.** `dashboard/` is a complete,
  previewable frontend with no real data to show until a dedicated Django
  Dashboard API (distinct from the `recipes` endpoints in §9, which are
  scoped to a single authenticated end user) is implemented (§10).

---

## 13. How to verify this yourself

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

**Smart-search (`Nlp/`) tests** — run from the repository root (the first four
are fast and need no data load; the last two load `cleaned_recipes.csv`,
~520 MB, and the Sentence-BERT model on first run):

```powershell
python Nlp/test_query_parser.py      # parser unit checks — PASS
python Nlp/check_nlp.py              # internal consistency diagnostic — clean
python Nlp/test_robustness.py        # 20/20 robustness checks
python Nlp/test_robustness_all.py    # 65/65 robustness checks
python Nlp/test_pipeline.py          # end-to-end demo (3 free-text queries) — PASS
python Nlp/test_medical_safety.py    # 4 medical-safety scenarios — 3/4 PASS
```

Note on `test_medical_safety.py`: Scenario 2 ("peanut-allergic user asks for a
peanut dish") still reports 1 NAME-ONLY recipe — "Peanutty Oatmeal Cookies",
ingredients verified clean (see §12). All other scenarios pass, and the
medical-safety guarantees hold through the real `DietaryExpertSystem`.

**Standalone module smoke tests** — run from the repository root (both load
`cleaned_recipes.csv`, ~520 MB, on startup; the corpus load takes several
seconds and only once per run):

```powershell
python "Expert System/ml/word2vec/test_alternatives.py"   # 3 blocked peanut recipes — 2 taste-similar + 1 nutrition fallback
python "Expert System/planner/test_meal_planner.py"       # daily plans (reachable + degraded) + 7-day weekly plan
```

`suggest_alternatives()` output is self-describing: each case's `"method"`
field states whether taste similarity or the nutrition fallback fired, and
the 2-ingredient blocked recipe (RecipeId 218) is expected to come back
`"method": "nutrition_fallback"` (see §7). For the planner, the constrained
profile's daily plan is expected to show `"target_reached": false` with an
honest `"warning"` — the documented degradation behavior, not a bug (see
§8).

**Backend API tests** — run from the `Backend/` folder (requires the same
data artifacts as above; most tests exercise the real pipeline end-to-end
rather than mocking it, so the full run takes several minutes):

```powershell
cd Backend
python manage.py test recipes
```

This covers every `recipes` endpoint's error paths, happy-path response
shapes, the meal_type precedence rules, the search endpoint's
anti-truncation-bias fix, the filter-recipes cache's behavior in isolation,
the preference-alias translation table, and per-user safety checks verified
against real corpus data (§9.4). `accounts` and `profiles` have no test
suite to run beyond Django's default scaffolding.

**Artifact regeneration** — see `Expert System/ml/setup_artifacts.md` for the
full inventory (what is committed vs regenerated locally) and the exact
commands, in order: `cleaner.py` (cleaned CSVs), then
`ml/health_classifier/build_labels.py` (training labels), then
`ml/word2vec/build_taste_embeddings.py` (word2vec + recipe embeddings, ~61 s),
and `ml/health_classifier/health_classifier.py` + `tune_thresholds.py` +
`evaluate_classifier.py` for Model 1 (detailed steps in
`ml/health_classifier/MODEL_DOCUMENTATION.md`).
