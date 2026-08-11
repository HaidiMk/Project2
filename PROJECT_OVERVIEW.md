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

## 5. Smart Search (Nlp/) — a UX feature, not a trained model

**What it is.** `Nlp/` is a search-bar convenience layer added at the repo
root, next to `Expert System/` and `TOPSIS/`. It wraps a **pretrained
Sentence-BERT model** (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim
embeddings) to translate a free-text query such as *"low sugar dinner for
diabetic"* into structured filters, then hands those filters to the
**existing, unmodified** `DietaryExpertSystem.filter_recipes()`.

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
terms. `Nlp/pipeline.py` exposes `search_recipes(query, profile, system,
top_n)`, which merges the parsed filters into the caller's base `UserProfile`
and calls the engine. It never modifies the engine's inputs, rules, or output
ordering.

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
`test_robustness.py`, `test_robustness_all.py`). Test commands are in §11.

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
this file; a backend calls `suggest_alternatives()` explicitly as a
separate endpoint — the same integration pattern as
`explain_health_score()`.

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
integration pattern as `explain_health_score()` and
`suggest_alternatives()`).

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

For the backend: construct the `DietaryExpertSystem` once (loading the
384k-row dataset takes several seconds) and pass it via `system`; with
`system=None` the module reloads the dataset on every call.

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

## 9. Repository structure (models)

The repository root holds three top-level folders: `Expert System/` (rule
engine + both trained models + data + the `planner/` meal-plan layer),
`TOPSIS/` (goal scoring + the
`sanity_check.py` regression suite), and `Nlp/` (smart search, §5). Inside
`Expert System/`, `ml/` was split into two clearly-named model folders (the
model code was previously mixed together in one flat folder), and
`planner/` (§8) was added later as a standalone composition layer that
builds on the unchanged pipeline rather than modifying it:

```
Smart-Dietary-Advisor/                    # repo root
├── Expert System/                        # rule engine + both trained models
│   ├── data/                             # model artifacts (committed, except large ones)
│   │   ├── health_classifier.pt          # Model 1 weights
│   │   ├── health_scaler.pkl             # Model 1 input standardizer
│   │   ├── health_classifier_labels.json # 22 condition keys (output order)
│   │   ├── health_classifier_thresholds.json  # per-condition tuned thresholds
│   │   ├── word2vec_ingredients.model    # Model 2 trained model (~5.3 MB)
│   │   └── recipe_taste_embeddings.pkl   # 383,293 recipe vectors (~230 MB, gitignored)
│   ├── ml/
│   │   ├── setup_artifacts.md            # artifact inventory + regeneration guide (both models)
│   │   ├── data/labeled_recipes.csv      # Model 1 training labels (~55 MB, gitignored)
│   │   ├── health_classifier/            # ⭐ Model 1 — multi-label neural classifier
│   │   │   ├── build_labels.py           #   soft-label generation (rule × rating)
│   │   │   ├── health_classifier.py      #   model definition + training + 70/15/15 split
│   │   │   ├── inference.py              #   ai_health_score (production singleton)
│   │   │   ├── tune_thresholds.py        #   per-condition threshold calibration (val only)
│   │   │   ├── evaluate_classifier.py    #   test-set reports (flat 0.5 / --tuned)
│   │   │   ├── results/                  #   metrics.md / metrics_tuned.md (+ json)
│   │   │   └── MODEL_DOCUMENTATION.md    #   detailed Arabic model documentation
│   │   └── word2vec/                     # ⭐ Model 2 — ingredient embeddings + taste
│   │       ├── build_taste_embeddings.py #   corpus cleaning + Word2Vec training
│   │       ├── taste_concepts.py         #   concept map (43 concepts, vocab-verified)
│   │       ├── taste_inference.py        #   user text → vector → recipe taste scores
│   │       └── alternatives.py           #   suggest_alternatives() — taste-similar safe alternatives (opt-in)
│   ├── planner/                          # ⭐ meal-plan builder — standalone, opt-in (§8)
│   │   ├── meal_planner.py               #   build_daily_plan / build_weekly_plan
│   │   └── __init__.py                   #   package marker
│   └── engine/, rules/, ui/, core/       # expert system (unchanged)
├── TOPSIS/                               # goal scoring + sanity_check.py suite
└── Nlp/                                  # ⭐ smart search — UX feature, NOT a trained model (§5)
    ├── pipeline.py                       #   search_recipes(): query → filters → engine
    ├── query_parser.py                   #   free-text → structured filters (embeddings)
    ├── vocab_terms.py                    #   term lists (synced with core/constants.py)
    ├── check_nlp.py                      #   internal consistency diagnostic
    └── test_*.py                         #   five test files (see §11)
```

---

## 10. Known current limitations

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
  (`Expert System/ml/health_classifier/explain.py`) therefore uses deliberately
  neutral wording for ProteinContent and Calories instead of asserting medical
  harm.
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
  ingredient detail.** The taste-similarity path needs a recipe embedding
  and at least `min_vocab_tokens` (default 4) in-vocab ingredient tokens;
  sparse or out-of-vocabulary recipes automatically fall back to
  nutrition-only similarity — the same 9 columns health_classifier uses,
  useful but lower-fidelity (it matches "similar nutrition", not "similar
  taste"), and the `"method"` field reports which path fired. The module
  can also only ever return recipes from the caller's safe set: a small
  safe set yields few options, and an empty one yields no alternatives
  (with a clear reason, never an error).
- **`meal_planner` cannot reach the calculated daily target for
  medically-constrained profiles.** Where per-recipe calorie caps (e.g.
  diabetes) create a hard ceiling below the day target, no 3-meal
  combination can hit it — the planner returns the closest achievable day
  with `"target_reached": false` and a warning that names the binding rule
  and the shortfall (see §8, design decision 3). This is a known, expected,
  and honestly-reported behavior — the alternative would be ignoring the
  medical caps — not a bug.

---

## 11. How to verify this yourself

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
ingredients verified clean (see §10). All other scenarios pass, and the
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

**Artifact regeneration** — see `Expert System/ml/setup_artifacts.md` for the
full inventory (what is committed vs regenerated locally) and the exact
commands, in order: `cleaner.py` (cleaned CSVs), then
`ml/health_classifier/build_labels.py` (training labels), then
`ml/word2vec/build_taste_embeddings.py` (word2vec + recipe embeddings, ~61 s),
and `ml/health_classifier/health_classifier.py` + `tune_thresholds.py` +
`evaluate_classifier.py` for Model 1 (detailed steps in
`ml/health_classifier/MODEL_DOCUMENTATION.md`).
