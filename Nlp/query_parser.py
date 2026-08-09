"""
query_parser.py — free-text English search query -> structured filter dict.

Method (no NER, no fine-tuning):
    1. Explicit numeric constraints -> regex (extract_numeric_constraints),
       completely independent of the semantic step.
    2. Vague descriptive terms ("low sugar", "high protein", ...) -> fixed
       reference values in VAGUE_TERMS. Conflict rule: an explicit number
       always overrides the vague-term default for the same field.
    3. Entities (condition, meal_type, diet_preference, main_ingredient)
       -> Sentence-BERT semantic similarity against a fixed vocabulary
       built from the local vocabulary lists in vocab_terms.py (mirrored
       from Expert System/core/constants.py — this module has zero import
       dependency on the rest of the project).
        "sentence-transformers/all-MiniLM-L6-v2" is used pretrained only.
        Every category claims a phrase only through its own measured gate
        (category-specific thresholds, minimum margins, generic-word /
        misleading-phrase exclusions — see the constants below), so a bare
        generic word can never turn into an inferred condition, meal,
        preference or ingredient.
    4. Allergies -> explicit negation ("no dairy", "nut-free") or marker
       ("allergic to nuts") phrases only, so positive mentions such as
       "seafood pasta" never produce a false allergy.

Medical safety (NFR-010):
    This module ONLY returns filters. It never decides what is medically
    safe, never filters and never blocks anything — that responsibility
    belongs entirely to the Expert System downstream.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .vocab_terms import (  # when imported as a package ("import Nlp")
        ALLERGY_EN,
        DISEASE_EN,
        MEAL_TYPE_EN,
        PREFERENCE_EN,
    )
except ImportError:  # when run standalone with Nlp/ directly on sys.path
    from vocab_terms import ALLERGY_EN, DISEASE_EN, MEAL_TYPE_EN, PREFERENCE_EN

# --------------------------------------------------------------------------- #
# Model & matching parameters
# --------------------------------------------------------------------------- #

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine-similarity thresholds on normalized all-MiniLM-L6-v2 embeddings.
# Paraphrase-level matches typically land in the 0.55-0.85 band; unrelated
# pairs sit well below 0.45. Preferences use a stricter bar because short
# nutrient words ("carb") hover near 0.5 against "low carb".
#
# main_ingredient uses an even stricter bar (measured, not guessed):
# genuine stated-ingredient matches score >= 0.72 (ingredient inside a
# multi-word phrase, e.g. "chicken lunch" -> 0.72) and >= 0.86 (the exact
# ingredient word, e.g. "chicken" -> 1.0). Generic single words that merely
# sound food-adjacent ("salad" -> "cheese" 0.56, "soup" -> "pasta" 0.56,
# "breakfast" -> "eggs" 0.58, "pizza" -> "cheese" 0.62) top out well below
# 0.65, so that bar cleanly separates the two populations.
THRESHOLD_DEFAULT = 0.50

# Raised from 0.62 after measurement: genuine preference phrases score
# >= 0.72 ("no meat" -> vegetarian 0.72, "loves chicken" -> chicken_lover
# 0.75), while misleading generic words ("diet" -> mediterranean 0.63,
# "vegetable" -> vegetarian 0.65) sit below 0.70.
THRESHOLD_DIET_PREFERENCE = 0.70
THRESHOLD_MAIN_INGREDIENT = 0.65

# Minimum gap between the best and second-best main_ingredient match of a
# query phrase. Real ingredient claims always stand out by a wide margin
# (>= 0.13 even for "rice" vs "brown rice"); weak/ambiguous guesses where
# nothing stands out (e.g. "meat" -> beef 0.82 vs pork, margin 0.09;
# "cabbage" -> carrot 0.67 vs broccoli, margin 0.05) are rejected.
MAIN_INGREDIENT_MIN_MARGIN = 0.10

# Same ambiguity guard for meal_type, with a lower bar: measured misleading
# words hover at margin ~0.00 ("snack" -> dinner 0.56 vs lunch 0.56,
# "quick meal" -> dinner 0.61 vs lunch 0.61), while every genuine meal
# phrase keeps a margin of at least 0.07 ("afternoon" -> lunch 0.55 vs
# breakfast 0.49).
MEAL_TYPE_MIN_MARGIN = 0.05

# Tokens that name a dish type, a meal type or a broad food category
# ("salad", "soup", "dinner", "meat", ...) rather than a specific
# ingredient. A query phrase made ONLY of such tokens can never claim a
# main_ingredient: a generic word must not turn into an inferred
# ingredient. Phrases that ALSO contain a real ingredient word
# ("chicken salad", "lentil soup", "quinoa salad") pass untouched, because
# the ingredient token itself is not in this set.
_GENERIC_FOOD_WORDS = frozenset("""
salad soup stew dish snack treat dessert appetizer entree side sides
pizza sandwich sandwiches burger burgers taco tacos wrap wraps quesadilla
nachos dip dips sauce sauces dressing smoothie shake shakes juice coffee
tea cheesecake pancake pancakes waffle waffles muffin muffins cake cakes
cookie cookies pie pies brownie brownies cereal granola porridge
casserole curry chili lasagna enchilada fajita kebab kebabs goulash
minestrone breakfast lunch dinner brunch supper meal meals food
meat poultry seafood vegetable vegetables veggie veggies fruit fruits
fries chips
""".split())

# Single words that must never claim a medical condition by themselves.
# Every entry was measured with all-MiniLM-L6-v2: it clears the 0.50
# default threshold against some condition label yet does not genuinely
# state a condition — "blood" -> "anemia" 0.56, "heart" 0.57,
# "kidney" -> "chronic_kidney_disease" 0.66, "liver" -> "hepatitis" 0.71,
# "disease" 0.61, "chronic" 0.59, "iron" -> "anemia" 0.56,
# "overweight" -> "underweight" 0.87, "gluten" 0.74, "dairy" 0.57,
# "lactose" 0.80, "fat" -> "obesity" 0.70, "cholesterol" 0.87,
# "bone" -> "osteoporosis" 0.61, "lung" -> "asthma" 0.60,
# "sugar" -> "diabetes" 0.60, "weight" -> "underweight" 0.70, and
# "thyroid" -> "hypothyroidism" 0.74 (hyperthyroidism scores ~0.72, so
# a bare "thyroid" cannot pick the right subtype). Only the bare single
# token is blocked: the same word inside a longer phrase keeps its
# genuine claim ("kidney disease" 0.84, "blood sugar" 0.71,
# "heart disease" 1.00, "iron deficiency" 0.89).
_GENERIC_CONDITION_WORDS = frozenset("""
blood heart kidney liver disease chronic iron overweight gluten dairy
lactose fat cholesterol bone lung sugar weight thyroid
""".split())

# Multi-word phrases that measure closest to a condition label but do not
# state that condition (measured): "blood problem" -> "anemia" 0.51,
# "heart healthy" -> "heart_disease" 0.66, "lose weight" 0.60 /
# "weight loss" 0.62 / "weight issue" 0.52 / "overweight person" 0.75 /
# "overweight people" 0.72 -> "obesity"/"underweight", "gluten free"
# 0.69, "dairy free" 0.53, "sensitive stomach" ->
# "irritable_bowel_syndrome" 0.54, "chronic disease" ->
# "chronic_kidney_disease" 0.69. "sugar" is deliberately NOT a stopword
# so that "blood sugar" -> "diabetes" 0.71 stays claimable; the
# food-context sugar phrases this opens up are blocked here instead
# (all -> "diabetes" unless noted): "high sugar" 0.57, "sugar meal"
# 0.51, "sugar cake" 0.51, "sugar drink" 0.54, "sugar milk" ->
# "lactose_intolerance" 0.52, "sugar levels" 0.50, "sugar control" 0.57,
# "sugar intake" 0.53, "reduce sugar" 0.56, "cut sugar" 0.50.
#
# Matching rule (see _semantic_match): a phrase is blocked when a
# misleading phrase occurs inside it ("heart healthy meal" contains
# "heart healthy"), EXCEPT sugar phrases that are "blood"-qualified:
# "blood sugar levels" contains "sugar levels" but is a genuine diabetes
# mention, so a sugar phrase is excused only when the phrase also
# contains "blood". Genuine diagnostics are never blocked: "blood
# sugar" 0.71, "blood sugar levels" 0.53, "blood sugar problem" 0.63,
# "high blood sugar" 0.66, "blood pressure" 0.75, "kidney disease"
# 0.84, "liver problem" 0.73.
_MISLEADING_CONDITION_PHRASES = frozenset([
    "blood problem", "heart healthy", "lose weight", "weight loss",
    "weight issue", "overweight person", "overweight people",
    "gluten free", "dairy free", "sensitive stomach", "chronic disease",
    "high sugar", "sugar meal", "sugar cake", "sugar drink",
    "sugar milk", "sugar levels", "sugar control", "sugar intake",
    "reduce sugar", "cut sugar",
])

# Fixed reference values for vague descriptive terms (do not invent others).
VAGUE_TERMS = {
    "low sugar":    {"sugar_max": 15},
    "low sodium":   {"sodium_max": 600},
    "high protein": {"protein_min": 20},
    "low calorie":  {"calories_max": 500},
    "high fiber":   {"fiber_min": 5},
}

# Regexes used to detect the vague terms (also accept "low in sugar",
# "low-sugar", "low calories", ...). Kept in sync with VAGUE_TERMS keys.
_VAGUE_PHRASE_RES: Dict[str, re.Pattern] = {
    "low sugar":    re.compile(r"\blow\s*(?:in\s+)?sugar(?:s)?\b"),
    "low sodium":   re.compile(r"\blow\s*(?:in\s+)?sodium\b"),
    "high protein": re.compile(r"\bhigh\s*(?:in\s+)?protein\b"),
    "low calorie":  re.compile(r"\blow\s*(?:in\s+)?calorie(?:s)?\b"),
    "high fiber":   re.compile(r"\bhigh\s*(?:in\s+)?fiber\b"),
}

# Tokens filtered out before building n-gram candidates for semantic matching.
# Nutrient words / units are handled by the numeric & vague steps instead.
_STOPWORDS = frozenset("""
a an and are at be by for from in is it my of on or the to with
i me we you your am is want would like need have has had having please give
something
some any there this that their what which who
eat eating recipe recipes food diet meal meals make made making cooking cook
no without free avoid avoiding exclude excluding only just
gram grams g mg milligram milligrams ounce ounces cup cups
calorie calories kcal cal protein sodium fiber fat
allergy allergic options option show looking
""".split())

# Direction qualifiers for explicit numbers ("under 600 mg sodium", ...).
_MAX_DIRECTION_RE = re.compile(
    r"\b(?:under|below|less than|at most|no more than|max(?:imum)?|"
    r"no higher than|or less|or fewer)\b", re.IGNORECASE)
_MIN_DIRECTION_RE = re.compile(
    r"\b(?:at least|at a minimum|at minimum|minimum|min\b|more than|over|"
    r"no less than)\b", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Vocabulary (fixed, canonical — mirrored from core/constants.py into
# vocab_terms.py so this package stays fully standalone)
# --------------------------------------------------------------------------- #

_MAIN_INGREDIENTS: List[str] = [
    "chicken", "chicken breast", "turkey", "beef", "lamb", "pork",
    "salmon", "tuna", "fish", "shrimp", "crab", "lobster",
    "eggs", "tofu", "tempeh",
    "lentils", "beans", "chickpeas",
    "rice", "brown rice", "pasta", "noodles", "quinoa", "oats", "oatmeal",
    "potato", "sweet potato",
    "avocado", "broccoli", "spinach", "kale", "cauliflower", "mushroom",
    "cheese", "yogurt", "eggplant", "tomato", "cucumber", "carrot", "pumpkin",
]

# (category, canonical key, text-to-embed). Condition / allergy / meal /
# preference entries come from the canonical constants lists so the parser's
# terminology stays consistent with the rest of the system.
_VOCAB: List[Tuple[str, str, str]] = []
for _key, _label in DISEASE_EN.items():
    _VOCAB.append(("condition", _key, _key.replace("_", " ")))
    _VOCAB.append(("condition", _key, _label))
for _key, _label in MEAL_TYPE_EN.items():
    if _key == "any":
        continue
    _VOCAB.append(("meal_type", _key, _label.lower()))
for _key, _label in ALLERGY_EN.items():
    _VOCAB.append(("allergy", _key, _key))
    _VOCAB.append(("allergy", _key, _label))
for _key, _label in PREFERENCE_EN.items():
    if _key == "no_preference":
        continue
    _VOCAB.append(("diet_preference", _key, _key.replace("_", " ")))
    _VOCAB.append(("diet_preference", _key, _label))
for _ingredient in _MAIN_INGREDIENTS:
    _VOCAB.append(("main_ingredient", _ingredient, _ingredient))

del _key, _label, _ingredient

# Lazy singleton: SentenceTransformer is loaded once on first semantic call.
_MODEL = None
_MODEL_FAILED = False
_VOCAB_EMBEDDINGS: Optional[np.ndarray] = None


def _load_model():
    """Load the Sentence-BERT model (pretrained only, zero fine-tuning)."""
    global _MODEL, _MODEL_FAILED, _VOCAB_EMBEDDINGS
    if _MODEL is None and not _MODEL_FAILED:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(MODEL_NAME)
            _VOCAB_EMBEDDINGS = _MODEL.encode(
                [text for (_, _, text) in _VOCAB],
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
        except Exception as exc:  # offline or missing dependency
            _MODEL_FAILED = True
            warnings.warn(
                "Semantic matcher unavailable (%s); only numeric/vague/"
                "negation extraction will be used." % exc, stacklevel=2
            )
    return _MODEL


# --------------------------------------------------------------------------- #
# 1. Explicit numeric extraction (regex, independent of Sentence-BERT)
# --------------------------------------------------------------------------- #

# schema field per nutrient and its default direction (the output schema
# only has protein_min / fiber_min as minimums and maxes for the rest).
_FIELD_DIRECTIONS: Dict[str, Tuple[str, str]] = {
    "protein":  ("protein_min",  "min"),
    "fiber":    ("fiber_min",    "min"),
    "sugar":    ("sugar_max",    "max"),
    "sodium":   ("sodium_max",   "max"),
    "calories": ("calories_max", "max"),
}

_NUMERIC_PATTERNS: Dict[str, List[str]] = {
    "protein": [
        r"(\d+)\s*(?:grams?|g)\s*(?:of\s+)?protein",
        r"protein\s*(?:at least|minimum|min\b|more than|over|no less than)"
        r"\s*(\d+)\s*(?:grams?|g)?",
        r"protein\s*(?:of\s*)?(\d+)\s*(?:grams?|g)?",
    ],
    "sugar": [
        r"(\d+)\s*(?:grams?|g)\s*(?:of\s+)?sugar",
        r"sugar\s*(?:under|below|less than|at most|no more than|max(?:imum)?)"
        r"\s*(\d+)\s*(?:grams?|g)?",
        r"sugar\s*(?:of\s*)?(\d+)\s*(?:grams?|g)?",
    ],
    "sodium": [
        r"(\d+)\s*(?:mg|milligrams?)\s*(?:of\s+)?(?:sodium|salt)",
        r"sodium\s*(?:under|below|less than|at most|no more than|max(?:imum)?)"
        r"\s*(\d+)",
        r"(?:under|below|less than|at most|no more than|max(?:imum)?)\s*(\d+)"
        r"\s*(?:mg|milligrams?)?\s*(?:of\s+)?sodium",
        r"sodium\s*(?:of\s*)?(\d+)\s*(?:mg|milligrams?)?",
    ],
    "fiber": [
        r"(\d+)\s*(?:grams?|g)\s*(?:of\s+)?fiber",
        r"fiber\s*(?:under|below|less than|at most|no more than)\s*(\d+)"
        r"\s*(?:grams?|g)?",
        r"fiber\s*(?:of\s*)?(\d+)\s*(?:grams?|g)?",
    ],
    "calories": [
        r"(\d+)\s*(?:calories|kcal|cal)\b",
        r"(?:under|below|less than|at most|no more than|max(?:imum)?)\s*(\d+)"
        r"\s*(?:calories|kcal)\b",
        r"calories\s*(?:under|below|less than|at most|no more than|max(?:imum)?)"
        r"\s*(\d+)",
    ],
}


def _match_direction(match: re.Match, text: str) -> str:
    """Return 'min' / 'max' / 'default' for an explicit-number match."""
    context = text[max(0, match.start() - 30):match.end()]
    if _MAX_DIRECTION_RE.search(context):
        return "max"
    if _MIN_DIRECTION_RE.search(context):
        return "min"
    return "default"


def extract_numeric_constraints(text: str) -> dict:
    """
    Extract explicit numeric constraints ("20 grams protein", "600 mg
    sodium", "under 500 calories") with regex — no semantic model involved.

    Direction qualifiers that contradict the output schema (e.g. "protein
    under 30 g" would need protein_max, which the schema has no key for)
    are skipped so a wrong direction is never encoded.
    """
    result = {}
    for field, patterns in _NUMERIC_PATTERNS.items():
        schema_key, default_direction = _FIELD_DIRECTIONS[field]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match is None:
                continue
            direction = _match_direction(match, text)
            if direction == "default":
                direction = default_direction
            if direction != default_direction:
                break  # qualifier contradicts schema direction -> not representable
            result[schema_key] = int(match.group(1))
            break
    return result


# --------------------------------------------------------------------------- #
# 2. Vague descriptive terms -> fixed thresholds
# --------------------------------------------------------------------------- #

def _apply_vague_terms(text: str, filters: dict) -> None:
    """Apply VAGUE_TERMS defaults (lowest priority; explicit numbers win)."""
    normalized = re.sub(r"-", " ", text.lower())
    for phrase, pattern in _VAGUE_PHRASE_RES.items():
        if pattern.search(normalized):
            filters.update(VAGUE_TERMS[phrase])


# --------------------------------------------------------------------------- #
# 3. Allergies — explicit negation / marker phrases only
# --------------------------------------------------------------------------- #

_NEGATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:no|without|avoid(?:ing)?|exclude|excluding)\s+([a-z]+)", re.I),
    re.compile(r"\bfree\s+of\s+([a-z]+)", re.I),
    re.compile(r"\b([a-z]+)\s*-?\s*free\b", re.I),
]

_ALLERGY_MARKER_RE = re.compile(
    r"\b(?:allergic to|allergy to|allergy)\s+([a-z]+)", re.I)

# Canonical keys are the ALLERGY_EN keys ("milk", "peanuts", ...).
# TODO: confirm key name with Expert System owner — the output-schema example
#       shows "dairy"/"nuts", but core/constants.py canonical names are
#       "milk"/"peanuts"; we always emit the canonical constants keys.
_ALLERGY_ALIASES: Dict[str, str] = {
    "milk": "milk", "dairy": "milk", "lactose": "milk", "cheese": "milk",
    "butter": "milk", "cream": "milk",
    "peanut": "peanuts", "peanuts": "peanuts", "nut": "peanuts",
    "nuts": "peanuts", "tree nut": "peanuts", "tree nuts": "peanuts",
    "egg": "eggs", "eggs": "eggs",
    "seafood": "seafood", "fish": "seafood", "shellfish": "seafood",
    "shrimp": "seafood", "prawn": "seafood",
    "soy": "soy", "soya": "soy", "soybean": "soy",
    "gluten": "gluten", "wheat": "gluten",
    "sesame": "sesame",
}


def _extract_allergy_heads(text: str) -> List[str]:
    """Heads of negation / allergy-marker phrases, e.g. "no dairy" -> "dairy"."""
    lowered = text.lower()
    heads = []
    for pattern in _NEGATION_PATTERNS:
        heads.extend(pattern.findall(lowered))
    heads.extend(_ALLERGY_MARKER_RE.findall(lowered))
    return [h.strip() for h in heads if h.strip() and h not in _STOPWORDS]


def _resolve_allergies(heads: List[str]) -> List[str]:
    """Map allergy heads to canonical ALLERGY_EN keys (alias -> semantic)."""
    resolved: List[str] = []
    seen = set()
    semantic_heads = []
    for head in heads:
        key = _ALLERGY_ALIASES.get(head)
        if key is not None:
            if key not in seen:
                seen.add(key)
                resolved.append(key)
        else:
            semantic_heads.append(head)
    if semantic_heads:
        matches = _semantic_match(semantic_heads, categories={"allergy"})
        key = matches.get("allergy")
        if key is not None and key not in seen:
            resolved.append(key)
    return resolved


def _extract_suppressed_tokens(text: str) -> set:
    """Tokens that must not feed generic entity matching ("no dairy" etc.)."""
    heads = []
    for pattern in _NEGATION_PATTERNS:
        heads.extend(pattern.findall(text.lower()))
    return set(heads) | {"no", "without", "free", "avoid", "avoiding",
                         "exclude", "excluding"}


# --------------------------------------------------------------------------- #
# 4. Semantic entity matching (Sentence-BERT, pretrained only)
# --------------------------------------------------------------------------- #

def _candidate_phrases(text: str, suppressed_tokens: set) -> List[str]:
    """1-3 word n-grams of the query, minus stopwords / negated / vague parts.

    Returned longest-first; ties are broken by first occurrence in the
    query so the ordering (and the tie-break it feeds) is deterministic.
    """
    lowered = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    tokens = [t for t in lowered.split()
              if t not in _STOPWORDS and not t.isdigit()]
    first_index = {}  # n-gram -> earliest token position
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i + n]
            if any(t in suppressed_tokens for t in window):
                continue
            gram = " ".join(window)
            if gram not in first_index:
                first_index[gram] = i
    filtered = [g for g in first_index
                if not any(vr.search(g) for vr in _VAGUE_PHRASE_RES.values())]
    return sorted(filtered, key=lambda g: (-len(g.split()), first_index[g]))


def _semantic_match(phrases, categories: Optional[set] = None) -> dict:
    """
    Cosine similarity of each query phrase against the precomputed
    vocabulary embeddings. The best match above the threshold wins; every
    category is claimed by its single highest-scoring phrase.

    Every category applies its own claim gate to a phrase (measured
    rationale in the constants above):

      main_ingredient — THRESHOLD_MAIN_INGREDIENT, the
                        MAIN_INGREDIENT_MIN_MARGIN gap over the
                        second-best ingredient, and at least one token
                        outside _GENERIC_FOOD_WORDS in the phrase.
      meal_type       — MEAL_TYPE_MIN_MARGIN gap over the second-best
                        meal (misleading words like "snack" hover at
                        margin ~0.00 while genuine meals keep >= 0.07).
      condition       — the phrase must not be a bare
                        _GENERIC_CONDITION_WORDS single token nor contain
                        a _MISLEADING_CONDITION_PHRASES multi-word phrase
                        (blood-qualified sugar phrases like "blood sugar
                        levels" stay claimable).
      diet_preference — only the higher THRESHOLD_DIET_PREFERENCE.

    A phrase whose strongest candidate fails its category gate is skipped
    entirely (it cannot be trusted for any category either).
    """
    if not phrases:
        return {}
    model = _load_model()
    if model is None or _VOCAB_EMBEDDINGS is None:
        return {}
    embeddings = model.encode(
        list(phrases), normalize_embeddings=True,
        batch_size=64, show_progress_bar=False,
    )
    scores = _VOCAB_EMBEDDINGS @ embeddings.T  # (V, P)
    allowed = None
    if categories is not None:
        allowed = {i for i, (cat, _, _) in enumerate(_VOCAB)
                   if cat in categories}

    # Per-phrase category gates (measured rationale above the constants).
    ing_rows = [i for i, (cat, _, _) in enumerate(_VOCAB)
                if cat == "main_ingredient"]
    meal_rows = [i for i, (cat, _, _) in enumerate(_VOCAB)
                 if cat == "meal_type"]
    ing_gate = []
    meal_gate = []
    for p in range(len(phrases)):
        ing_sorted = np.sort(scores[ing_rows, p])[::-1]
        top1 = float(ing_sorted[0])
        top2 = float(ing_sorted[1]) if len(ing_sorted) > 1 else 0.0
        ing_gate.append(
            top1 >= THRESHOLD_MAIN_INGREDIENT
            and top1 - top2 >= MAIN_INGREDIENT_MIN_MARGIN
            and any(t not in _GENERIC_FOOD_WORDS
                    for t in phrases[p].split())
        )
        meal_sorted = np.sort(scores[meal_rows, p])[::-1]
        meal_gate.append(
            float(meal_sorted[0]) - float(meal_sorted[1])
            >= MEAL_TYPE_MIN_MARGIN
        )
    cond_gate = []
    for p in range(len(phrases)):
        ph = phrases[p]
        blocked = (
            ph in _GENERIC_CONDITION_WORDS
            or any(m in ph and ("blood" not in ph or "sugar" not in m)
                   for m in _MISLEADING_CONDITION_PHRASES)
        )
        cond_gate.append(not blocked)

    candidates = []  # (score, category, key, phrase_rank)
    for p in range(len(phrases)):
        order = np.argsort(-scores[:, p])
        for vi in order:
            if allowed is not None and vi not in allowed:
                continue
            category, key, _ = _VOCAB[vi]
            if scores[vi, p] < (THRESHOLD_DIET_PREFERENCE
                                if category == "diet_preference"
                                else THRESHOLD_MAIN_INGREDIENT
                                if category == "main_ingredient"
                                else THRESHOLD_DEFAULT):
                break
            if (category == "main_ingredient" and not ing_gate[p]
                    or category == "meal_type" and not meal_gate[p]
                    or category == "condition" and not cond_gate[p]):
                break
            candidates.append((float(scores[vi, p]), category, key, p))
            break

    best = {}
    # Exact ties (e.g. "shrimp pasta": both words match their own vocab
    # entry at 1.0) are broken by phrase order, which is deterministic and
    # follows first occurrence in the query — not by set/hash order.
    for score, category, key, p in sorted(
            candidates, key=lambda t: (-t[0], t[3])):
        if category not in best:
            best[category] = key
    return best


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def parse_query(text: str) -> dict:
    """
    Converts a free-text English search query into a structured filter dict.
    """
    filters = {
        "condition":        None,  # e.g. "diabetes", "hypertension"
        "meal_type":        None,  # "breakfast" / "lunch" / "dinner"
        "protein_min":      None,  # grams
        "sugar_max":        None,  # grams
        "sodium_max":       None,  # mg
        "fiber_min":        None,  # grams
        "calories_max":     None,
        "allergy":          [],    # canonical ALLERGY_EN keys
        "diet_preference":  None,
        "main_ingredient":  None,
    }
    if not text or not text.strip():
        return filters

    # 1) Vague descriptive terms (lowest priority).
    _apply_vague_terms(text, filters)

    # 2) Explicit numbers — conflict rule: explicit always wins over vague.
    filters.update(extract_numeric_constraints(text))

    # 3) Allergies — only from explicit negation / marker phrases.
    allergy_heads = _extract_allergy_heads(text)
    filters["allergy"] = _resolve_allergies(allergy_heads)
    # TODO: confirm key name with Expert System owner — the output-schema
    #       example shows "dairy"/"nuts" but core/constants.py canonical
    #       allergy names are "milk"/"peanuts"; we emit the constants keys.

    # 4) Semantic entity matching (Sentence-BERT) for the rest. The allergy
    #    category is NOT claimable here — allergies come only from explicit
    #    negation / marker phrases (step 3), so a positive mention such as
    #    "egg breakfast" can never produce a false allergy, and the mention
    #    stays available to claim main_ingredient instead.
    suppressed = _extract_suppressed_tokens(text)
    phrases = _candidate_phrases(text, suppressed)
    matches = _semantic_match(
        phrases,
        categories={"condition", "meal_type", "diet_preference",
                    "main_ingredient"},
    )
    filters["condition"] = matches.get("condition")
    filters["meal_type"] = matches.get("meal_type")
    # TODO: confirm key name with Expert System owner — the output-schema
    #       example mentions "keto", which is NOT in core/constants.py
    #       PREFERENCE_EN and therefore can never match.
    filters["diet_preference"] = matches.get("diet_preference")
    filters["main_ingredient"] = matches.get("main_ingredient")
    return filters
