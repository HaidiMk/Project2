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


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

THRESHOLD_DEFAULT = 0.50

THRESHOLD_DIET_PREFERENCE = 0.70
THRESHOLD_MAIN_INGREDIENT = 0.65

MAIN_INGREDIENT_MIN_MARGIN = 0.10

MEAL_TYPE_MIN_MARGIN = 0.05

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

_GENERIC_CONDITION_WORDS = frozenset("""
blood heart kidney liver disease chronic iron overweight gluten dairy
lactose fat cholesterol bone lung sugar weight thyroid
""".split())


_MISLEADING_CONDITION_PHRASES = frozenset([
    "blood problem", "heart healthy", "lose weight", "weight loss",
    "weight issue", "overweight person", "overweight people",
    "gluten free", "dairy free", "sensitive stomach", "chronic disease",
    "high sugar", "sugar meal", "sugar cake", "sugar drink",
    "sugar milk", "sugar levels", "sugar control", "sugar intake",
    "reduce sugar", "cut sugar",
])

VAGUE_TERMS = {
    "low sugar":    {"sugar_max": 15},
    "low sodium":   {"sodium_max": 600},
    "high protein": {"protein_min": 20},
    "low calorie":  {"calories_max": 500},
    "high fiber":   {"fiber_min": 5},
}

_VAGUE_PHRASE_RES: Dict[str, re.Pattern] = {
    "low sugar":    re.compile(r"\blow\s*(?:in\s+)?sugar(?:s)?\b"),
    "low sodium":   re.compile(r"\blow\s*(?:in\s+)?sodium\b"),
    "high protein": re.compile(r"\bhigh\s*(?:in\s+)?protein\b"),
    "low calorie":  re.compile(r"\blow\s*(?:in\s+)?calorie(?:s)?\b"),
    "high fiber":   re.compile(r"\bhigh\s*(?:in\s+)?fiber\b"),
}

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

_MAX_DIRECTION_RE = re.compile(
    r"\b(?:under|below|less than|at most|no more than|max(?:imum)?|"
    r"no higher than|or less|or fewer)\b", re.IGNORECASE)
_MIN_DIRECTION_RE = re.compile(
    r"\b(?:at least|at a minimum|at minimum|minimum|min\b|more than|over|"
    r"no less than)\b", re.IGNORECASE)

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
        except Exception as exc:  
            _MODEL_FAILED = True
            warnings.warn(
                "Semantic matcher unavailable (%s); only numeric/vague/"
                "negation extraction will be used." % exc, stacklevel=2
            )
    return _MODEL


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
    context = text[max(0, match.start() - 30):match.end()]
    if _MAX_DIRECTION_RE.search(context):
        return "max"
    if _MIN_DIRECTION_RE.search(context):
        return "min"
    return "default"


def extract_numeric_constraints(text: str) -> dict:
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


def _apply_vague_terms(text: str, filters: dict) -> None:
    """Apply VAGUE_TERMS defaults (lowest priority; explicit numbers win)."""
    normalized = re.sub(r"-", " ", text.lower())
    for phrase, pattern in _VAGUE_PHRASE_RES.items():
        if pattern.search(normalized):
            filters.update(VAGUE_TERMS[phrase])


_NEGATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:no|without|avoid(?:ing)?|exclude|excluding)\s+([a-z]+)", re.I),
    re.compile(r"\bfree\s+of\s+([a-z]+)", re.I),
    re.compile(r"\b([a-z]+)\s*-?\s*free\b", re.I),
]

_ALLERGY_MARKER_RE = re.compile(
    r"\b(?:allergic to|allergy to|allergy)\s+([a-z]+)", re.I)

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
    lowered = text.lower()
    heads = []
    for pattern in _NEGATION_PATTERNS:
        heads.extend(pattern.findall(lowered))
    heads.extend(_ALLERGY_MARKER_RE.findall(lowered))
    return [h.strip() for h in heads if h.strip() and h not in _STOPWORDS]


def _resolve_allergies(heads: List[str]) -> List[str]:
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
    heads = []
    for pattern in _NEGATION_PATTERNS:
        heads.extend(pattern.findall(text.lower()))
    return set(heads) | {"no", "without", "free", "avoid", "avoiding",
                         "exclude", "excluding"}



def _candidate_phrases(text: str, suppressed_tokens: set) -> List[str]:
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
    if not phrases:
        return {}
    model = _load_model()
    if model is None or _VOCAB_EMBEDDINGS is None:
        return {}
    embeddings = model.encode(
        list(phrases), normalize_embeddings=True,
        batch_size=64, show_progress_bar=False,
    )
    scores = _VOCAB_EMBEDDINGS @ embeddings.T  
    allowed = None
    if categories is not None:
        allowed = {i for i, (cat, _, _) in enumerate(_VOCAB)
                   if cat in categories}

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

    candidates = []  
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
    for score, category, key, p in sorted(
            candidates, key=lambda t: (-t[0], t[3])):
        if category not in best:
            best[category] = key
    return best



def parse_query(text: str) -> dict:
    """
    Converts a free-text English search query into a structured filter dict.
    """
    filters = {
        "condition":        None,  
        "meal_type":        None,  
        "protein_min":      None,  
        "sugar_max":        None,  
        "sodium_max":       None,  
        "fiber_min":        None,  
        "calories_max":     None,
        "allergy":          [],    
        "diet_preference":  None,
        "main_ingredient":  None,
    }
    if not text or not text.strip():
        return filters

    _apply_vague_terms(text, filters)

    filters.update(extract_numeric_constraints(text))

    allergy_heads = _extract_allergy_heads(text)
    filters["allergy"] = _resolve_allergies(allergy_heads)
    suppressed = _extract_suppressed_tokens(text)
    phrases = _candidate_phrases(text, suppressed)
    matches = _semantic_match(
        phrases,
        categories={"condition", "meal_type", "diet_preference",
                    "main_ingredient"},
    )
    filters["condition"] = matches.get("condition")
    filters["meal_type"] = matches.get("meal_type")
    filters["diet_preference"] = matches.get("diet_preference")
    filters["main_ingredient"] = matches.get("main_ingredient")
    return filters
