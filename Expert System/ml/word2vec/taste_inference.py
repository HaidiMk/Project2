import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]        
sys.path.insert(0, str(BASE_DIR))

from gensim.models import Word2Vec                    
from ml.word2vec.taste_concepts import CONCEPT_MAP    

try:
    from rapidfuzz import fuzz as _rf_fuzz                
    from rapidfuzz import process as _rf_process         
    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False
    import difflib                                  

MODEL_PATH = BASE_DIR / "data" / "word2vec_ingredients.model"
EMBED_PATH = BASE_DIR / "data" / "recipe_taste_embeddings.pkl"

NEUTRAL_SCORE = 0.5

FUZZY_THRESHOLD = 0.82
FUZZY_MIN_TOKEN_LEN = 5

_model = None
_wv = None
_embeddings = None


def _load():
    global _model, _wv, _embeddings
    if _model is not None:
        return _wv, _embeddings

    _model = Word2Vec.load(str(MODEL_PATH))
    _wv = _model.wv

    with open(EMBED_PATH, "rb") as f:
        _embeddings = pickle.load(f)

    return _wv, _embeddings


def _tokenize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = text.split()
    tokens = list(words)
    tokens += [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return tokens


DISLIKE_TRIGGERS = [
    r"\bdislike\b",
    r"\bdon't like\b",
    r"\bdo not like\b",
    r"\bhate\b",
    r"\bavoid\b",
    r"\bnot a fan of\b",
]

BUT_WORD_RE = re.compile(r"\bbut\b", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"[;.]")


def _split_clauses(text: str):
    text = BUT_WORD_RE.sub(";", text)
    return [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]


def _is_disliked_clause(clause: str) -> bool:
    for trigger in DISLIKE_TRIGGERS:
        if re.search(trigger, clause, re.IGNORECASE):
            return True
    return False


def _strip_trigger_phrases(clause: str) -> str:
    cleaned = clause
    for trigger in DISLIKE_TRIGGERS:
        cleaned = re.sub(trigger, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _fuzzy_match_token(token: str, wv, threshold: float = FUZZY_THRESHOLD):
    if not isinstance(token, str) or len(token) < FUZZY_MIN_TOKEN_LEN:
        return None

    if _HAVE_RAPIDFUZZ:
        result = _rf_process.extractOne(
            token, wv.index_to_key, scorer=_rf_fuzz.ratio,
            score_cutoff=threshold * 100.0)
        if result is None:
            return None
        return result[0]

    matches = difflib.get_close_matches(token, wv.index_to_key, n=1, cutoff=threshold)
    return matches[0] if matches else None


def _resolve_token(token: str, wv):
    if token in wv:
        return [wv[token]]

    targets = CONCEPT_MAP.get(token)
    if targets:
        return [wv[t] for t in targets if t in wv]

    matched = _fuzzy_match_token(token, wv)
    if matched:
        return [wv[matched]]

    return []


def taste_vector_from_text(text: str):
    wv, _ = _load()

    if not text or not isinstance(text, str) or not text.strip():
        return None

    liked_vectors = []
    disliked_vectors = []

    for clause in _split_clauses(text):
        if _is_disliked_clause(clause):
            tokens = _tokenize_text(_strip_trigger_phrases(clause))
            bucket = disliked_vectors
        else:
            tokens = _tokenize_text(clause)
            bucket = liked_vectors

        for token in tokens:
            bucket.extend(_resolve_token(token, wv))

    liked_vec = np.mean(np.vstack(liked_vectors), axis=0) if liked_vectors else None
    disliked_vec = np.mean(np.vstack(disliked_vectors), axis=0) if disliked_vectors else None

    if liked_vec is not None and disliked_vec is not None:
        return liked_vec - disliked_vec
    if liked_vec is not None:
        return liked_vec
    if disliked_vec is not None:
        return -disliked_vec       
    return None


def taste_score(df: pd.DataFrame, user_text: str) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    wv, embeddings = _load()
    user_vec = taste_vector_from_text(user_text)

    if user_vec is None:
        print("WARNING: taste_score — no ingredients recognized from input "
              f"text ({user_text!r}); returning neutral scores (0.5).")
        return np.full(n, NEUTRAL_SCORE, dtype=np.float32)

    norm_u = np.linalg.norm(user_vec)
    if norm_u == 0:
        print("WARNING: taste_score — degenerate (zero) user vector from input "
              f"text ({user_text!r}); returning neutral scores (0.5).")
        return np.full(n, NEUTRAL_SCORE, dtype=np.float32)

    user_vec = user_vec / norm_u
    scores = np.full(n, NEUTRAL_SCORE, dtype=np.float32)

    ids = df["RecipeId"].astype(int).tolist()
    for i, rid in enumerate(ids):
        vec = embeddings.get(rid)
        if vec is None:
            continue                      
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        cos = float(np.dot(user_vec, vec) / norm)
        scores[i] = (cos + 1.0) / 2.0     

    return scores
