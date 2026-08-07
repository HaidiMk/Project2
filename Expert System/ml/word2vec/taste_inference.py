"""
taste_inference.py — الاستدلال على درجة الذوق (taste score) من Word2Vec
======================================================================
يحمل (مرة واحدة فقط، singleton) نموذج Word2Vec لمكونات الوصفات + قاموس
تضمينات الوصفات المحسوبة مسبقاً في build_taste_embeddings.py:
    - Expert System/data/word2vec_ingredients.model
    - Expert System/data/recipe_taste_embeddings.pkl
      (dict: RecipeId int -> np.ndarray shape (150,) float32)

الدوال:
    taste_vector_from_text(text) -> np.ndarray | None
        يرمّز النص الحر إلى مكوّنات مرشّحة: خفض الحالة، إزالة علامات
        الترقيم، التقسيم على الفراغات، ثم محاولة الكلمات المفردة
        والأزواج المتجاورة (بشرطة سفلية) معاً — لأن مفردات النموذج
        تستخدم رموزاً بشرطة سفلية للمكونات متعددة الكلمات
        ("olive oil" -> "olive_oil" أولاً، وعند الفشل "olive" و "oil").

        معالجة الكره/النفور (dislike):
        - يُقسَّم النص أولاً إلى جمل (clauses) على حدود ";" و "." وعلى
          كلمة "but" القائمة بذاتها (حدود كلمة كاملة — لا يُقسَّم على
          الفواصل "،" لأنها تُعدّد المكونات داخل الجملة الواحدة).
        - كل جملة تُصنَّف كـ LIKED أو DISLIKED: DISLIKED إن احتوت
          إحدى العبارات (بلا تمييز حالة، بحدود كلمة):
              dislike | don't like | do not like | hate | avoid | not a fan of
          وإلا LIKED (يشمل الجمل بلا كلمة نفي إطلاقاً، والجمل الموجبة
          مثل love/like/enjoy — لا معالجة خاصة لها).
        - في جملة DISLIKED تُحذف عبارة التحفيز نفسها من الرموز
          المرشّحة (مثل "dislike" أو "hate") قبل الترميز.
        - يُجمع متجهان منفصلان: liked_vectors و disliked_vectors، ثم:
              كلاهما موجود  -> net = liked_vec - disliked_vec
              liked فقط     -> net = liked_vec   (كما كان سلوك اليوم)
              disliked فقط  -> net = -disliked_vec
                               (حالة جديدة: بلا مكونات محبوبة إطلاقاً،
                               يتجه متجه المستخدم بعيداً عن المكروهة —
                               نفور خالص)
              لا شيء        -> None (كما كان)
        - لا يُطبَّع net إلى طول الوحدة هنا — taste_score() هو من
          يطبّعه لاحقاً (user_vec = user_vec / ||user_vec||). هذا يطابق
          اتفاق التطبيع الموجود سابقاً.

        حل الرموز الفردية داخل كل جملة (3 خطوات، بالترتيب):
            1. مطابقة تامة في مفردات النموذج (السلوك الحالي —
               الرموز المطابقة تبقى حرفياً كما كانت، بلا أي تغيير).
            2. تمدد مفهوم: إن كان الرمز مفتاحاً في CONCEPT_MAP
               (ml/word2vec/taste_concepts.py — مثل "spicy" أو "italian")
               تُضاف كل رموز الهدف كنواقل (وليس واحداً فقط).
            3. مطابقة تقريبية (fuzzy): ملاذ أخير فقط، للرموز ذات
               4 أحرف فأكثر، بعتبة تشابه FUZZY_THRESHOLD (0.82) —
               مثل "garlik"→garlic و "chiken"→chicken.
            وينطبق هذا الترتيب نفسه داخل جمل DISLIKED تماماً
            (مثال: "I dislike spicy food" يوسّع "spicy" ويضع رموز
            التوابل الحارة في سلة المكروه).

    taste_score(df, user_text) -> np.ndarray
        درجة ذوق لكل وصفة في df:
            cos = تشابه جيب التمام بين متجه المستخدم ومتجه الوصفة
            score = (cos + 1) / 2        # إعادة تصغير من [-1,1] إلى [0,1]
        - لا متجه مستخدم (لا مكونات معروفة) → كل القيم 0.5 (محايد) + تحذير.
        - متجه مستخدم منحل (صِفري — مثل "garlic; dislike garlic")
          → كل القيم 0.5 (محايد) + تحذير.
        - وصفة بلا تضمين محفوظ (الوصفات المحذوفة/بلا مفردات في Step 2)
          → 0.5 (محايد) لذلك الصف.
        يعيد مصفوفة بنفس ترتيب صفوف df.

التكامل:
    TOPSIS/topsis_model.rank_with_topsis(df, goal, user_taste_text)
    يستدعي taste_score ويضيف عمود _taste_score بوزن 0.15:
        final = 0.35*TOPSIS + 0.35*AI + 0.15*expert_norm + 0.15*taste
    (بدون user_taste_text تبقى المعادلة السابقة 0.4/0.4/0.2 كما هي.)
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]          # Expert System/
sys.path.insert(0, str(BASE_DIR))

from gensim.models import Word2Vec                      # noqa: E402
from ml.word2vec.taste_concepts import CONCEPT_MAP      # noqa: E402

try:
    from rapidfuzz import fuzz as _rf_fuzz                    # noqa: E402
    from rapidfuzz import process as _rf_process              # noqa: E402
    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False
    import difflib                                            # noqa: E402

MODEL_PATH = BASE_DIR / "data" / "word2vec_ingredients.model"
EMBED_PATH = BASE_DIR / "data" / "recipe_taste_embeddings.pkl"

NEUTRAL_SCORE = 0.5

# عتبة المطابقة التقريبية — جرّبتها يدوياً على "garlik"→garlic (0.833)
# و "chiken"→chicken (0.923) دون أخطاء إيجابية للكلمات القصيرة
# (انظر smoke test). ملاحظتان من التجربة:
#   - يجب استخدام fuzz.ratio البسيط كمعيار، وليس WRatio الافتراضي —
#     WRatio يستخدم partial-ratio فتطابق كلمات إنجليزية شائعة قصيرة
#     كأجزاء متتالية (love ⊂ cloves، food ⊂ best_foods_mayonnaise).
#   - الرموز الأقصر من 5 أحرف تُستثنى كلياً: "love" تطابق "clove"
#     حتى بـ ratio البسيط.
FUZZY_THRESHOLD = 0.82
FUZZY_MIN_TOKEN_LEN = 5

_model = None
_wv = None
_embeddings = None


def _load():
    """حمّل النموذج + التضمينات مرة واحدة فقط (singleton)."""
    global _model, _wv, _embeddings
    if _model is not None:
        return _wv, _embeddings

    _model = Word2Vec.load(str(MODEL_PATH))
    _wv = _model.wv

    with open(EMBED_PATH, "rb") as f:
        _embeddings = pickle.load(f)

    return _wv, _embeddings


def _tokenize_text(text: str):
    """نص حر -> رموز مرشّحة: كلمات مفردة + أزواج متجاورة بشرطة سفلية.

    ملاحظة: تُحفَظ الواصلة (-) لأن بعض رموز المفردات تحويها
    (مثل "chili-flavored_olive_oil")، وتُزال بقية علامات الترقيم.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = text.split()
    tokens = list(words)
    tokens += [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return tokens


# عبارات النفور — أي جملة تحتوي واحدة منها تُصنَّف DISLIKED
DISLIKE_TRIGGERS = [
    r"\bdislike\b",
    r"\bdon't like\b",
    r"\bdo not like\b",
    r"\bhate\b",
    r"\bavoid\b",
    r"\bnot a fan of\b",
]

# "but" ككلمة قائمة بذاتها (حدود كلمة) — حدود تقسيم الجمل؛ لكن "butter" لا تُقسَّم
BUT_WORD_RE = re.compile(r"\bbut\b", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"[;.]")


def _split_clauses(text: str):
    """قسّم النص الحر إلى جمل على ';' و '.' و 'but' (لا على الفواصل '،')."""
    text = BUT_WORD_RE.sub(";", text)
    return [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]


def _is_disliked_clause(clause: str) -> bool:
    """هل الجملة تعبّر عن كره/نفور (أي عبارة تحفيز موجودة)؟"""
    for trigger in DISLIKE_TRIGGERS:
        if re.search(trigger, clause, re.IGNORECASE):
            return True
    return False


def _strip_trigger_phrases(clause: str) -> str:
    """احذف عبارات التحفيز (dislike/hate/...) من الجملة قبل الترميز."""
    cleaned = clause
    for trigger in DISLIKE_TRIGGERS:
        cleaned = re.sub(trigger, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _fuzzy_match_token(token: str, wv, threshold: float = FUZZY_THRESHOLD):
    """أقرب رمز في المفردات بتشابه >= العتبة — None وإلا.

    يُطبَّق كملاذ أخير فقط (بعد المطابقة التامة وتمدد المفهوم)،
    وفقط للرموز ذات 4 أحرف فأكثر لتجنب الأخطاء الإيجابية للكلمات
    القصيرة. مكتبة rapidfuzz (C-مُسرَّعة) إن توفرت، وإلا difflib.
    """
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
    """حل رمز مرشّح إلى قائمة نواقل — بالترتيب:
        1) مطابقة تامة في المفردات (السلوك الحالي حرفياً)
        2) تمدد مفهوم (كل رموز الهدف في CONCEPT_MAP)
        3) مطابقة تقريبية (fuzzy) — ملاذ أخير للرموز >= 4 أحرف
    """
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
    """متجه ذوق المستخدم من نص حر — None إن لم يُعرف أي مكوّن.

    يحلل الجمل (؛ / . / but) ويجمع المتجهات إلى حبيتين: liked و disliked.
    النتيجة: liked - disliked (أو أحدهما فقط، أو نفور خالص -disliked
    عند غياب أي مكوّن محبوب). لا يطبّع إلى طول وحدة (يقوم taste_score
    بذلك لاحقاً — اتفاق التطبيع الحالي).
    """
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
        return -disliked_vec          # نفور خالص: بلا مكوّنات محبوبة إطلاقاً
    return None


def taste_score(df: pd.DataFrame, user_text: str) -> np.ndarray:
    """درجة ذوق لكل وصفة في df (محاذاة ترتيب صفوف df)."""
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
            continue                        # تبقى 0.5 المحايدة
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        cos = float(np.dot(user_vec, vec) / norm)
        scores[i] = (cos + 1.0) / 2.0       # [-1, 1] -> [0, 1]

    return scores
