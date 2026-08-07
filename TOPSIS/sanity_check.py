"""
sanity_check.py — تحقق شامل من تكامل طبقة الصحة الذكية (AI health)
====================================================================
يفحص:
    1. 3 ملفات تعريف (سكري/زيادة عضل/مسنّ) — دون أخطاء
    2. أعمدة الدرجات _expert_score و _ai_health_score موجودة وبلا NaN
    3. الترتيب القديم (TOPSIS فقط) مقابل الجديد (0.4*TOPSIS + 0.4*AI
       + 0.2*expert_normalized) — وأفضل 5 متداخلة جزئياً
    4. لا عودة لوصفات غير آمنة: الترتيب المدمج يعمل على الوصفات
       الآمنة فقط حصراً
    5. درجة الذوق (taste): تشغيل ملف السكري مرتين — بدون نص ذوق
       (توافق رجعي كامل مع المعادلة السابقة 0.4/0.4/0.2) وبعد نص
       ذوق ("أحب الثوم وزيت الزيتون والدجاج وأكره المأكولات
       البحرية") — عمود _taste_score موجود وبلا NaN وداخل [0,1]
       والترتيب تغيّر؛ وأخيراً نص عشوائي (nonsense) يتحول إلى
       درجات محايدة 0.5 دون أخطاء.
    6. انحدار معالجة الكره (dislike): ملف 25 سنة/أنثى بلا حالات
       وهدف healthy_eating — بنص "chocolate, vanilla, cinnamon,
       sweet desserts; dislike garlic and ginger" يجب ألا تبقى أي
       وصفة اسمها يحوي garlic/ginger ضمن أفضل 15 بدرجة ذوق مرتفعة
       (القديمة 0.758 → الجديدة أقل من 0.5 أو خرجت بالكامل)، وبنص
       موجب "garlic, ginger, soy sauce, chicken; dislike dairy"
       تبقى وصفات الثوم/الزنجبيل/الدجاج بدرجات ذوق مرتفعة.

التشغيل (من داخل مجلد TOPSIS):
    python sanity_check.py
"""

import re
import sys
from pathlib import Path

# ── إضافة مجلد "Expert System" لمسار البحث ──────────────────
EXPERT_SYSTEM_DIR = Path(__file__).resolve().parent.parent / "Expert System"
sys.path.insert(0, str(EXPERT_SYSTEM_DIR))

import numpy as np
import pandas as pd

from main import load_data
from engine.filtering_engine import DietaryExpertSystem
from core.user_profile import UserProfile

from topsis_model import rank_with_topsis, _normalize_minmax
from ml.word2vec import taste_inference

TASTE_TEXT = "I love garlic, olive oil, and chicken. I dislike seafood."
NONSENSE_TEXT = "asdkjfhaskdjfh"

# ملف تعريف انحدار الكره — نفس شكل الحالة الواقعية التي أظهرت الخطأ
DISLIKE_REGRESSION_TEXT = ("chocolate, vanilla, cinnamon, sweet desserts; "
                           "dislike garlic and ginger")
POSITIVE_REGRESSION_TEXT = "garlic, ginger, soy sauce, chicken; dislike dairy"
REGRESSION_PROFILE = UserProfile(
    age=25, height=165, weight=55, gender="female",
    conditions=[], goal="healthy_eating",
)
GARLIC_GINGER_NAME_RE = re.compile(r"garlic|ginger", re.IGNORECASE)
GARLIC_GINGER_CHICKEN_NAME_RE = re.compile(r"garlic|ginger|chicken", re.IGNORECASE)


PROFILES = [
    ("diabetic", UserProfile(
        age=45, height=172, weight=88, gender="male",
        conditions=["diabetes", "hypertension"], goal="weight_loss",
    )),
    ("muscle_gain", UserProfile(
        age=28, height=180, weight=75, gender="male",
        conditions=[], goal="muscle_gain",
    )),
    ("elderly", UserProfile(
        age=70, height=165, weight=70, gender="female",
        conditions=["hypertension", "osteoporosis"], goal="heart_health",
    )),
]


def check_profile(name, profile, system, all_ok):
    print(f"\n--- Profile: {name} ---")
    result = system.filter_recipes(profile)
    safe = result["safe_recipes"]
    print(f"  safe={result['total_safe']} | original={result['total_original']}")

    required = ["_expert_score", "_ai_health_score"]
    missing = [c for c in required if c not in safe.columns]
    if missing:
        print(f"  FAIL: missing columns {missing}")
        return False

    for col in required:
        vals = safe[col].astype(float)
        n_nan = int(vals.isna().sum())
        print(f"  {col}: n={len(vals)}, NaN={n_nan}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")
        if n_nan:
            all_ok = False

    ranked = rank_with_topsis(safe, profile.goal)
    if np.isnan(ranked["final_score"].astype(float)).any():
        print("  FAIL: NaN in final_score")
        all_ok = False

    in_bounds = bool(((ranked["final_score"].astype(float) >= -1e-6)
                      & (ranked["final_score"].astype(float) <= 1 + 1e-6)).all())
    print(f"  final_score within [0,1]: {in_bounds}")
    if not in_bounds:
        all_ok = False

    old_top = ranked.sort_values("_topsis_score", ascending=False).head(5)
    new_top = ranked.head(5)

    print("  Old TOPSIS-only top-5:")
    for i, (_, r) in enumerate(old_top.iterrows(), 1):
        nm = str(r.get("Name", "Unknown"))
        nm = (nm[:43] + "..") if len(nm) > 45 else nm.ljust(45)
        print(f"    {i}. {nm} topsis={r['_topsis_score']:.4f}")

    print("  New blended top-5:")
    for i, (_, r) in enumerate(new_top.iterrows(), 1):
        nm = str(r.get("Name", "Unknown"))
        nm = (nm[:43] + "..") if len(nm) > 45 else nm.ljust(45)
        print(f"    {i}. {nm} final={r['final_score']:.4f} "
              f"(topsis={r['_topsis_score']:.4f}, ai={r['_ai_health_score']:.4f})")

    overlap = len(set(old_top.index).intersection(set(new_top.index)))
    print(f"  Overlap old/new top-5: {overlap}/5")

    all_inside = set(ranked.index).issubset(set(safe.index))
    print(f"  Blended ranking within safe set only: {all_inside}")
    if not all_inside:
        all_ok = False

    return all_ok


def check_taste(system, all_ok):
    """تحقق من تكامل درجة الذوق: توافق رجعي + ذوق + نص nonsense."""
    print("\n--- Taste score integration (diabetic profile) ---")
    profile = dict(PROFILES)["diabetic"]
    result = system.filter_recipes(profile)
    safe = result["safe_recipes"]
    print(f"  safe={result['total_safe']}")

    # 1) بدون نص ذوق → توافق رجعي كامل مع 0.4/0.4/0.2
    base = rank_with_topsis(safe, profile.goal)
    has_taste_col = "_taste_score" in base.columns
    print(f"  No-taste run: _taste_score column absent: {not has_taste_col}")
    if has_taste_col:
        all_ok = False

    expected = (0.4 * base["_topsis_score"].astype(float)
                + 0.4 * base["_ai_health_score"].astype(float)
                + 0.2 * _normalize_minmax(base["_expert_score"].astype(float)))
    match = bool(np.allclose(base["final_score"].astype(float), expected, atol=1e-10))
    print(f"  Backward compat: final_score == 0.4*TOPSIS+0.4*AI+0.2*exp: {match}")
    if not match:
        all_ok = False

    # 2) بنص ذوق حقيقي
    ranked = rank_with_topsis(safe, profile.goal, user_taste_text=TASTE_TEXT)
    missing = [c for c in ("_taste_score", "final_score") if c not in ranked.columns]
    if missing:
        print(f"  FAIL: missing columns {missing}")
        return False

    t = ranked["_taste_score"].astype(float)
    n_nan = int(t.isna().sum())
    in_bounds = bool(((t >= -1e-6) & (t <= 1 + 1e-6)).all())
    finals_ok = bool(((ranked["final_score"].astype(float) >= -1e-6)
                      & (ranked["final_score"].astype(float) <= 1 + 1e-6)).all())
    print(f"  Taste run: n={len(t)}, NaN={n_nan}, "
          f"range=[{t.min():.4f}, {t.max():.4f}], in [0,1]: {in_bounds}")
    print(f"  Taste run: final_score in [0,1]: {finals_ok}")
    if n_nan or not in_bounds or not finals_ok:
        all_ok = False

    print("  Top-5 with taste:")
    for i, (_, r) in enumerate(ranked.head(5).iterrows(), 1):
        nm = str(r.get("Name", "Unknown"))
        nm = (nm[:40] + "..") if len(nm) > 42 else nm.ljust(42)
        print(f"    {i}. {nm} final={r['final_score']:.4f} "
              f"(topsis={r['_topsis_score']:.4f}, ai={r['_ai_health_score']:.4f}, "
              f"taste={r['_taste_score']:.4f})")

    top5_base = set(base.head(5)["RecipeId"])
    top5_taste = set(ranked.head(5)["RecipeId"])
    changed = top5_base != top5_taste
    print(f"  Top-5 changed vs no-taste run: {changed}")
    if not changed:
        all_ok = False

    # 3) نص nonsense → درجات محايدة 0.5 دون أخطاء
    nonsense = rank_with_topsis(safe, profile.goal, user_taste_text=NONSENSE_TEXT)
    nt = nonsense["_taste_score"].astype(float)
    all_neutral = bool((nt == 0.5).all())
    print(f"  Nonsense text: all taste scores neutral 0.5: {all_neutral}")
    if not all_neutral:
        all_ok = False

    n_nonsense = int(nonsense["final_score"].astype(float).isna().sum())
    print(f"  Nonsense text: NaN in final_score: {n_nonsense}")
    if n_nonsense:
        all_ok = False

    return all_ok


def _old_logic_taste_score(recipe_id, text):
    """أعد إنتاج درجة الذوق بالمنطق القديم (buggy): كامل النص كجملة
    واحدة وكل المكونات تُعامل كمحبوبة — للمقارنة قبل/بعد فقط."""
    wv, embeddings = taste_inference._load()
    emb = embeddings.get(recipe_id)
    if emb is None:
        return None
    tokens = taste_inference._tokenize_text(text)
    matched = [wv[t] for t in tokens if t in wv]
    if not matched:
        return None
    old_vec = np.mean(np.vstack(matched), axis=0)
    old_vec = old_vec / np.linalg.norm(old_vec)
    emb = np.asarray(emb, dtype=float)
    emb = emb / np.linalg.norm(emb)
    return (float(np.dot(old_vec, emb)) + 1.0) / 2.0


def check_taste_concepts(safe, goal, all_ok):
    """فحوصات المفاهيم + fuzzy (خفيفة): بلا NaN وداخل [0,1] + طباعة أفضل 5."""
    print("\n--- Taste concepts + fuzzy: combined smoke checks "
          "(25yo female, healthy_eating) ---")
    for label, text in [
        ("concept-only", "I love spicy and sweet food"),
        ("typos+concept+dislike", "garlik and chiken, but dislike spicy food"),
    ]:
        ranked = rank_with_topsis(safe, goal, user_taste_text=text)
        t = ranked["_taste_score"].astype(float)
        n_nan = int(t.isna().sum())
        bounds = bool(((ranked["final_score"].astype(float) >= -1e-6)
                       & (ranked["final_score"].astype(float) <= 1 + 1e-6)).all())
        print(f"  [{label}] text={text!r}")
        print(f"    n={len(t)}, NaN={n_nan}, taste range=[{t.min():.4f}, "
              f"{t.max():.4f}], final in [0,1]: {bounds}")
        if n_nan or not bounds:
            all_ok = False
        print("    Top-5:")
        for i, (_, r) in enumerate(ranked.head(5).iterrows(), 1):
            nm = str(r.get("Name", "Unknown"))
            nm = (nm[:42] + "..") if len(nm) > 44 else nm.ljust(44)
            print(f"      {i}. {nm} final={r['final_score']:.4f} "
                  f"taste={r['_taste_score']:.4f}")
    return all_ok


def check_taste_regression(system, all_ok):
    """انحدار معالجة الكره: الوصفات المكروهة يجب ألا تُسجَّل كمحبوبة."""
    print("\n--- Taste regression: dislike parsing (25yo female, healthy_eating) ---")
    result = system.filter_recipes(REGRESSION_PROFILE)
    safe = result["safe_recipes"]
    print(f"  safe={result['total_safe']}")

    # 1) "dislike garlic and ginger" → وصفات garlic/ginger بدرجات منخفضة
    ranked = rank_with_topsis(safe, REGRESSION_PROFILE.goal,
                              user_taste_text=DISLIKE_REGRESSION_TEXT)
    t = ranked["_taste_score"].astype(float)
    finals_ok = bool(((ranked["final_score"].astype(float) >= -1e-6)
                      & (ranked["final_score"].astype(float) <= 1 + 1e-6)).all())
    print(f"  Dislike run: n={len(t)}, NaN={int(t.isna().sum())}, "
          f"taste range=[{t.min():.4f}, {t.max():.4f}], final in [0,1]: {finals_ok}")
    if int(t.isna().sum()) or not finals_ok:
        all_ok = False

    top15 = ranked.head(15)
    hits = top15[top15["Name"].astype(str).str.contains(GARLIC_GINGER_NAME_RE)]
    if hits.empty:
        print("  PASS (stronger): NO garlic/ginger-named recipe in top-15 — "
              "disliked recipes pushed out entirely")
    else:
        for _, r in hits.iterrows():
            nm = str(r["Name"])[:48]
            print(f"    {nm:<50} taste={r['_taste_score']:.4f} "
                  f"final={r['final_score']:.4f}")
        max_taste = float(hits["_taste_score"].max())
        low = max_taste < 0.5
        print(f"  max taste among garlic/ginger-named in top-15: "
              f"{max_taste:.4f} (< 0.5: {low})")
        if not low:
            all_ok = False

        r = hits.iloc[0]
        old = _old_logic_taste_score(int(r["RecipeId"]), DISLIKE_REGRESSION_TEXT)
        print(f"  before/after for '{str(r['Name'])[:48]}':")
        print(f"    old (buggy avg-all) taste = {old:.4f}  |  "
              f"new (dislike-aware) taste = {r['_taste_score']:.4f}")
        if old is not None and not r["_taste_score"] < old:
            all_ok = False

    # 2) انحدار موجب: "garlic, ginger, soy sauce, chicken" → تبقى مرتفعة
    ranked_pos = rank_with_topsis(safe, REGRESSION_PROFILE.goal,
                                  user_taste_text=POSITIVE_REGRESSION_TEXT)
    tp = ranked_pos["_taste_score"].astype(float)
    finals_pos_ok = bool(((ranked_pos["final_score"].astype(float) >= -1e-6)
                          & (ranked_pos["final_score"].astype(float) <= 1 + 1e-6)).all())
    print(f"  Positive run: n={len(tp)}, NaN={int(tp.isna().sum())}, "
          f"final in [0,1]: {finals_pos_ok}")
    if int(tp.isna().sum()) or not finals_pos_ok:
        all_ok = False

    hits_pos = ranked_pos.head(15)[
        ranked_pos.head(15)["Name"].astype(str).str.contains(
            GARLIC_GINGER_CHICKEN_NAME_RE)]
    if hits_pos.empty:
        print("  FAIL: no garlic/ginger/chicken-named recipe in top-15 "
              "under the positive text")
        all_ok = False
    else:
        lo = float(hits_pos["_taste_score"].min())
        hi = float(hits_pos["_taste_score"].max())
        high = lo > 0.5
        print(f"  Positive run: {len(hits_pos)} garlic/ginger/chicken-named "
              f"in top-15, taste range=[{lo:.4f}, {hi:.4f}] (min > 0.5: {high})")
        if not high:
            all_ok = False

    all_ok = check_taste_concepts(safe, REGRESSION_PROFILE.goal, all_ok) and all_ok
    return all_ok


def main():
    print("=" * 100)
    print("SANITY CHECK — AI health + taste integration (Layer 1.5 + blended Layer 2)")
    print("=" * 100)

    df = load_data()
    system = DietaryExpertSystem(df)

    all_ok = True
    for name, profile in PROFILES:
        all_ok = check_profile(name, profile, system, all_ok) and all_ok

    all_ok = check_taste(system, all_ok) and all_ok
    all_ok = check_taste_regression(system, all_ok) and all_ok
    print("\n" + "=" * 100)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "ISSUES FOUND")
    print("=" * 100)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
