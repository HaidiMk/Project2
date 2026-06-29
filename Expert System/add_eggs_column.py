"""
add_eggs_column.py — إضافة عمود HasEggs لملف cleaned_recipes.csv الموجود
==========================================================================
سكريبت لمرة واحدة فقط. يضيف عمود "HasEggs" (True/False) لكل وصفة بناءً
على فحص نصي لعمود IngredientsList، بنفس أسلوب الأعمدة الموجودة أصلاً
(HasLactose, HasGluten, HasNuts, HasSoy, HasSeafood) المبنية بـ cleaner.py.

⚠️ ملاحظة فنية مهمة (مكتشَفة بالفحص الفعلي على بياناتك):
   IngredientsList بملف الـ CSV محفوظ كنص خام (str) يمثّل قائمة، مثل:
       "['blueberries', 'granulated sugar', 'vanilla yogurt']"
   وليس كقائمة بايثون فعلية (list) — لأن CSV لا يحفظ أنواع بايثون،
   فقط نصوص. لذلك هذا السكريبت يفحص الكلمات المفتاحية على مستوى
   النص الخام مباشرة (substring check)، وهذا متوافق تماماً مع كيفية
   تعامل filtering_engine.py مع نفس العمود لاحقاً عند الفلترة.

طريقة الاستخدام:
    1. ضع هذا الملف داخل مجلد "Expert System" (نفس مستوى main.py)
    2. شغّله مرة واحدة:
           python add_eggs_column.py
    3. سيقرأ data/cleaned_recipes.csv، يضيف العمود، ويحفظ نسخة جديدة
       (لا يَكتب فوق الملف الأصلي مباشرة، حماية من فقدان البيانات).
    4. بعد التحقق من النتيجة، تستبدل الملف القديم بالجديد يدوياً.
"""

from pathlib import Path
import pandas as pd

# ── نفس قائمة المصطلحات بأسلوب باقي الحساسيات (LACTOSE, GLUTEN...) ──
EGGS = {"egg", "eggs", "egg white", "egg yolk", "mayonnaise", "meringue", "albumin"}


def has_egg_trigger(raw_text) -> bool:
    """
    يفحص إذا النص الخام لعمود IngredientsList يحتوي أي كلمة من EGGS.
    يتعامل بأمان مع NaN/None ومع الحالة النادرة (list حقيقية محمَّلة
    مسبقاً عبر ast.literal_eval إن وُجد مستقبلاً).
    """
    is_missing = raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text))
    text = "" if is_missing else (
        " ".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
    )
    text_lower = text.lower()
    return any(term in text_lower for term in EGGS)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    input_path = data_dir / "cleaned_recipes.csv"
    output_path = data_dir / "cleaned_recipes_with_eggs.csv"

    print(f"📥  Reading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"✅  Loaded {len(df):,} recipes")

    if "IngredientsList" not in df.columns:
        print("❌  Column 'IngredientsList' not found — aborting.")
        return

    print("⚕️  Computing HasEggs column...")
    df["HasEggs"] = df["IngredientsList"].apply(has_egg_trigger)

    egg_count = int(df["HasEggs"].sum())
    pct = round(egg_count / len(df) * 100, 1)
    print(f"   Found eggs in {egg_count:,} recipes ({pct}%)")

    print(f"💾  Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("✅  Done.")
    print()
    print("الخطوة التالية اليدوية:")
    print(f"  1. تحقق من الملف الجديد: {output_path}")
    print(f"  2. لو كل شي تمام، استبدل الملف القديم:")
    print(f"       احذف: {input_path}")
    print(f"       أعد تسمية: {output_path} → cleaned_recipes.csv")


if __name__ == "__main__":
    main()
