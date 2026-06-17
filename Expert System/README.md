````markdown
# Smart Dietary Advisor — v4.0
## نظام التغذية الذكي — Expert System

قسم الذكاء الاصطناعي — مشروع السنة الرابعة 2024-2025

---

## نبذة عن المشروع

نظام ذكاء اصطناعي متكامل يجمع بين ثلاث تقنيات:
- **Expert System** — قواعد طبية صارمة (24 حالة، 40+ تركيبة دمج)
- **Content-Based Filtering** — تصفية وترتيب بناءً على القيم الغذائية
- **Neural Collaborative Filtering (NCF)** — توصيات مخصصة بالشبكات العصبونية

قاعدة البيانات: **384,541 وصفة** مع قيمها الغذائية الكاملة

---

## متطلبات التشغيل

```bash
pip install pandas numpy torch scikit-learn
```

---

## تشغيل النظام

```bash
# واجهة تفاعلية
python main.py

# مثال جاهز (سكري + ضغط دم)
python main.py --demo

# تحليل البيانات EDA
python eda_report.py

# اختبار شامل للنظام
python test_system.py
```

---

## هيكل الملفات

```
expert_split/
│
├── main.py                        ← نقطة الدخول
├── eda_report.py                  ← تحليل البيانات
├── test_system.py                 ← اختبارات النظام (16/16)
│
├── core/
│   ├── constants.py               ← الثوابت والقواميس
│   └── user_profile.py            ← كلاس UserProfile (BMI, BMR, السعرات)
│
├── rules/
│   ├── medical_rules.py           ← القواعد الطبية (24 حالة)
│   ├── combined_rules.py          ← قواعد الدمج (40+ تركيبة)
│   ├── halal_and_allergies.py     ← فلتر الحلال + الحساسيات
│   └── goals_and_preferences.py   ← الأهداف والتفضيلات
│
├── engine/
│   ├── rule_builder.py            ← تجميع القواعد
│   ├── scorer.py                  ← تقييم الوصفات
│   ├── filtering_engine.py        ← المحرك الرئيسي
│   └── ncf_model.py               ← نموذج NCF (PyTorch)
│
├── ui/
│   └── cli_interface.py           ← واجهة سطر الأوامر
│
├── data/
│   ├── cleaned_recipes.csv        ← قاعدة البيانات (384,541 وصفة)
│   └── ncf_model.pt               ← النموذج المدرب
│
└── output/
    ├── safe_recipes.csv
    ├── safe_recipes.json
    └── eda_report.txt
```

---

## المستخدمون المدعومون

| الفئة | العمر | الأمراض المتاحة | ملاحظة |
|-------|-------|-----------------|--------|
| Child | 1-12 | 7 أمراض | بدون سؤال النشاط البدني |
| Teen | 13-17 | 11 مرض | |
| Adult | 18-64 | 21 مرض + PCOS للإناث | |
| Elderly | 65+ | 16 مرض | |

---

## الحالات الطبية المدعومة (24 حالة)

| المجموعة | الحالات |
|----------|---------|
| القلب والأوعية | Hypertension, Heart Disease, High Cholesterol |
| الغدد | Diabetes, Hypothyroidism, Hyperthyroidism, PCOS |
| الجهاز الهضمي | GERD, IBS, Crohn's Disease, Constipation, Hepatitis |
| الكلى والمفاصل | Chronic Kidney Disease, Gout, Osteoporosis |
| الدم والوزن | Anemia, Obesity, Underweight |
| الحساسيات | Gluten Intolerance, Lactose Intolerance |
| خاص | Pregnancy, Asthma |

---

## الحساسيات المدعومة (6 أنواع)

Peanuts / Tree Nuts, Milk and Dairy, Eggs, Seafood, Soy, Gluten

---

## نتائج EDA — تحليل البيانات

| المقياس | القيمة |
|---------|--------|
| إجمالي الوصفات | 384,541 |
| متوسط السعرات | 351 kcal/وجبة |
| متوسط التقييم | 4.63 / 5.0 |
| وصفات آمنة للسكري | 66.6% |
| وصفات آمنة لضغط الدم | 71.3% |
| وصفات تحتوي لاكتوز | 59.5% |
| قيم مفقودة | 0% |

---

## طبقات نظام الخبير

```
المستخدم يدخل بياناته
        ↓
1. فلتر الحلال          ← أولوية مطلقة
        ↓
2. فلتر الحساسيات       ← أعمدة CSV
        ↓
3. الحدود الرقمية       ← القواعد الطبية
        ↓
4. فلتر نوع الوجبة      ← فطور/غداء/عشاء
        ↓
5. فلتر المكونات        ← strict_block
        ↓
6. فلتر اسم الوصفة      ← Name column
        ↓
7. Expert Score         ← scorer.py (70%)
   +
   NCF Score            ← ncf_model.py (30%)
        ↓
النتائج الآمنة المرتبة
```

---

## نموذج NCF

```
Input:  القيم الغذائية + هدف المستخدم (10 ميزات)
        ↓
Layer 1: Linear(10 → 64) + ReLU + Dropout(0.2)
        ↓
Layer 2: Linear(64 → 32) + ReLU + Dropout(0.2)
        ↓
Layer 3: Linear(32 → 16) + ReLU
        ↓
Output: Linear(16 → 1) + Sigmoid × 5
        ↓
درجة توقعية (0 - 5)
```

التدريب: 5 epochs على 384,541 وصفة — Final Loss: 0.0329

---

## اختبارات النظام

```
✅ 16/16 Tests Passed
   - Diabetes rules
   - Hypertension rules
   - Halal filter
   - PCOS females only
   - Impossible combinations
   - Vegetarian filter
   - Pregnancy + Diabetes combined
   - Activity level affects calories
```

---

## المصادر الطبية

| المصدر | الحالة |
|--------|--------|
| ADA Standards of Care 2025 | Diabetes |
| AHA/ACC Guidelines 2025 | Heart Disease, Hypertension |
| WHO Guidelines 2023 | Hypertension, General |
| NKF/KDIGO Guidelines 2024 | Chronic Kidney Disease |
| ACG Clinical Guidelines 2022-2023 | GERD, IBS, Crohn's |
| ACOG Guidelines 2021 | Pregnancy |
| ESPEN Guidelines 2019 | Elderly Nutrition |
| NOF + AACE 2020 | Osteoporosis |
| ACR Guidelines 2020 | Gout |
| GINA 2024 | Asthma |
````
