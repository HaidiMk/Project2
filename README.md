# Smart Dietary Advisor

Smart Dietary Advisor takes a user's health profile — age, height, weight,
medical conditions, allergies, and a dietary goal — plus an optional
free-text taste description ("I love garlic and spicy food, dislike
seafood"), and returns a ranked list of recipes that are medically safe,
matched to their goal, and matched to their taste. It's exposed to real
clients (a mobile app, and a small admin dashboard) through a Django REST
API.

## How it works

Every candidate recipe goes through the same four stages:

1. **Rule-based safety filtering** — hard medical, allergy, halal, and
   diet-preference rules throw out anything unsafe for this specific user.
2. **Goal-based ranking (TOPSIS)** — a multi-criteria ranking method scores
   what's left against the user's goal (weight loss, muscle gain, ...).
3. **A trained health-suitability model** — scores how well a recipe's
   nutrition fits the user's specific medical conditions.
4. **A trained taste model** — scores how well a recipe matches what the
   user said they like or dislike, using ingredient embeddings.

The final ranking blends all four: roughly 40% goal ranking, 40% health
score, 20% rule score when there's no taste text; 35/35/15/15 (adding taste)
when there is.

Recipes come from the Food.com dataset. After cleaning, the corpus has
**384,541 recipes**, each with full nutrition info and (where available)
photos.

## The two trained models, briefly

- **Health-suitability model** — a small neural network trained to predict,
  from a recipe's nutrition, how suitable it is for each of 22 medical
  conditions. It gives a graded, learned score on top of the hard rules,
  rather than a plain "safe/unsafe" flag. (An earlier version of this
  project used a weaker collaborative-filtering model here instead; it was
  replaced once this model showed clearly better, more defensible results.)
- **Taste model** — a Word2Vec model trained on the recipe corpus's own
  ingredients, so ingredients that tend to appear together end up "close"
  to each other. A user's free-text taste description gets turned into a
  vector the same way, and recipes are scored by how close they land to it.

Both live under `Expert System/ml/` — see that folder's results files for
the actual training numbers.

## Smart search

`Nlp/` is a search-bar convenience layer, not a third trained model. It uses
a general-purpose pretrained sentence embedding model to turn a query like
"low sugar dinner for diabetic" into structured filters, then hands those
filters to the same safety-filtering engine everything else uses. It makes
no safety or ranking decisions of its own — it's just a nicer way to type a
request.

## A couple of extra pieces

- **Meal planner** — builds a full day or week of breakfast/lunch/dinner
  out of the same safe, ranked recipes, with no repeats and within a
  calorie target. It's live over the API (`meal-planner/` below).
- **Alternative-recipe suggestions** — a taste-similarity-based module that
  suggests safe substitutes for a blocked recipe. It exists and is tested,
  but the live `alternatives/` endpoint currently uses a simpler approach
  (see below) rather than this module.

## Backend REST API

Django + Django REST Framework, using token authentication. Three apps:
`accounts`, `profiles`, `recipes`. Recipe data itself isn't stored in the
database — it lives in memory, loaded from the dataset — only accounts and
health profiles are.

**Accounts**

| Endpoint | Method | Auth | What it does |
|---|---|---|---|
| `/api/accounts/register/` | POST | none | Create an account, get a token back immediately |
| `/api/accounts/login/` | POST | none | Log in with username or email + password |
| `/api/accounts/logout/` | POST | token | Invalidate the current token |

**Profile** (`/api/profiles/me/` — always your own profile)

| Method | What it does |
|---|---|
| GET | Get your profile |
| POST | Create it (first time only) |
| PUT | Replace it |
| PATCH | Update part of it |

**Recipes** (all require a token and a completed profile)

| Endpoint | Method | What it does |
|---|---|---|
| `/api/recipes/recommendations/` | GET | Ranked recipes for your profile |
| `/api/recipes/search/` | GET | Free-text search, same ranking underneath |
| `/api/recipes/<id>/` | GET | Full detail for one recipe, plus whether it's safe for you |
| `/api/recipes/<id>/alternatives/` | GET | Up to 3 alternatives to a given recipe |
| `/api/recipes/<id>/explanation/` | GET | Plain-language reasons a recipe suits your conditions |
| `/api/recipes/meal-planner/` | GET | A daily or weekly meal plan (`?type=daily` or `?type=weekly`) |
| `/api/recipes/dashboard/stats/` | GET | System-wide stats for the admin dashboard (no login required) |

## Admin dashboard

`dashboard/` is a small, separate React app for an admin/viewer to check
system-wide stats — total recipes, users, supported conditions/allergies,
and how well the health model is performing. It's live and working against
the real backend. See `dashboard/README.md` for its own details.

## Repository layout

```
Smart-Dietary-Advisor/
├── Expert System/     # rule engine, both trained models, meal planner
├── TOPSIS/            # goal-based ranking + the regression test suite
├── Nlp/               # smart search
├── Backend/           # the Django REST API
└── dashboard/         # the admin dashboard
```

See `Expert System/README.md` and `dashboard/README.md` for those folders'
own layout in more detail.

## Getting it running

**Expert System (standalone CLI):**
```bash
cd "Expert System"
pip install pandas numpy torch scikit-learn
python main.py --demo
```

**Backend (the API):**
```bash
cd Backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

**Dashboard:**
```bash
cd dashboard
npm install
cp .env.example .env
npm run dev
```

## Known limitations

- Taste-text search only understands English.
- Recommendations are limited to the 384,541 recipes in the current
  dataset — nothing outside it exists to recommend.
- Smart search needs an internet connection the first time it runs, to
  download its pretrained model.
- The Django backend is currently set up for local development, not
  production (debug mode on, open CORS, SQLite).
- Only the `recipes` app has an automated test suite; `accounts` and
  `profiles` are checked manually. The two newest endpoints
  (`meal-planner/`, `dashboard/stats/`) don't have automated tests yet
  either.
- The dashboard's `recommendations_count` and `search_count` stats stay
  empty — the project doesn't track usage history yet.
- `suggest_alternatives()` (the taste-based alternatives module) isn't
  wired into the live `alternatives/` endpoint yet — that endpoint uses a
  simpler ranking-based approach instead.

## How to verify it works

```bash
# full pipeline regression suite (~15-20 min)
cd TOPSIS
python sanity_check.py

# smart-search checks
python Nlp/test_query_parser.py
python Nlp/test_robustness_all.py

# standalone module smoke tests
python "Expert System/ml/word2vec/test_alternatives.py"
python "Expert System/planner/test_meal_planner.py"

# backend API tests
cd Backend
python manage.py test recipes
```

Everything above is expected to pass. See `Expert System/ml/setup_artifacts.md`
if you need to regenerate any of the trained model files from scratch.

## More detail

- `Expert System/README.md` — the rule engine and trained models
- `dashboard/README.md` — the admin dashboard
