# Smart Dietary Advisor — Dashboard

A small, read-only React dashboard for an admin/viewer to check on the
system at a glance — total recipes, users, supported conditions and
allergies, and how well the health-scoring model is performing.

This is a separate project from `Backend/`, `Expert System/`, `TOPSIS/`, and
`Nlp/` — it only calls the Django API over HTTP and never touches backend
code or data directly.

## Scope

This dashboard is for an admin/viewer, not the everyday end user. It doesn't
use the mobile app's login flow, doesn't need a personal health profile, and
has no recommendation, search, or meal-planning controls — those belong to
the end-user app. There's no user management or recipe management here,
just statistics.

## What it shows

- Stat cards: total recipes, total users, completed profiles, supported
  goals, supported conditions, supported allergies.
- A toggle to view those as numbers, a bar chart, or both.
- A "Health Classifier Performance" section (accuracy/precision/recall/F1)
  for the model that scores recipes against medical conditions.

Every number comes straight from the backend — nothing is hard-coded or
faked. If the backend is unreachable, the page shows a clear "unavailable"
message instead of guessing.

## How it connects to the backend

The dashboard calls `GET /api/recipes/dashboard/stats/` on the Django
backend. That endpoint is live and working, and currently doesn't require
logging in. Two of the numbers it returns — `recommendations_count` and
`search_count` — are still always `null`, since the project doesn't track
recommendation/search history yet.

## Setup

```bash
cd dashboard
npm install
```

Then create your local `.env` file from the template:

- **Windows (PowerShell):** `Copy-Item .env.example .env`
- **Windows (cmd):** `copy .env.example .env`
- **macOS/Linux:** `cp .env.example .env`

`VITE_API_BASE_URL` in `.env` controls which backend the dashboard talks to
(default `http://localhost:8000`). Then:

```bash
npm run dev
```

Open the printed local URL (default `http://localhost:5173`). Make sure the
Django backend (see `Backend/`) is running first, or you'll see the
"unavailable" state.

## Build

```bash
npm run build
npm run preview
```

## Project layout

```
dashboard/
├── src/
│   ├── App.jsx                     ← main page layout
│   ├── components/
│   │   ├── Header.jsx
│   │   ├── StatCard.jsx
│   │   ├── StatusMessage.jsx
│   │   ├── ViewModeSelector.jsx    ← numbers/chart/both toggle
│   │   └── OverviewChart.jsx       ← bar chart
│   ├── hooks/useDashboardData.js   ← fetches + tracks loading/error state
│   ├── services/api.js             ← the actual fetch call
│   ├── config/api.js               ← API base URL + endpoint path
│   └── utils/format.js             ← number/percent display formatting
└── .env.example
```

## More detail

For the full picture of how the backend and the rest of the project fit
together, see the `README.md` at the repository root.
