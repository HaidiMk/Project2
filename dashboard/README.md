# Smart Dietary Advisor — Dashboard

A standalone, read-only React dashboard implementing **UC-11 (View Dashboard)**:
Admin/Viewer → React Dashboard → Django Dashboard API → statistics and results.

This project is intentionally separate from `Backend/`, `Expert System/`, `TOPSIS/`,
and `Nlp/` — it only *consumes* a Django REST API over HTTP and never modifies
backend code, data, or behavior.

## Scope

This Dashboard is for an **Admin/Viewer**, not the normal end user. It does not
use the Flutter app's login flow, does not depend on a `UserProfile`, and has
no meal-type/taste-text/personal-recommendation controls — those belong to the
end-user application. There is no user management, recipe management, or
medical-condition management here.

## Backend status: awaiting implementation

The dedicated Django Dashboard API this project is built against **does not
exist yet**. The backend team will implement it separately. Until then:

- Stat cards display `—`
- The results section displays "No results available yet."
- The chart displays "Chart data will appear when Dashboard API data is available."
- A single banner reads "Waiting for backend Dashboard API."

No fake/sample numbers are hard-coded anywhere in this project. The Dashboard
is designed to run and be previewed even with no backend running at all — it
simply shows the neutral waiting/empty states above.

## Proposed future API contract

This is a **provisional frontend/backend integration contract only**. Every
`null`/empty value below means "value not supplied yet" — it is a shape
placeholder, not a real project statistic. The backend team may adjust field
names or structure when they implement the endpoint.

```json
{
  "stats": {
    "total_recipes": null,
    "supported_conditions": null,
    "supported_allergies": null,
    "recommendations_count": null
  },
  "results": [],
  "chart_data": []
}
```

`results[]` items and `chart_data[]` items are expected to eventually look like:

```json
{
  "name": "string",
  "final_score": null,
  "calories": null,
  "protein": null,
  "ai_health_score": null,
  "expert_score": null
}
```

```json
{ "label": "string", "value": null }
```

When the backend team confirms the final contract, update:

- `src/config/api.js` — `DASHBOARD_STATS_PATH` (and `API_BASE_URL` via `.env` if needed)
- `src/services/api.js` — `fetchDashboardData()` and the contract comment
- `src/components/RecommendationsTable.jsx` / `RecommendationChart.jsx` — field names read from `results[]` / `chart_data[]`, if they change

## Prerequisites

- **Node.js with npm** — required, to run this frontend at all.
- **Django backend running** (see `Backend/`) — only required once the backend
  team implements the Dashboard endpoint and you want to test real API
  integration. Not needed just to preview the Dashboard's layout and empty
  states.

## Setup

```bash
cd dashboard
npm install
```

Then create your local `.env` file from the template:

- **Windows (PowerShell):**
  ```powershell
  Copy-Item .env.example .env
  ```
- **Windows (Command Prompt):**
  ```cmd
  copy .env.example .env
  ```
- **macOS/Linux:**
  ```bash
  cp .env.example .env
  ```

Or simply duplicate `.env.example` in File Explorer and rename the copy to `.env`.

`VITE_API_BASE_URL` in `.env` only matters once a real Dashboard API exists to
point at (default `http://localhost:8000`). Then run:

```bash
npm run dev
```

Open the printed local URL (default `http://localhost:5173`).

## Build

```bash
npm run build
npm run preview
```

## Configuration

The Django API base URL and Dashboard endpoint path are set once, in
`src/config/api.js` (base URL from `.env`'s `VITE_API_BASE_URL`). No component
hard-codes a URL.
