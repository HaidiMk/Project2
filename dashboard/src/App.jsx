import { useDashboardData } from "./hooks/useDashboardData.js";
import Header from "./components/Header.jsx";
import StatCard from "./components/StatCard.jsx";
import RecommendationsTable from "./components/RecommendationsTable.jsx";
import RecommendationChart from "./components/RecommendationChart.jsx";
import StatusMessage from "./components/StatusMessage.jsx";

// Admin/Viewer Dashboard (UC-11). Opens directly — no normal-user login,
// no UserProfile dependency. Reads exclusively from the future Django
// Dashboard API (src/services/api.js); until that endpoint exists, every
// section renders its documented neutral empty state instead of fake data.
export default function App() {
  const { data, loading, unavailable } = useDashboardData();
  const stats = data?.stats;

  return (
    <div className="dashboard-shell">
      <Header />
      <main className="dashboard-main">
        {loading && <StatusMessage type="waiting">Loading dashboard data…</StatusMessage>}
        {!loading && unavailable && (
          <StatusMessage type="waiting">
            Waiting for backend Dashboard API. Statistics and results below are
            placeholders until the backend team implements the Dashboard endpoint.
          </StatusMessage>
        )}

        <section className="stats-grid">
          <StatCard label="Total Recipes" value={stats?.total_recipes ?? null} />
          <StatCard label="Supported Medical Conditions" value={stats?.supported_conditions ?? null} />
          <StatCard label="Supported Allergies" value={stats?.supported_allergies ?? null} />
          <StatCard label="Recommendations" value={stats?.recommendations_count ?? null} />
        </section>

        <section>
          <h2>System / Recommendation Results</h2>
          <RecommendationsTable results={data?.results} />
        </section>

        <RecommendationChart chartData={data?.chart_data} />
      </main>
    </div>
  );
}
