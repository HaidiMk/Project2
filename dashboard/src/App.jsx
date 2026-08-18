import { useState } from "react";
import { useDashboardData } from "./hooks/useDashboardData.js";
import Header from "./components/Header.jsx";
import StatCard from "./components/StatCard.jsx";
import StatusMessage from "./components/StatusMessage.jsx";
import ViewModeSelector from "./components/ViewModeSelector.jsx";
import OverviewChart from "./components/OverviewChart.jsx";

function hasValue(value) {
  return value !== null && value !== undefined;
}

export default function App() {
  const { data, loading, unavailable } = useDashboardData();
  const [viewMode, setViewMode] = useState("numbers");
  const stats = data?.stats;
  const healthMetrics = data?.model_metrics?.health_classifier;

  const hasSupportedConditions = hasValue(stats?.supported_conditions);
  const hasSupportedAllergies = hasValue(stats?.supported_allergies);
  const hasAdditionalStats = hasSupportedConditions || hasSupportedAllergies;

  const showNumbers = viewMode === "numbers" || viewMode === "both";
  const showChart = viewMode === "chart" || viewMode === "both";

  const hasHealthMetrics =
    healthMetrics &&
    [
      healthMetrics.accuracy,
      healthMetrics.precision,
      healthMetrics.recall,
      healthMetrics.f1_score,
    ].some(hasValue);

  return (
    <div className="dashboard-shell">
      <Header />
      <main className="dashboard-main">
        {loading && <StatusMessage type="waiting">Loading dashboard data...</StatusMessage>}
        {!loading && unavailable && (
          <StatusMessage type="error">Dashboard data is currently unavailable.</StatusMessage>
        )}

        {!loading && !unavailable && (
          <>
            <section>
              <div className="section-header">
                <h2>System Overview</h2>
                <ViewModeSelector value={viewMode} onChange={setViewMode} />
              </div>

              {showNumbers && (
                <>
                  <div className="stats-grid">
                    <StatCard label="Total Recipes" value={stats?.total_recipes} />
                    <StatCard label="Total Users" value={stats?.total_users} />
                    <StatCard label="Completed Profiles" value={stats?.completed_profiles} />
                    <StatCard label="Supported Goals" value={stats?.supported_goals} />
                  </div>
                  {hasAdditionalStats && (
                    <div className="stats-grid stats-grid-secondary">
                      {hasSupportedConditions && (
                        <StatCard label="Supported Conditions" value={stats?.supported_conditions} />
                      )}
                      {hasSupportedAllergies && (
                        <StatCard label="Supported Allergies" value={stats?.supported_allergies} />
                      )}
                    </div>
                  )}
                </>
              )}

              {showChart && (
                <div className="overview-chart-wrap">
                  <OverviewChart stats={stats} />
                </div>
              )}
            </section>

            {hasHealthMetrics && (
              <section>
                <h2>Health Classifier Performance</h2>
                <div className="stats-grid">
                  <StatCard label="Accuracy" value={healthMetrics?.accuracy} format="percent" />
                  <StatCard label="Precision" value={healthMetrics?.precision} format="percent" />
                  <StatCard label="Recall" value={healthMetrics?.recall} format="percent" />
                  <StatCard label="F1 Score" value={healthMetrics?.f1_score} format="percent" />
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}
