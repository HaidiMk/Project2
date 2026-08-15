import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

/**
 * The one approved chart. Reads the future Dashboard API's `chart_data[]`
 * ({ label, value }[]). Shows a neutral empty state until real data exists —
 * never renders fabricated bars.
 */
export default function RecommendationChart({ chartData }) {
  if (!chartData || chartData.length === 0) {
    return (
      <div className="card chart-card">
        <h2>Chart</h2>
        <p className="muted">Chart data will appear when Dashboard API data is available.</p>
      </div>
    );
  }

  return (
    <div className="card chart-card">
      <h2>Chart</h2>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" angle={-30} textAnchor="end" interval={0} height={60} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#2e7d32" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
