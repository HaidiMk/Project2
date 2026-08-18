import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LabelList,
} from "recharts";
import { formatStatValue } from "../utils/format.js";

function hasValue(value) {
  return value !== null && value !== undefined;
}

// Chart mode for System Overview: renders only the four real, always-available
// API values (total_recipes, total_users, completed_profiles, supported_goals).
// Values are shown exactly as returned — no scaling, normalizing, or inventing —
// the exact number is always labeled on/above its bar since Total Recipes can
// dwarf the other counts.
export default function OverviewChart({ stats }) {
  const chartData = [
    { label: "Total Recipes", value: stats?.total_recipes },
    { label: "Total Users", value: stats?.total_users },
    { label: "Completed Profiles", value: stats?.completed_profiles },
    { label: "Supported Goals", value: stats?.supported_goals },
  ].filter((item) => hasValue(item.value));

  if (chartData.length === 0) {
    return (
      <div className="card chart-card">
        <p className="muted">No overview data is currently available.</p>
      </div>
    );
  }

  return (
    <div className="card chart-card">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} margin={{ top: 24, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" interval={0} />
          <YAxis allowDecimals={false} />
          <Tooltip formatter={(value) => formatStatValue(value)} />
          <Bar dataKey="value" fill="#2e7d32" radius={[4, 4, 0, 0]}>
            <LabelList dataKey="value" position="top" formatter={(value) => formatStatValue(value)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
