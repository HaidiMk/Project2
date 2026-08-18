import { formatStatValue } from "../utils/format.js";

export default function StatCard({ label, value, format = "number" }) {
  const display = formatStatValue(value, format);
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{display}</div>
    </div>
  );
}
