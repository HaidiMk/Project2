export default function StatCard({ label, value }) {
  const display = value === null || value === undefined ? "—" : value;
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{display}</div>
    </div>
  );
}
