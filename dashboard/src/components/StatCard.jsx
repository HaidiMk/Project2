// Renders "—" for null/undefined so unavailable backend data reads as a
// neutral placeholder, never a fabricated number (0 is a real value and is
// displayed as-is; only "no data yet" collapses to the dash).
export default function StatCard({ label, value }) {
  const display = value === null || value === undefined ? "—" : value;
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{display}</div>
    </div>
  );
}
