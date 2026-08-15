// Admin/Viewer Dashboard header. No session/login UI here by design —
// this Dashboard opens directly (see App.jsx).
export default function Header() {
  return (
    <header className="dashboard-header">
      <div>
        <h1>Smart Dietary Advisor Dashboard</h1>
        <p className="muted">System Statistics &amp; Recommendation Results</p>
      </div>
    </header>
  );
}
