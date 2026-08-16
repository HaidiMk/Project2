function fmt(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(2);
  return value;
}

export default function RecommendationsTable({ results }) {
  if (!results || results.length === 0) {
    return <p className="muted">No results available yet.</p>;
  }

  return (
    <div className="table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Final Score</th>
            <th>Calories</th>
            <th>Protein</th>
            <th>AI Health Score</th>
            <th>Expert System Score</th>
          </tr>
        </thead>
        <tbody>
          {results.map((row, i) => (
            <tr key={row.name ? `${row.name}-${i}` : i}>
              <td>{row.name}</td>
              <td>{fmt(row.final_score)}</td>
              <td>{fmt(row.calories)}</td>
              <td>{fmt(row.protein)}</td>
              <td>{fmt(row.ai_health_score)}</td>
              <td>{fmt(row.expert_score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
