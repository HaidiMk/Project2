export function formatStatValue(value, format = "number") {
  if (value === null || value === undefined) return "—";

  if (format === "percent") {
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    return `${(value * 100).toFixed(2)}%`;
  }

  if (typeof value === "number") {
    return Number.isNaN(value) ? "—" : value.toLocaleString("en-US");
  }

  return String(value);
}
