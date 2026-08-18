// Centralized display formatting for dashboard values.
//
// Rules:
// - null / undefined  -> "—"
// - 0                  -> "0"   (never treat 0 as "missing" via `value || fallback`)
// - large integers      -> thousands separators, e.g. 384541 -> "384,541"
// - format="percent"    -> decimal fraction (e.g. 0.992684) -> "99.27%"

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
