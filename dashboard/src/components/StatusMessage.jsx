/**
 * Shared status presentation. Never renders raw stack traces — only clean,
 * human-readable text passed in by the caller.
 *
 * type="waiting" — neutral state used while the future Dashboard API is
 *   unimplemented or loading (not an error; expected during this phase).
 * type="error"   — a real request failure once the API exists.
 * type="info"    — general informational text.
 */
export default function StatusMessage({ type = "info", children }) {
  return <div className={`status-message status-${type}`}>{children}</div>;
}
