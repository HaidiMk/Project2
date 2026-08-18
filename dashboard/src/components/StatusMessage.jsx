export default function StatusMessage({ type = "info", children }) {
  return <div className={`status-message status-${type}`}>{children}</div>;
}
