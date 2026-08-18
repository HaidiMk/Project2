const OPTIONS = [
  { value: "numbers", label: "Numbers" },
  { value: "chart", label: "Chart" },
  { value: "both", label: "Both" },
];

export default function ViewModeSelector({ value, onChange }) {
  return (
    <div className="view-mode-selector" role="group" aria-label="System overview display mode">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`view-mode-btn${value === option.value ? " active" : ""}`}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
