const STAGES = [
  { key: "see", label: "SEE", icon: "👁", desc: "Vision Analysis" },
  { key: "reason", label: "REASON", icon: "🧠", desc: "Plan & Decide" },
  { key: "act", label: "ACT", icon: "⚡", desc: "Execute Tools" },
  { key: "audit", label: "AUDIT", icon: "🛡", desc: "Safety Check" },
];

function stageStatus(key, result, loading) {
  if (!loading && !result) return "idle";
  if (result) return "done";

  if (loading === key) return "active";
  const idx = STAGES.findIndex((s) => s.key === key);
  const loadIdx = STAGES.findIndex((s) => s.key === loading);
  if (loadIdx > idx) return "done";
  return "pending";
}

export default function PipelineView({ activeStage, result }) {
  return (
    <div className="flex items-center justify-between gap-2 py-4">
      {STAGES.map((stage, i) => {
        const status = stageStatus(stage.key, result, activeStage);
        return (
          <div key={stage.key} className="flex items-center gap-2 flex-1">
            <div className="flex flex-col items-center flex-1">
              <div
                className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-all duration-300 ${
                  status === "active"
                    ? "bg-accent/20 ring-2 ring-accent scale-110 animate-pulse-dot"
                    : status === "done"
                    ? "bg-success/15 ring-2 ring-success"
                    : status === "pending"
                    ? "bg-dark-600"
                    : "bg-dark-700"
                }`}
              >
                {status === "done" ? (
                  <svg className="w-6 h-6 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <span>{stage.icon}</span>
                )}
              </div>
              <span
                className={`text-xs font-semibold mt-1.5 tracking-wide ${
                  status === "active"
                    ? "text-accent-light"
                    : status === "done"
                    ? "text-success"
                    : "text-slate-500"
                }`}
              >
                {stage.label}
              </span>
              <span className="text-[10px] text-slate-600">{stage.desc}</span>
            </div>

            {i < STAGES.length - 1 && (
              <div
                className={`h-0.5 flex-1 min-w-4 rounded transition-colors duration-500 ${
                  status === "done" ? "bg-success/40" : "bg-dark-600"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
