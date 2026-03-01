import { useState, useEffect, useRef } from "react";

const TEAMS = [
  { id: "Facilities", label: "Facilities", icon: "🔧", color: "amber" },
  { id: "Inventory", label: "Inventory", icon: "📦", color: "blue" },
  { id: "Pest Control", label: "Pest Control", icon: "🛡", color: "red" },
];

const STAGES = [
  { id: "SEE", label: "SEE", icon: "👁", desc: "Vision Analysis" },
  { id: "REASON", label: "REASON", icon: "🧠", desc: "Plan & Decide" },
  { id: "ACT", label: "ACT", icon: "⚡", desc: "Execute Tools" },
  { id: "AUDIT", label: "AUDIT", icon: "🛡", desc: "Safety Check" },
];

export default function WarehouseDashboard({ result, activeStage, loading }) {
  const [alerts, setAlerts] = useState([]);
  const [safetyLog, setSafetyLog] = useState([]);
  const [videoError, setVideoError] = useState(null);
  const logRef = useRef(null);

  useEffect(() => {
    const fetchAlerts = () =>
      fetch("/api/alerts")
        .then((r) => r.json())
        .then((d) => setAlerts(d.alerts || []))
        .catch(() => {});
    const fetchSafetyLog = () =>
      fetch("/api/safety-log")
        .then((r) => r.json())
        .then((d) => setSafetyLog(d.entries || []))
        .catch(() => {});

    fetchAlerts();
    fetchSafetyLog();
    const t1 = setInterval(fetchAlerts, 3000);
    const t2 = setInterval(fetchSafetyLog, 2000);
    return () => {
      clearInterval(t1);
      clearInterval(t2);
    };
  }, [result]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = 0;
    }
  }, [safetyLog]);

  const toolCalls = result?.tool_calls_after_validation || result?.tool_calls || [];
  const toolResults = result?.tool_results || [];
  const dispatchedTeams = new Set(
    toolCalls
      .filter((tc) => tc.tool === "dispatch_visual_ticket")
      .map((tc) => tc.params?.team)
      .filter(Boolean)
  );

  const obs = result?.observations || {};
  const anomaly = obs.anomaly || "none";
  const severity = obs.severity || "low";
  const location = obs.location || "unknown";
  const humanPresent = obs.human_present || false;
  const framesAnalyzed = obs.frames_analyzed || result?.frames_extracted || 0;
  const allAnomalyTypes = obs.anomaly_types_seen || (anomaly !== "none" ? [anomaly] : []);
  const stagesCompleted = result?.stages_completed || [];

  const latestAlert = alerts[0];
  const latestImage = latestAlert?.image_filename
    ? `/api/alerts/${latestAlert.image_filename}`
    : latestAlert?.image_path
    ? `/api/alerts/${String(latestAlert.image_path).split("/").pop()}`
    : null;

  const inventory = result?.inventory_at_location || [];
  const audit = result?.audit;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 animate-slide-up">
      {/* ─── Dark Sector Feed ─────────────────────────────────────── */}
      <div className="lg:col-span-7">
        <div className="bg-dark-800 border border-dark-600 rounded-2xl overflow-hidden animate-glow-border">
          <div className="px-4 py-2 border-b border-dark-600 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-danger animate-pulse" />
              <span className="text-sm font-semibold text-slate-200 tracking-wide">DARK SECTOR FEED</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
              {framesAnalyzed > 0 && <span>{framesAnalyzed} frames</span>}
              <span>2 FPS</span>
            </div>
          </div>
          <div className="aspect-video bg-black flex items-center justify-center relative overflow-hidden">
            <video
              src="/api/video/demo"
              className="w-full h-full object-contain"
              controls
              muted
              loop
              autoPlay
              onError={() => setVideoError("Demo video not found. Add data/demo.mp4 or set VIDEO_DEMO_PATH")}
            />

            {/* Anomaly Overlay */}
            {allAnomalyTypes.length > 0 && (
              <div className="absolute top-3 left-3 flex flex-col gap-1">
                {allAnomalyTypes.map((a, i) => (
                  <span key={i} className={`px-2 py-1 rounded text-xs font-bold tracking-wider ${
                    severity === "critical" ? "bg-danger/90 text-white" :
                    severity === "high" ? "bg-warning/90 text-dark-900" :
                    "bg-orange-500/90 text-white"
                  }`}>
                    {a.toUpperCase()} [{severity.toUpperCase()}]
                  </span>
                ))}
                {location !== "unknown" && (
                  <span className="px-2 py-0.5 rounded bg-dark-900/80 text-slate-300 text-[10px] font-mono w-fit">
                    {location}
                  </span>
                )}
              </div>
            )}

            {humanPresent && (
              <div className="absolute top-3 right-3 px-2 py-1 rounded bg-success/90 text-white text-xs font-bold">
                HUMAN DETECTED
              </div>
            )}

            {loading && (
              <div className="absolute inset-0 bg-dark-900/60 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 border-2 border-sentinel/40 border-t-sentinel rounded-full animate-spin" />
                  <span className="text-sentinel text-sm font-mono">ANALYZING FEED...</span>
                </div>
              </div>
            )}

            {videoError && (
              <div className="absolute inset-0 flex items-center justify-center bg-dark-800/95 p-4">
                <p className="text-sm text-slate-400 text-center">{videoError}</p>
              </div>
            )}
          </div>
        </div>

        {/* Observations Summary Bar */}
        {result && (
          <div className="mt-3 bg-dark-800 border border-dark-600 rounded-xl p-3 grid grid-cols-4 gap-3 text-center">
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Anomalies</p>
              {allAnomalyTypes.length > 0 ? (
                <div className="flex flex-wrap gap-1 justify-center">
                  {allAnomalyTypes.map((a, i) => (
                    <span key={i} className="text-xs font-bold text-danger bg-danger/10 px-1.5 py-0.5 rounded">
                      {a}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-sm font-bold text-success">CLEAR</p>
              )}
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Severity</p>
              <p className={`text-sm font-bold ${
                severity === "critical" ? "text-danger" : severity === "high" ? "text-warning" : "text-slate-400"
              }`}>
                {severity.toUpperCase()}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Human</p>
              <p className={`text-sm font-bold ${humanPresent ? "text-success" : "text-slate-500"}`}>
                {humanPresent ? "PRESENT" : "CLEAR"}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Frames</p>
              <p className="text-sm font-bold text-accent-light">{framesAnalyzed}</p>
            </div>
          </div>
        )}
      </div>

      {/* ─── Right Column ─────────────────────────────────────────── */}
      <div className="lg:col-span-5 flex flex-col gap-4">

        {/* Agent Coordination Widget */}
        <div className="bg-dark-800 border border-dark-600 rounded-2xl p-4">
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
              loading ? "bg-sentinel/20 animate-brain-pulse" : result ? "bg-success/20" : "bg-dark-700"
            }`}>
              <svg className={`w-6 h-6 ${loading ? "text-sentinel" : result ? "text-success" : "text-slate-500"}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <span className="text-sm font-semibold text-slate-200">Agent Coordination</span>
              <p className="text-[10px] text-slate-500 font-mono">
                {loading ? "PROCESSING..." : result ? "CYCLE COMPLETE" : "IDLE"}
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between">
            {STAGES.map((stage, i) => {
              const isActive = activeStage === stage.id.toLowerCase();
              const isDone = stagesCompleted.includes(stage.id);
              return (
                <div key={stage.id} className="flex items-center gap-1">
                  <div className={`flex flex-col items-center gap-1 px-2 py-1.5 rounded-lg transition-all ${
                    isActive
                      ? "bg-sentinel/20 ring-1 ring-sentinel/50"
                      : isDone
                      ? "bg-success/10"
                      : "bg-dark-700"
                  }`}>
                    <span className="text-lg">{stage.icon}</span>
                    <span className={`text-[10px] font-bold ${
                      isActive ? "text-sentinel" : isDone ? "text-success" : "text-slate-500"
                    }`}>
                      {stage.label}
                    </span>
                    <span className="text-[8px] text-slate-600">{stage.desc}</span>
                  </div>
                  {i < 3 && (
                    <svg className={`w-4 h-4 ${isDone ? "text-success" : "text-dark-500"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Team Dispatch Panel */}
        <div className="bg-dark-800 border border-dark-600 rounded-2xl p-4">
          <p className="text-xs font-semibold text-slate-300 mb-3 tracking-wider uppercase">Team Dispatch</p>
          <div className="grid grid-cols-3 gap-3">
            {TEAMS.map((t) => {
              const dispatched = dispatchedTeams.has(t.id);
              return (
                <div
                  key={t.id}
                  className={`flex flex-col items-center gap-2 p-3 rounded-xl transition-all ${
                    dispatched
                      ? `bg-${t.color}-500/15 ring-2 ring-${t.color}-500/60 animate-team-alert`
                      : "bg-dark-700"
                  }`}
                >
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-2xl ${
                    dispatched ? `bg-${t.color}-500/25` : "bg-dark-600"
                  }`}>
                    {t.icon}
                  </div>
                  <span className={`text-[10px] font-semibold ${
                    dispatched ? `text-${t.color}-400` : "text-slate-500"
                  }`}>
                    {t.label}
                  </span>
                  {dispatched && (
                    <span className="text-[9px] font-bold text-danger uppercase tracking-wider animate-pulse">DISPATCHED</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Evidence Gallery */}
        <div className="bg-dark-800 border border-dark-600 rounded-2xl overflow-hidden">
          <div className="px-4 py-2 border-b border-dark-600 flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-200">Latest Visual Evidence</span>
            {latestAlert && (
              <span className="text-[10px] font-mono text-slate-500">{latestAlert.ticket_id}</span>
            )}
          </div>
          <div className="p-3 min-h-[160px]">
            {latestImage ? (
              <div>
                <img
                  src={latestImage}
                  alt="Latest alert frame"
                  className="w-full rounded-lg object-contain max-h-48 border border-dark-500"
                />
                {latestAlert && (
                  <div className="mt-2 flex items-center justify-between text-[10px]">
                    <span className="text-danger font-semibold">
                      {latestAlert.anomaly?.toUpperCase()} @ {latestAlert.location}
                    </span>
                    <span className="text-slate-500">→ {latestAlert.team}</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 text-slate-500">
                <svg className="w-12 h-12 opacity-30 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" />
                </svg>
                <p className="text-xs">No evidence captured yet</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ─── Bottom Row: Inventory + Safety Log ──────────────────── */}
      <div className="lg:col-span-5">
        <div className="bg-dark-800 border border-dark-600 rounded-2xl overflow-hidden">
          <div className="px-4 py-2 border-b border-dark-600 flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-200">Inventory Check</span>
            {inventory.length > 0 && (
              <span className="text-[10px] bg-accent/20 text-accent-light px-2 py-0.5 rounded-full font-mono">
                {inventory.length} SKUs
              </span>
            )}
          </div>
          <div className="p-3 max-h-48 overflow-y-auto safety-log-scroll">
            {inventory.length === 0 ? (
              <p className="text-xs text-slate-500 p-2">No inventory data for current location</p>
            ) : (
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-slate-500 border-b border-dark-600">
                    <th className="text-left py-1 px-2">SKU</th>
                    <th className="text-left py-1 px-2">Product</th>
                    <th className="text-right py-1 px-2">Status</th>
                    <th className="text-right py-1 px-2">Qty</th>
                  </tr>
                </thead>
                <tbody>
                  {inventory.map((item, i) => (
                    <tr key={i} className="border-b border-dark-700/50">
                      <td className="py-1.5 px-2 font-mono text-accent-light">{item.sku}</td>
                      <td className="py-1.5 px-2 text-slate-400">{item.product_name}</td>
                      <td className={`py-1.5 px-2 text-right font-semibold ${
                        item.status === "QUARANTINED" ? "text-danger" : "text-success"
                      }`}>
                        {item.status}
                      </td>
                      <td className="py-1.5 px-2 text-right text-slate-400">{item.quantity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {/* Safety Log Terminal */}
      <div className="lg:col-span-7">
        <div className="bg-dark-800 border border-dark-600 rounded-2xl overflow-hidden">
          <div className="px-4 py-2 border-b border-dark-600 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
              <span className="text-sm font-semibold text-slate-200">Safety Log</span>
              <span className="text-[10px] font-mono text-slate-500">SENTINEL AUDIT TRAIL</span>
            </div>
            {audit && (
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                audit.approved ? "bg-success/15 text-success" : "bg-danger/15 text-danger"
              }`}>
                {audit.approved ? "APPROVED" : "BLOCKED"}
              </span>
            )}
          </div>

          {/* Policy checks from current result */}
          {audit?.policy_checks?.length > 0 && (
            <div className="px-4 py-2 border-b border-dark-700 bg-dark-900/50">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Policy Enforcement</p>
              <div className="flex flex-wrap gap-2">
                {audit.policy_checks.map((pc, i) => (
                  <span
                    key={i}
                    className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                      pc.result === "PASS"
                        ? "bg-success/10 text-success border border-success/20"
                        : "bg-danger/10 text-danger border border-danger/20"
                    }`}
                  >
                    {pc.policy}: {pc.result}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div ref={logRef} className="p-3 h-52 overflow-y-auto font-mono text-[11px] space-y-1.5 safety-log-scroll bg-dark-900/30">
            {safetyLog.length === 0 ? (
              <p className="text-slate-600 p-2">Waiting for audit events...</p>
            ) : (
              safetyLog.map((entry, i) => (
                <div
                  key={i}
                  className={`p-2 rounded flex items-start gap-2 ${
                    entry.verdict === "BLOCKED"
                      ? "bg-danger/8 border-l-2 border-danger"
                      : "bg-dark-800 border-l-2 border-success/40"
                  }`}
                >
                  <span className={`font-bold shrink-0 ${
                    entry.verdict === "BLOCKED" ? "text-danger" : "text-success"
                  }`}>
                    [{entry.verdict}]
                  </span>
                  <div className="flex-1 min-w-0">
                    <span className="text-slate-400 break-words">{entry.reason || "—"}</span>
                    {entry.tools?.length > 0 && (
                      <span className="text-slate-600 ml-1">
                        | {entry.tools.join(", ")}
                      </span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
