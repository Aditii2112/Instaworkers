function Card({ title, icon, children, variant = "default" }) {
  const border =
    variant === "success"
      ? "border-success/20"
      : variant === "danger"
      ? "border-danger/20"
      : variant === "warning"
      ? "border-warning/20"
      : "border-dark-500";

  return (
    <div className={`bg-dark-700 border ${border} rounded-xl p-5 animate-slide-up`}>
      <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-3">
        <span className="text-lg">{icon}</span>
        {title}
      </h3>
      {children}
    </div>
  );
}

function KV({ label, value, variant }) {
  const valueColor =
    variant === "danger" ? "text-danger" :
    variant === "success" ? "text-success" :
    variant === "warning" ? "text-warning" :
    "text-slate-300";
  return (
    <div className="flex justify-between py-1.5 border-b border-dark-600 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`text-xs text-right max-w-[65%] break-words ${valueColor}`}>
        {value ?? "—"}
      </span>
    </div>
  );
}

function ObservationsCard({ obs }) {
  if (!obs) return null;
  const anomaly = obs.anomaly || "none";
  const severity = obs.severity || "low";
  const allAnomalies = obs.anomaly_types_seen || (anomaly !== "none" ? [anomaly] : []);

  return (
    <Card title="Observations" icon="👁">
      <KV label="Source" value={obs.source} />
      <KV label="Frame" value={obs.frame_shape ? obs.frame_shape.join(" × ") : "No image"} />
      <KV
        label="Anomalies Detected"
        value={allAnomalies.length > 0 ? allAnomalies.map(a => a.toUpperCase()).join(", ") : "CLEAR"}
        variant={allAnomalies.length > 0 ? "danger" : "success"}
      />
      <KV
        label="Severity"
        value={severity.toUpperCase()}
        variant={severity === "critical" ? "danger" : severity === "high" ? "warning" : undefined}
      />
      <KV label="Location" value={obs.location} />
      <KV
        label="Human present"
        value={obs.human_present != null ? String(obs.human_present) : "—"}
        variant={obs.human_present ? "success" : undefined}
      />
      <KV label="Faces detected" value={obs.face_count ?? 0} />
      <KV label="Hands detected" value={obs.hand_count ?? 0} />
      {obs.frames_analyzed > 0 && <KV label="Frames analyzed" value={obs.frames_analyzed} />}
      {obs.raw_vlm && (
        <div className="mt-3 pt-3 border-t border-dark-600">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">VLM Output</span>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed font-mono bg-dark-800 rounded p-2">
            {obs.raw_vlm.slice(0, 300)}
          </p>
        </div>
      )}
      {obs.text_input && (
        <p className="text-xs text-slate-400 mt-3 bg-dark-800 rounded-lg p-3 leading-relaxed">
          {obs.text_input}
        </p>
      )}
    </Card>
  );
}

function VideoInfoCard({ videoInfo, framesExtracted }) {
  if (!videoInfo) return null;
  return (
    <Card title="Video Analysis" icon="🎥">
      <KV label="Duration" value={`${videoInfo.duration_s}s`} />
      <KV label="Resolution" value={videoInfo.resolution?.join(" × ")} />
      <KV label="Source FPS" value={videoInfo.fps} />
      <KV label="Sample FPS" value={videoInfo.sample_fps} />
      <KV label="Frames extracted" value={framesExtracted} />
      <KV label="Total source frames" value={videoInfo.total_frames} />
    </Card>
  );
}

function ContextCard({ context }) {
  if (!context) return null;
  const memories = context.memories || [];
  return (
    <Card title={`Context Retrieved (${memories.length})`} icon="📚">
      {memories.length === 0 ? (
        <p className="text-xs text-slate-500 italic">No context found</p>
      ) : (
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {memories.map((m, i) => (
            <div key={i} className="bg-dark-800 rounded-lg p-3">
              <p className="text-xs text-slate-400 leading-relaxed">{m.content}</p>
              {m.distance != null && (
                <span className="text-[10px] text-slate-600 mt-1 inline-block">
                  similarity: {(1 - m.distance).toFixed(3)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function PlanCard({ plan }) {
  if (!plan) return null;
  return (
    <Card title="Reasoning Plan" icon="🧠">
      {plan.analysis && (
        <div className="mb-3">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">Analysis</span>
          <p className="text-sm text-slate-300 mt-1 leading-relaxed">{plan.analysis}</p>
        </div>
      )}
      {plan.rationale && (
        <div className="mb-3">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">Rationale</span>
          <p className="text-sm text-slate-400 mt-1 leading-relaxed">{plan.rationale}</p>
        </div>
      )}
      {plan.plan?.length > 0 && (
        <div>
          <span className="text-[10px] uppercase tracking-wider text-slate-500">Planned Steps</span>
          <div className="mt-2 space-y-1.5">
            {plan.plan.map((step, i) => (
              <div key={i} className="flex items-start gap-2 bg-dark-800 rounded-lg px-3 py-2">
                <span className="text-accent font-mono text-xs mt-0.5">{i + 1}.</span>
                <span className="text-xs text-slate-300">
                  {typeof step === "string"
                    ? step
                    : `${step.tool_name || step.tool || "?"}(${JSON.stringify(step.parameters || step.params || {})})`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {plan.raw_output && !plan.plan?.length && (
        <pre className="text-xs text-slate-500 bg-dark-800 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
          {plan.raw_output}
        </pre>
      )}
    </Card>
  );
}

function ToolCallsCard({ toolCalls, toolResults }) {
  if (!toolCalls?.length) return null;
  return (
    <Card title={`Tool Calls (${toolCalls.length})`} icon="⚡">
      <div className="space-y-3">
        {toolCalls.map((tc, i) => {
          const result = toolResults?.[i];
          const ok = result?.status === "ok";
          return (
            <div
              key={i}
              className={`bg-dark-800 rounded-lg p-3 border-l-2 ${
                result ? (ok ? "border-success/60" : "border-danger/60") : "border-slate-600"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-mono text-accent-light">{tc.tool}</span>
                {result && (
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                    ok ? "bg-success/15 text-success" : "bg-danger/15 text-danger"
                  }`}>
                    {ok ? "OK" : "ERR"}{" "}
                    {result.latency_ms != null && `${result.latency_ms.toFixed(1)}ms`}
                  </span>
                )}
              </div>
              <pre className="text-[11px] text-slate-500 whitespace-pre-wrap">
                {JSON.stringify(tc.params || {}, null, 2)}
              </pre>
              {result?.result && (
                <details className="mt-2">
                  <summary className="text-[10px] text-slate-500 cursor-pointer hover:text-slate-400">
                    View Result
                  </summary>
                  <pre className="text-[11px] text-slate-400 mt-1 whitespace-pre-wrap bg-dark-900 rounded p-2 max-h-32 overflow-y-auto">
                    {JSON.stringify(result.result, null, 2)}
                  </pre>
                </details>
              )}
              {result?.error && (
                <p className="text-[11px] text-danger mt-1">{result.error}</p>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function AuditCard({ audit }) {
  if (!audit) return null;
  const approved = audit.approved;
  return (
    <Card title="Audit Verdict" icon="🛡" variant={approved ? "success" : "danger"}>
      <div className="flex items-center gap-3 mb-3">
        <span className={`text-sm font-bold px-3 py-1 rounded-full ${
          approved ? "bg-success/15 text-success" : "bg-danger/15 text-danger"
        }`}>
          {approved ? "APPROVED" : "BLOCKED"}
        </span>
      </div>
      {audit.reason && (
        <p className="text-xs text-slate-400 leading-relaxed">{audit.reason}</p>
      )}
      {audit.policy_checks?.length > 0 && (
        <div className="mt-3 space-y-1">
          {audit.policy_checks.map((pc, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className={`font-mono font-bold ${pc.result === "PASS" ? "text-success" : "text-danger"}`}>
                {pc.policy}
              </span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                pc.result === "PASS" ? "bg-success/10 text-success" : "bg-danger/10 text-danger"
              }`}>
                {pc.result}
              </span>
              <span className="text-slate-500 text-[11px]">{pc.detail}</span>
            </div>
          ))}
        </div>
      )}
      {audit.warnings?.length > 0 && (
        <div className="mt-3 space-y-1">
          {audit.warnings.map((w, i) => (
            <p key={i} className="text-xs text-warning">⚠ {w}</p>
          ))}
        </div>
      )}
    </Card>
  );
}

function ValidationCard({ validation }) {
  if (!validation) return null;
  const queued = validation.queued || [];
  const blocked = validation.blocked || [];
  const records = validation.records || [];
  const connectivity = validation.connectivity?.state || "unknown";

  return (
    <Card title="Hybrid Validation" icon="🛰" variant={blocked.length ? "warning" : "default"}>
      <KV label="Connectivity" value={connectivity} />
      <KV label="Records" value={records.length} />
      <KV label="Queued for Teacher" value={queued.length} />
      <KV label="Blocked / HITL" value={blocked.length} />
      {queued.length > 0 && (
        <div className="mt-3 space-y-1">
          {queued.map((q, i) => (
            <p key={i} className="text-xs text-warning">⏳ {q.tool}: {q.reason}</p>
          ))}
        </div>
      )}
      {blocked.length > 0 && (
        <div className="mt-3 space-y-1">
          {blocked.map((b, i) => (
            <p key={i} className="text-xs text-danger">⛔ {b.tool}: {b.reason}</p>
          ))}
        </div>
      )}
    </Card>
  );
}

export default function ResultsPanel({ result }) {
  if (!result) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold text-slate-200">Results</h2>
        <div className="flex items-center gap-3">
          {result.stages_completed?.length > 0 && (
            <span className="text-[10px] font-mono text-sentinel/70">
              {result.stages_completed.join(" → ")}
            </span>
          )}
          <span className="text-[10px] font-mono text-slate-600">
            {result.correlation_id?.slice(0, 8)}
          </span>
        </div>
      </div>

      <ObservationsCard obs={result.observations} />
      <VideoInfoCard videoInfo={result.video_info} framesExtracted={result.frames_extracted} />
      <ContextCard context={result.context} />
      <PlanCard plan={result.plan} />
      <ToolCallsCard
        toolCalls={result.tool_calls}
        toolResults={result.tool_results}
      />
      <AuditCard audit={result.audit} />
      <ValidationCard validation={result.validation} />
    </div>
  );
}
