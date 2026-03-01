import { useState, useEffect } from "react";
import InputPanel from "./components/InputPanel";
import PipelineView from "./components/PipelineView";
import ResultsPanel from "./components/ResultsPanel";
import ObservabilityDashboard from "./components/ObservabilityDashboard";
import WarehouseDashboard from "./components/WarehouseDashboard";

const STAGE_SEQUENCE = ["see", "reason", "act", "audit"];
const STAGE_TIMINGS = [800, 2500, 1500, 600];

export default function App() {
  const [view, setView] = useState("pipeline");
  const [showWarehouse, setShowWarehouse] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resettingSession, setResettingSession] = useState(false);
  const [activeStage, setActiveStage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [setupDone, setSetupDone] = useState(false);
  const [sessionNotice, setSessionNotice] = useState(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  async function setupDomainTools() {
    try {
      await fetch("/api/register-domain-tools", { method: "POST" });
      setSetupDone(true);
    } catch {
      /* already registered or not available */
    }
  }

  useEffect(() => {
    if (health?.status && health.status !== "offline" && !setupDone) {
      setupDomainTools();
    }
  }, [health]);

  async function handleSubmit({ text, image, video }) {
    setLoading(true);
    setResult(null);
    setError(null);

    let stageIdx = 0;
    setActiveStage(STAGE_SEQUENCE[0]);

    const interval = setInterval(() => {
      stageIdx++;
      if (stageIdx < STAGE_SEQUENCE.length) {
        setActiveStage(STAGE_SEQUENCE[stageIdx]);
      }
    }, STAGE_TIMINGS[stageIdx] || 1500);

    try {
      const form = new FormData();
      if (text) form.append("text", text);
      if (image) form.append("image", image);
      if (video) form.append("video", video);

      const res = await fetch("/api/run", { method: "POST", body: form });

      clearInterval(interval);

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setActiveStage(null);
      setResult(data);
    } catch (err) {
      clearInterval(interval);
      setActiveStage(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleNewSession() {
    if (resettingSession || loading) return;
    setResettingSession(true);
    setError(null);
    try {
      const res = await fetch("/api/session/new", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setResult(null);
      setActiveStage(null);
      setSessionNotice({
        sessionId: data.session_id,
        ts: new Date().toLocaleTimeString(),
      });
    } catch (err) {
      setError(err.message || "Failed to start a new session");
    } finally {
      setResettingSession(false);
    }
  }

  const online = health?.status === "ok" || health?.status === "degraded";

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <header className="border-b border-dark-600 bg-dark-800/90 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sentinel/30 to-accent/20 flex items-center justify-center">
              <svg className="w-6 h-6 text-sentinel" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">
                Sovereign Sentinel
              </h1>
              <p className="text-[10px] text-sentinel/70 font-mono tracking-wider">
                FACILITY INTELLIGENCE SYSTEM
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {health && (
              <div className="flex items-center gap-2 text-xs">
                <div className={`w-2 h-2 rounded-full ${online ? "bg-success animate-pulse" : "bg-danger"}`} />
                <span className={online ? "text-success" : "text-danger"}>
                  {health.llm_connected ? "LLM Connected" : "LLM Offline"}
                </span>
              </div>
            )}
            {health?.matformer && (
              <span className="text-[10px] font-mono text-slate-600 bg-dark-700 px-2 py-1 rounded">
                MatFormer {health.matformer}
              </span>
            )}
            <button
              onClick={() => setShowWarehouse((p) => !p)}
              className="text-xs text-slate-500 hover:text-sentinel transition-colors"
            >
              {showWarehouse ? "Agent View" : "Command Center"}
            </button>
            <button
              onClick={() => setView((prev) => prev === "pipeline" ? "observability" : "pipeline")}
              className="text-xs text-slate-500 hover:text-sentinel transition-colors"
            >
              {view === "pipeline" ? "InstaControl" : "Agent"}
            </button>
            <button
              type="button"
              onClick={handleNewSession}
              disabled={resettingSession || loading}
              className="text-xs px-3 py-1.5 rounded border border-dark-500 text-slate-300 hover:text-sentinel hover:border-sentinel/40 disabled:opacity-60 disabled:cursor-not-allowed transition-all"
            >
              {resettingSession ? "Resetting..." : "New Session"}
            </button>
          </div>
        </div>
      </header>

      {view === "pipeline" ? (
        <>
          {(loading || result) && !showWarehouse && (
            <div className="max-w-4xl mx-auto px-6 pt-6">
              <PipelineView activeStage={activeStage} result={result} />
            </div>
          )}

          <main className="max-w-[1600px] mx-auto px-6 py-6">
            {showWarehouse ? (
              <WarehouseDashboard result={result} activeStage={activeStage} loading={loading} />
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-4">
                  <div className="bg-dark-800 border border-dark-600 rounded-2xl p-6 sticky top-24">
                    <h2 className="text-base font-semibold text-slate-200 mb-5">
                      Agent Input
                    </h2>
                    <InputPanel onSubmit={handleSubmit} loading={loading} />

                    {!online && (
                      <div className="mt-4 bg-danger/10 border border-danger/20 rounded-lg p-3">
                        <p className="text-xs text-danger">
                          LLM server is not reachable. Run: ollama serve
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="lg:col-span-8">
                  {sessionNotice && (
                    <div className="bg-success/10 border border-success/20 rounded-xl p-4 mb-4 animate-slide-up">
                      <p className="text-sm text-success font-medium">Fresh session started</p>
                      <p className="text-xs text-success/80 mt-1">
                        Session ID: {sessionNotice.sessionId} · {sessionNotice.ts}
                      </p>
                    </div>
                  )}
                  {error && (
                    <div className="bg-danger/10 border border-danger/20 rounded-xl p-5 mb-4 animate-slide-up">
                      <p className="text-sm text-danger font-medium">Error</p>
                      <p className="text-xs text-danger/80 mt-1">{error}</p>
                    </div>
                  )}

                  {result ? (
                    <ResultsPanel result={result} />
                  ) : (
                    !loading && (
                      <div className="flex flex-col items-center justify-center h-96 text-center">
                        <div className="w-20 h-20 rounded-2xl bg-dark-700 flex items-center justify-center mb-5">
                          <svg className="w-10 h-10 text-sentinel/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <p className="text-slate-500 text-sm">
                          Upload a warehouse video or describe a task to run the Sentinel pipeline.
                        </p>
                        <p className="text-slate-600 text-xs mt-2 font-mono">
                          SEE → REASON → ACT → AUDIT
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </main>
        </>
      ) : (
        <main className="max-w-[1600px] mx-auto px-6 py-8">
          <ObservabilityDashboard />
        </main>
      )}
    </div>
  );
}
