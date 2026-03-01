import { useCallback, useEffect, useMemo, useState } from "react";

function StatCard({ label, value, subtext }) {
  return (
    <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-2xl font-semibold text-slate-200 mt-1">{value}</p>
      {subtext && <p className="text-[11px] text-slate-600 mt-1">{subtext}</p>}
    </div>
  );
}

function StageRow({ stage, stats }) {
  const avg = Number(stats?.avg_ms || 0).toFixed(1);
  const max = Number(stats?.max_ms || 0).toFixed(1);
  const count = stats?.count || 0;
  return (
    <div className="flex items-center justify-between py-2 border-b border-dark-700 last:border-0">
      <span className="text-sm text-slate-300">{stage}</span>
      <span className="text-xs text-slate-500">{count} calls</span>
      <div className="text-right">
        <p className="text-sm font-mono text-accent-light">{avg} ms</p>
        <p className="text-[11px] text-slate-600">max {max} ms</p>
      </div>
    </div>
  );
}

function TinyBar({ label, value, total }) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-500">{value}</span>
      </div>
      <div className="w-full bg-dark-700 rounded h-2">
        <div className="h-2 rounded bg-accent" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function DonutChart({ segments, centerValue, centerLabel }) {
  const size = 138;
  const strokeWidth = 18;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const total = segments.reduce((acc, s) => acc + s.value, 0);

  if (!total) {
    return (
      <div className="h-[138px] w-[138px] rounded-full border border-dark-600 bg-dark-700/50 grid place-items-center">
        <p className="text-xs text-slate-500">No data</p>
      </div>
    );
  }

  let dashOffset = 0;
  return (
    <div className="relative h-[138px] w-[138px]">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="rgba(37,37,53,0.8)"
          strokeWidth={strokeWidth}
        />
        {segments.map((segment) => {
          const segmentLength = (segment.value / total) * circumference;
          const circle = (
            <circle
              key={segment.label}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke={segment.color}
              strokeWidth={strokeWidth}
              strokeDasharray={`${segmentLength} ${circumference - segmentLength}`}
              strokeDashoffset={-dashOffset}
              strokeLinecap="butt"
            />
          );
          dashOffset += segmentLength;
          return circle;
        })}
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <p className="text-xl font-semibold text-slate-100 leading-none">{centerValue}</p>
        <p className="text-[11px] text-slate-500 mt-1">{centerLabel}</p>
      </div>
    </div>
  );
}

function Sparkline({ points }) {
  const width = 560;
  const height = 140;
  if (!points.length) {
    return (
      <div className="h-[140px] rounded-lg border border-dark-700 bg-dark-700/50 grid place-items-center">
        <p className="text-xs text-slate-500">No timeline points yet.</p>
      </div>
    );
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(1, max - min);

  const normalized = points.map((value, idx) => {
    const x = points.length === 1 ? width / 2 : (idx / (points.length - 1)) * width;
    const y = height - ((value - min) / span) * height;
    return [x, y];
  });

  const polyline = normalized.map(([x, y]) => `${x},${y}`).join(" ");
  const area = `${normalized.map(([x, y]) => `${x},${y}`).join(" ")} ${width},${height} 0,${height}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-[140px]">
      <defs>
        <linearGradient id="latencyFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(99,102,241,0.45)" />
          <stop offset="100%" stopColor="rgba(99,102,241,0)" />
        </linearGradient>
      </defs>
      <polygon points={area} fill="url(#latencyFill)" />
      <polyline
        points={polyline}
        fill="none"
        stroke="rgb(129,140,248)"
        strokeWidth="3"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function LegendItem({ label, value, color }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="inline-flex items-center gap-2 text-slate-400">
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
        {label}
      </span>
      <span className="text-slate-200">{value}</span>
    </div>
  );
}

function statusTone(status) {
  if (status === "success") return "text-success";
  if (status === "error" || status === "blocked") return "text-danger";
  if (status === "queued") return "text-warning";
  return "text-slate-400";
}

export default function ObservabilityDashboard() {
  const [windowMinutes, setWindowMinutes] = useState(60);
  const [eventStageFilter, setEventStageFilter] = useState("all");
  const [eventStatusFilter, setEventStatusFilter] = useState("all");
  const [summary, setSummary] = useState(null);
  const [timeseries, setTimeseries] = useState([]);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const eventsParams = new URLSearchParams({ limit: "120" });
      if (eventStageFilter !== "all") {
        eventsParams.set("stage", eventStageFilter);
      }
      if (eventStatusFilter !== "all") {
        eventsParams.set("status", eventStatusFilter);
      }
      const [sRes, tRes, eRes] = await Promise.all([
        fetch(`/api/metrics/summary?window_minutes=${windowMinutes}`),
        fetch(`/api/metrics/timeseries?window_minutes=${windowMinutes}&bucket_seconds=60`),
        fetch(`/api/events?${eventsParams.toString()}`),
      ]);

      const [sData, tData, eData] = await Promise.all([
        sRes.json(),
        tRes.json(),
        eRes.json(),
      ]);

      setSummary(sData);
      setTimeseries(tData.points || []);
      setEvents(eData.events || []);
    } catch (err) {
      setError(err.message || "Failed to load observability metrics");
    }
  }, [windowMinutes, eventStageFilter, eventStatusFilter]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10000);
    return () => clearInterval(id);
  }, [refresh]);

  const latestPoint = timeseries.length ? timeseries[timeseries.length - 1] : null;
  const granularityTotal = useMemo(
    () =>
      Object.values(summary?.granularity || {}).reduce(
        (acc, item) => acc + (item?.count || 0),
        0
      ),
    [summary]
  );
  const errorSummary = useMemo(() => {
    const raw = Object.values(summary?.error_rate || {});
    const total = raw.reduce((acc, row) => acc + (row?.total || 0), 0);
    const errors = raw.reduce((acc, row) => acc + (row?.errors || 0), 0);
    const rate = total ? (errors / total) * 100 : 0;
    return { total, errors, rate };
  }, [summary]);
  const timelinePoints = useMemo(
    () => timeseries.map((p) => Number(p?.avg_latency_ms || 0)),
    [timeseries]
  );
  const timelineEventTotal = useMemo(
    () => timeseries.reduce((acc, p) => acc + Number(p?.event_count || 0), 0),
    [timeseries]
  );
  const timelineErrorTotal = useMemo(
    () => timeseries.reduce((acc, p) => acc + Number(p?.error_count || 0), 0),
    [timeseries]
  );

  const chartPalette = [
    "#818cf8",
    "#22d3ee",
    "#f59e0b",
    "#22c55e",
    "#f472b6",
    "#ef4444",
  ];
  const granularitySegments = Object.entries(summary?.granularity || {}).map(
    ([label, details], idx) => ({
      label,
      value: details?.count || 0,
      color: chartPalette[idx % chartPalette.length],
    })
  );
  const connectivitySegments = Object.entries(summary?.connectivity || {}).map(
    ([label, value], idx) => ({
      label,
      value: Number(value || 0),
      color: chartPalette[idx % chartPalette.length],
    })
  );
  const pendingSegments = Object.entries(summary?.pending_validation_queue || {}).map(
    ([label, value], idx) => ({
      label,
      value: Number(value || 0),
      color: chartPalette[idx % chartPalette.length],
    })
  );
  const totalPending = pendingSegments.reduce((acc, s) => acc + s.value, 0);
  const checkpointRate = Number(summary?.checkpoints?.per_hour || 0).toFixed(2);
  const retrievalCalls = Number(summary?.retrieval?.retrieval_calls || 0);
  const retrievalLatency = Number(summary?.retrieval?.avg_retrieval_latency_ms || 0).toFixed(1);
  const hasExportData = Boolean(summary || timeseries.length || events.length);

  const handleExport = useCallback(() => {
    const payload = {
      exported_at: new Date().toISOString(),
      window_minutes: windowMinutes,
      filters: {
        stage: eventStageFilter,
        status: eventStatusFilter,
      },
      summary: summary || {},
      timeseries,
      events,
    };

    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `instacontrol-summary-${stamp}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [summary, timeseries, events, windowMinutes, eventStageFilter, eventStatusFilter]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-200">
            InstaControl Observability Surface
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Edge dashboard (React) from ADK event trace + local metrics.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={windowMinutes}
            onChange={(e) => setWindowMinutes(Number(e.target.value))}
            className="bg-dark-800 border border-dark-600 text-slate-300 text-xs rounded px-2 py-1"
          >
            <option value={15}>15m</option>
            <option value={60}>1h</option>
            <option value={240}>4h</option>
            <option value={1440}>24h</option>
          </select>
          <button
            onClick={refresh}
            className="text-xs px-3 py-1.5 rounded bg-dark-700 hover:bg-dark-600 text-slate-300"
          >
            Refresh
          </button>
          <button
            onClick={handleExport}
            disabled={!hasExportData}
            className="text-xs px-3 py-1.5 rounded bg-accent/20 text-accent-light border border-accent/30 hover:bg-accent/30 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Download current dashboard snapshot as JSON"
          >
            Export JSON
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-danger/10 border border-danger/20 rounded-xl p-4">
          <p className="text-xs text-danger">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          label="CPU"
          value={`${Number(summary?.system?.cpu_percent || 0).toFixed(1)}%`}
        />
        <StatCard
          label="RAM"
          value={`${Number(summary?.system?.ram_percent || 0).toFixed(1)}%`}
          subtext={`${Number(summary?.system?.ram_used_mb || 0).toFixed(0)} / ${Number(
            summary?.system?.ram_total_mb || 0
          ).toFixed(0)} MB`}
        />
        <StatCard
          label="End-to-End Avg"
          value={`${Number(summary?.end_to_end?.avg_ms || 0).toFixed(1)} ms`}
          subtext={`${summary?.end_to_end?.count || 0} cycles`}
        />
        <StatCard
          label="Latest Bucket"
          value={`${Number(latestPoint?.avg_latency_ms || 0).toFixed(1)} ms`}
          subtext={`${latestPoint?.event_count || 0} events`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">Latency Timeline</h3>
          <p className="text-[11px] text-slate-500 mb-3">
            Trend of average stage latency per bucket.
          </p>
          <Sparkline points={timelinePoints} />
          <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
            <div className="bg-dark-700 rounded p-2">
              <p className="text-slate-500">Buckets</p>
              <p className="text-slate-200">{timeseries.length}</p>
            </div>
            <div className="bg-dark-700 rounded p-2">
              <p className="text-slate-500">Events</p>
              <p className="text-slate-200">{timelineEventTotal}</p>
            </div>
            <div className="bg-dark-700 rounded p-2">
              <p className="text-slate-500">Errors</p>
              <p className="text-slate-200">{timelineErrorTotal}</p>
            </div>
          </div>
        </div>

        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Stage Latency (Detail)</h3>
          {["SEE", "REASON", "ACT", "AUDIT"].map((stage) => (
            <StageRow key={stage} stage={stage} stats={summary?.stage_latency?.[stage]} />
          ))}
        </div>

        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Reliability Snapshot</h3>
          <div className="space-y-2 text-xs mb-4">
            <p className="text-slate-400">
              Total events checked: <span className="text-slate-200">{errorSummary.total}</span>
            </p>
            <p className="text-slate-400">
              Error events: <span className="text-danger">{errorSummary.errors}</span>
            </p>
            <p className="text-slate-400">
              Error rate: <span className="text-slate-200">{errorSummary.rate.toFixed(2)}%</span>
            </p>
          </div>
          <div className="space-y-2 text-xs">
            <p className="text-slate-400">
              Blocked by policy:{" "}
              <span className="text-slate-200">{summary?.safety?.blocked_by_policy || 0}</span>
            </p>
            <p className="text-slate-400">
              Auditor rejections:{" "}
              <span className="text-slate-200">{summary?.safety?.auditor_rejections || 0}</span>
            </p>
            <p className="text-slate-400">
              Queued actions:{" "}
              <span className="text-slate-200">{summary?.safety?.queued_actions || 0}</span>
            </p>
            <p className="text-slate-400">
              HITL prompts:{" "}
              <span className="text-slate-200">{summary?.safety?.hitl_prompts || 0}</span>
            </p>
            <p className="text-slate-400">
              Retrieval calls: <span className="text-slate-200">{retrievalCalls}</span>
            </p>
            <p className="text-slate-400">
              Avg retrieval latency: <span className="text-slate-200">{retrievalLatency} ms</span>
            </p>
            <p className="text-slate-400">
              Checkpoints/hour: <span className="text-slate-200">{checkpointRate}</span>
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Granularity Mix</h3>
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            <DonutChart
              segments={granularitySegments}
              centerValue={granularityTotal}
              centerLabel="total decisions"
            />
            <div className="space-y-2 flex-1">
              {granularitySegments.map((segment) => (
                <LegendItem
                  key={segment.label}
                  label={segment.label}
                  value={`${segment.value}`}
                  color={segment.color}
                />
              ))}
              {!granularitySegments.length && (
                <p className="text-xs text-slate-500">No granularity data yet.</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Connectivity States</h3>
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            <DonutChart
              segments={connectivitySegments}
              centerValue={connectivitySegments.reduce((acc, s) => acc + s.value, 0)}
              centerLabel="state updates"
            />
            <div className="space-y-2 flex-1">
              {connectivitySegments.map((segment) => (
                <LegendItem
                  key={segment.label}
                  label={segment.label}
                  value={`${segment.value}`}
                  color={segment.color}
                />
              ))}
              {!connectivitySegments.length && (
                <p className="text-xs text-slate-500">No connectivity data yet.</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Pending Validation Queue</h3>
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            <DonutChart
              segments={pendingSegments}
              centerValue={totalPending}
              centerLabel="queued items"
            />
            <div className="space-y-2 flex-1">
              {pendingSegments.map((segment) => (
                <LegendItem
                  key={segment.label}
                  label={segment.label}
                  value={`${segment.value}`}
                  color={segment.color}
                />
              ))}
              {!pendingSegments.length && (
                <p className="text-xs text-slate-500">No pending validations recorded.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Granularity Volume Bars</h3>
        <div className="space-y-3">
          {Object.entries(summary?.granularity || {}).map(([name, details]) => (
            <TinyBar
              key={name}
              label={name}
              value={details?.count || 0}
              total={granularityTotal}
            />
          ))}
          {!Object.keys(summary?.granularity || {}).length && (
            <p className="text-xs text-slate-500">No granularity data yet.</p>
          )}
        </div>
      </div>

      <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
          <h3 className="text-sm font-semibold text-slate-300">Latest Events</h3>
          <div className="flex items-center gap-2">
            <select
              value={eventStageFilter}
              onChange={(e) => setEventStageFilter(e.target.value)}
              className="bg-dark-700 border border-dark-600 text-slate-300 text-xs rounded px-2 py-1"
            >
              <option value="all">All stages</option>
              <option value="SEE">SEE</option>
              <option value="REASON">REASON</option>
              <option value="ACT">ACT</option>
              <option value="AUDIT">AUDIT</option>
            </select>
            <select
              value={eventStatusFilter}
              onChange={(e) => setEventStatusFilter(e.target.value)}
              className="bg-dark-700 border border-dark-600 text-slate-300 text-xs rounded px-2 py-1"
            >
              <option value="all">All statuses</option>
              <option value="success">success</option>
              <option value="queued">queued</option>
              <option value="blocked">blocked</option>
              <option value="error">error</option>
            </select>
          </div>
        </div>
        <p className="text-[11px] text-slate-500 mb-3">
          Showing {events.length} events
          {eventStageFilter !== "all" ? ` · stage: ${eventStageFilter}` : ""}
          {eventStatusFilter !== "all" ? ` · status: ${eventStatusFilter}` : ""}
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-dark-700">
                <th className="text-left py-2 pr-2">Time</th>
                <th className="text-left py-2 pr-2">Stage</th>
                <th className="text-left py-2 pr-2">Status</th>
                <th className="text-left py-2 pr-2">Decision</th>
                <th className="text-left py-2 pr-2">Latency</th>
                <th className="text-left py-2 pr-2">Granularity</th>
              </tr>
            </thead>
            <tbody>
              {events.slice().reverse().map((ev, i) => (
                <tr key={`${ev.correlation_id}-${i}`} className="border-b border-dark-700/70">
                  <td className="py-2 pr-2 text-slate-500">{ev.ts?.slice(11, 19)}</td>
                  <td className="py-2 pr-2 text-slate-300">{ev.stage}</td>
                  <td className={`py-2 pr-2 ${statusTone(ev.decision?.status || "unknown")}`}>
                    {ev.decision?.status || "unknown"}
                  </td>
                  <td className="py-2 pr-2 text-slate-500">{ev.decision?.type || "-"}</td>
                  <td className="py-2 pr-2 text-slate-400">
                    {Number(ev.latency_ms || 0).toFixed(1)} ms
                  </td>
                  <td className="py-2 pr-2 text-slate-500">{ev.model_granularity || "M"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!events.length && <p className="text-xs text-slate-500 py-2">No events yet.</p>}
        </div>
      </div>
    </div>
  );
}
