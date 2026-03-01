"""InstaControl View — Streamlit dashboard for edge agent observability.

Run with:  streamlit run observability/dashboard.py
"""

import sys
import time as _time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from observability.event_trace import EventTracer
from observability.metrics import MetricsCollector

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="InstaControl View",
    page_icon="📊",
    layout="wide",
)

cfg = load_config()
metrics = MetricsCollector(cfg.observability.events_path)
tracer = EventTracer(cfg.observability.events_path)

# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("InstaControl View")
st.sidebar.markdown("Edge Agent Observability Dashboard")
st.sidebar.markdown("---")

refresh_rate = st.sidebar.selectbox(
    "Auto-refresh (seconds)", [5, 10, 30, 60], index=1
)
window_minutes = st.sidebar.slider("Time window (minutes)", 5, 1440, 60)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**MatFormer Profile:** `{cfg.matformer.granularity}`  \n"
    f"**Checkpointing:** `{'ON' if cfg.checkpoint.enabled else 'OFF'}`  \n"
    f"**Auditor:** `{'ON' if cfg.auditor.enabled else 'OFF'}`"
)

# ── Header ───────────────────────────────────────────────────────────

st.title("📊 InstaControl View")
st.caption("Real-time observability for the Edge AI Agent")

# ── System Health ────────────────────────────────────────────────────

st.header("System Health")
sys_met = metrics.system_metrics()

col1, col2, col3 = st.columns(3)
col1.metric("CPU Usage", f"{sys_met['cpu_percent']:.1f}%")
col2.metric(
    "RAM Usage",
    f"{sys_met['ram_percent']:.1f}%",
    delta=f"{sys_met['ram_used_mb']:.0f} / {sys_met['ram_total_mb']:.0f} MB",
    delta_color="off",
)
temps = sys_met.get("temperatures", {})
if temps:
    first_temp = list(temps.values())[0]
    col3.metric("Temperature", f"{first_temp:.1f} °C")
else:
    col3.metric("Temperature", "N/A")

# ── Stage Latency ────────────────────────────────────────────────────

st.header("Stage Latency")
stage_lat = metrics.stage_latency(window_minutes)
e2e = metrics.end_to_end_latency(window_minutes)

cols = st.columns(5)
for col, stage in zip(cols[:4], ["SEE", "REASON", "ACT", "AUDIT"]):
    data = stage_lat.get(stage, {"avg_ms": 0, "count": 0})
    col.metric(
        f"{stage} avg",
        f"{data['avg_ms']:.1f} ms",
        delta=f"{data['count']} calls",
        delta_color="off",
    )
cols[4].metric(
    "End-to-End avg",
    f"{e2e['avg_ms']:.1f} ms",
    delta=f"{e2e['count']} cycles",
    delta_color="off",
)

# ── Time Series ──────────────────────────────────────────────────────

st.header("Latency Over Time")
ts_data = metrics.timeseries(window_minutes)
if ts_data:
    df_ts = pd.DataFrame(ts_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ts["timestamp"], y=df_ts["avg_latency_ms"],
        mode="lines+markers", name="Avg Latency (ms)",
    ))
    fig.add_trace(go.Bar(
        x=df_ts["timestamp"], y=df_ts["error_count"],
        name="Errors", yaxis="y2", opacity=0.4,
    ))
    fig.update_layout(
        yaxis=dict(title="Latency (ms)"),
        yaxis2=dict(title="Errors", overlaying="y", side="right"),
        height=350,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No events recorded yet. Run the agent to generate data.")

# ── MatFormer Granularity ────────────────────────────────────────────

st.header("MatFormer Granularity Distribution")
gran = metrics.granularity_distribution(window_minutes)
if gran:
    df_gran = pd.DataFrame([
        {"Granularity": g, "Percentage": v["pct"], "Count": v["count"]}
        for g, v in gran.items()
    ])
    fig = px.pie(
        df_gran, values="Percentage", names="Granularity",
        title="Time Spent in Each Granularity Profile",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=300, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

# ── Safety & Guardrails ─────────────────────────────────────────────

st.header("Safety & Guardrails")
safety = metrics.safety_stats(window_minutes)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Blocked by Policy", safety["blocked_by_policy"])
col2.metric("Auditor Rejections", safety["auditor_rejections"])
col3.metric("Queued Actions", safety["queued_actions"])
col4.metric("HITL Prompts", safety["hitl_prompts"])

# ── Error Rates ──────────────────────────────────────────────────────

st.header("Error Rates by Stage")
err = metrics.error_rate(window_minutes)
if err:
    df_err = pd.DataFrame([
        {
            "Stage": s,
            "Error Rate (%)": v["rate"] * 100,
            "Errors": v["errors"],
            "Total": v["total"],
        }
        for s, v in err.items()
    ])
    fig = px.bar(
        df_err, x="Stage", y="Error Rate (%)",
        text="Errors", color="Stage",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(height=300, margin=dict(t=30), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Memory & Retrieval ───────────────────────────────────────────────

st.header("Memory & Retrieval")
ret = metrics.retrieval_stats(window_minutes)
ckpt = metrics.checkpoint_stats(window_minutes)

col1, col2, col3 = st.columns(3)
col1.metric("Retrieval Calls", ret["retrieval_calls"])
col2.metric("Avg Retrieval Latency", f"{ret['avg_retrieval_latency_ms']:.1f} ms")
col3.metric("Checkpoints / Hour", f"{ckpt['per_hour']:.1f}")

# ── Connectivity ─────────────────────────────────────────────────────

st.header("Connectivity")
conn = metrics.connectivity_stats(window_minutes)
if conn:
    df_conn = pd.DataFrame([
        {"State": s, "Count": c} for s, c in conn.items()
    ])
    fig = px.bar(
        df_conn, x="State", y="Count",
        color="State",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(height=250, margin=dict(t=30), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Event Log ────────────────────────────────────────────────────────

st.header("Event Log (Latest 200)")

fcol1, fcol2 = st.columns(2)
filter_stage = fcol1.selectbox(
    "Filter by stage", ["ALL", "SEE", "REASON", "ACT", "AUDIT"]
)
filter_status = fcol2.selectbox(
    "Filter by decision status", ["ALL", "success", "blocked", "queued", "error"]
)

events = tracer.load_events(
    limit=200,
    stage=filter_stage if filter_stage != "ALL" else None,
    status=filter_status if filter_status != "ALL" else None,
)

if events:
    df_ev = pd.DataFrame(events)
    display_cols = [
        "ts", "stage", "latency_ms", "model_granularity",
        "tokens_in", "tokens_out", "decision", "safety",
    ]
    available = [c for c in display_cols if c in df_ev.columns]
    st.dataframe(df_ev[available], use_container_width=True, height=400)
else:
    st.info("No events match the current filters.")

# ── Auto-refresh ─────────────────────────────────────────────────────
_time.sleep(refresh_rate)
st.rerun()
