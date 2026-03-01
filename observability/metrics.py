"""Rolling metrics aggregation from event trace + system telemetry.

Reads the JSONL event log, computes per-stage latency, error rates,
granularity distribution, safety stats, retrieval stats, checkpoint
rates, and connectivity summaries. Also collects live CPU/RAM/temp
via psutil.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil


class MetricsCollector:
    def __init__(self, events_path: str):
        self.events_path = Path(events_path)

    # ── event loading ────────────────────────────────────────────────

    def _load_events(self, since_minutes: int = 60) -> list[dict]:
        if not self.events_path.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
        events: list[dict] = []
        with open(self.events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts = datetime.fromisoformat(ev["ts"])
                    if ts >= cutoff:
                        events.append(ev)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return events

    # ── system telemetry ─────────────────────────────────────────────

    def system_metrics(self) -> dict:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        temps: dict = {}
        try:
            t = psutil.sensors_temperatures()
            if t:
                for name, entries in t.items():
                    if entries:
                        temps[name] = entries[0].current
        except (AttributeError, RuntimeError):
            pass

        return {
            "cpu_percent": cpu,
            "ram_used_mb": mem.used / (1024 * 1024),
            "ram_total_mb": mem.total / (1024 * 1024),
            "ram_percent": mem.percent,
            "temperatures": temps,
        }

    # ── stage latency ────────────────────────────────────────────────

    def stage_latency(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        by_stage: dict[str, list[float]] = defaultdict(list)
        for ev in events:
            by_stage[ev["stage"]].append(ev["latency_ms"])

        return {
            stage: {
                "avg_ms": sum(lats) / len(lats),
                "min_ms": min(lats),
                "max_ms": max(lats),
                "count": len(lats),
            }
            for stage, lats in by_stage.items()
        }

    def end_to_end_latency(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        by_corr: dict[str, float] = defaultdict(float)
        for ev in events:
            by_corr[ev["correlation_id"]] += ev["latency_ms"]

        if not by_corr:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "count": 0}

        vals = list(by_corr.values())
        return {
            "avg_ms": sum(vals) / len(vals),
            "min_ms": min(vals),
            "max_ms": max(vals),
            "count": len(vals),
        }

    # ── error rate ───────────────────────────────────────────────────

    def error_rate(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        by_stage: dict[str, dict] = defaultdict(lambda: {"total": 0, "errors": 0})
        for ev in events:
            stage = ev["stage"]
            by_stage[stage]["total"] += 1
            if ev.get("decision", {}).get("status") == "error":
                by_stage[stage]["errors"] += 1

        return {
            stage: {
                **counts,
                "rate": counts["errors"] / counts["total"] if counts["total"] else 0,
            }
            for stage, counts in by_stage.items()
        }

    # ── MatFormer granularity ────────────────────────────────────────

    def granularity_distribution(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        counts: dict[str, int] = defaultdict(int)
        total = 0
        for ev in events:
            g = ev.get("model_granularity", "M")
            counts[g] += 1
            total += 1

        return {
            g: {"count": c, "pct": c / total * 100 if total else 0}
            for g, c in counts.items()
        }

    # ── safety / guardrails ──────────────────────────────────────────

    def safety_stats(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        blocked = 0
        auditor_rejects = 0
        queued = 0
        hitl_count = 0

        for ev in events:
            if ev.get("safety", {}).get("blocked"):
                blocked += 1
            decision = ev.get("decision", {})
            if decision.get("status") == "blocked":
                auditor_rejects += 1
            if decision.get("status") == "queued":
                queued += 1
            if decision.get("type") == "hitl_prompt":
                hitl_count += 1

        return {
            "blocked_by_policy": blocked,
            "auditor_rejections": auditor_rejects,
            "queued_actions": queued,
            "hitl_prompts": hitl_count,
        }

    # ── retrieval stats ──────────────────────────────────────────────

    def retrieval_stats(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        retrieval_calls = 0
        retrieval_latencies: list[float] = []

        for ev in events:
            for tc in ev.get("tool_calls", []):
                name = tc.get("name", "").lower()
                if "retriev" in name or "query" in name:
                    retrieval_calls += 1
                    retrieval_latencies.append(tc.get("latency_ms", 0))

        return {
            "retrieval_calls": retrieval_calls,
            "avg_retrieval_latency_ms": (
                sum(retrieval_latencies) / len(retrieval_latencies)
                if retrieval_latencies
                else 0
            ),
        }

    # ── checkpoint stats ─────────────────────────────────────────────

    def checkpoint_stats(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        ckpt_count = sum(
            1 for ev in events
            if ev.get("decision", {}).get("type") == "checkpoint"
        )
        hours = since_minutes / 60
        return {
            "checkpoints": ckpt_count,
            "per_hour": ckpt_count / hours if hours > 0 else 0,
        }

    # ── connectivity ─────────────────────────────────────────────────

    def connectivity_stats(self, since_minutes: int = 60) -> dict:
        events = self._load_events(since_minutes)
        states: dict[str, int] = defaultdict(int)
        for ev in events:
            st = ev.get("connectivity", {}).get("state", "unknown")
            states[st] += 1
        return dict(states)

    # ── time series (for dashboard charts) ───────────────────────────

    def timeseries(
        self, since_minutes: int = 60, bucket_seconds: int = 60
    ) -> list[dict]:
        events = self._load_events(since_minutes)
        if not events:
            return []

        buckets: dict[str, dict] = defaultdict(
            lambda: {"latencies": [], "errors": 0, "count": 0}
        )
        for ev in events:
            ts = datetime.fromisoformat(ev["ts"])
            bucket_ts = ts.replace(
                second=(ts.second // bucket_seconds) * bucket_seconds,
                microsecond=0,
            )
            key = bucket_ts.isoformat()
            buckets[key]["count"] += 1
            buckets[key]["latencies"].append(ev["latency_ms"])
            if ev.get("decision", {}).get("status") == "error":
                buckets[key]["errors"] += 1

        return [
            {
                "timestamp": ts_key,
                "avg_latency_ms": sum(b["latencies"]) / len(b["latencies"]),
                "event_count": b["count"],
                "error_count": b["errors"],
            }
            for ts_key, b in sorted(buckets.items())
        ]
