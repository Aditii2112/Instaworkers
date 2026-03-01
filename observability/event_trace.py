"""JSONL structured event tracing for every agent stage.

Every SEE / REASON / ACT / AUDIT step emits one AgentEvent.
Events are appended to a JSONL file for the dashboard and offline analysis.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


@dataclass
class AgentEvent:
    ts: str
    correlation_id: str
    session_id: str
    stage: str
    latency_ms: float
    model: str = ""
    model_granularity: str = "M"
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_calls: list = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    safety: dict = field(default_factory=lambda: {"blocked": False, "policy": None})
    connectivity: dict = field(default_factory=lambda: {"state": "unknown"})


class EventTracer:
    def __init__(self, events_path: str):
        self.events_path = Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: AgentEvent):
        """Append a single event to the JSONL file (thread-safe)."""
        with self._lock:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")

    def create_event(
        self,
        session_id: str,
        correlation_id: str,
        stage: str,
        latency_ms: float,
        **kwargs,
    ) -> AgentEvent:
        """Create, emit, and return an event in one call."""
        event = AgentEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
            session_id=session_id,
            stage=stage,
            latency_ms=latency_ms,
            **kwargs,
        )
        self.emit(event)
        return event

    def load_events(
        self,
        limit: int = 200,
        stage: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Load recent events from the JSONL file with optional filters."""
        if not self.events_path.exists():
            return []

        events: list[dict] = []
        with open(self.events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if stage and ev.get("stage") != stage:
                        continue
                    if status and ev.get("decision", {}).get("status") != status:
                        continue
                    events.append(ev)
                except json.JSONDecodeError:
                    continue

        return events[-limit:]
