"""AUDIT stage — Sovereign Watchdog using Gemma 1B via Ollama.

Policy enforcement (deterministic pre-checks + LLM validation):
  SENT-001 (Containment): Spoilage must trigger quarantine_inventory_sku.
  SENT-002 (Proximity): Block industrial tools if human_present in video frame.
  SENT-003 (Governance): Block 'System Shutdown' unless 3+ separate anomalies detected.

All verdicts (APPROVED/BLOCKED) stream to /api/safety-log in real time.
"""

import json
import time
from pathlib import Path

from config import AppConfig
from llm.gemma_runtime import GemmaRuntime
from observability.event_trace import EventTracer

AUDIT_SYSTEM_PROMPT = """You are the Sovereign Watchdog for a warehouse AI system.
Enforce these safety policies strictly:
- SENT-001 (Containment): Spoilage anomalies MUST trigger quarantine_inventory_sku. BLOCK all actions if spoilage detected but quarantine is missing.
- SENT-002 (Proximity): If human_present is true, BLOCK any industrial tool that could endanger workers (cleaning, chemical, spray, shutdown).
- SENT-003 (Governance): Only approve "System Shutdown" if 3+ separate video-verified anomalies exist.

Evaluate each proposed tool call. Output ONLY valid JSON:
{"approved": bool, "reason": str, "blocked_actions": [indices], "warnings": [str], "policy_checks": [{"policy": "SENT-XXX", "result": "PASS|FAIL", "detail": str}]}
Be conservative. Safety first."""


BLOCKED_WHEN_HUMAN = {
    "run_shell", "industrial_clean", "chemical_spray",
    "system_shutdown", "emergency_shutdown",
}


class AuditorAgent:
    def __init__(self, config: AppConfig, gemma: GemmaRuntime, tracer: EventTracer):
        self.config = config
        self.gemma = gemma
        self.tracer = tracer
        self._safety_log_path = Path(config.observability.events_path).parent / "safety-log.jsonl"
        self._safety_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._anomaly_count = 0

    def set_anomaly_count(self, count: int):
        """Track video-verified anomaly count for SENT-003."""
        self._anomaly_count = count

    def audit(
        self,
        tool_calls: list[dict],
        system_state: dict | None = None,
        session_id: str = "",
        correlation_id: str = "",
    ) -> dict:
        if not tool_calls:
            verdict = {
                "approved": True,
                "reason": "No actions to audit",
                "blocked_actions": [],
                "warnings": [],
                "policy_checks": [],
            }
            self._append_safety_log(verdict, tool_calls, session_id, correlation_id)
            return verdict

        system_state = system_state or {}
        human_present = system_state.get("human_present", False)
        observations = system_state.get("observations", {})
        anomaly = observations.get("anomaly", "none")

        blocked_indices = set()
        warnings = []
        policy_checks = []

        # ── SENT-001: Spoilage → quarantine required ─────────────────
        has_quarantine = any(
            "quarantine" in (c.get("tool") or "").lower()
            for c in tool_calls
        )
        if anomaly == "spoilage":
            if has_quarantine:
                policy_checks.append({"policy": "SENT-001", "result": "PASS", "detail": "Quarantine present for spoilage"})
            else:
                policy_checks.append({"policy": "SENT-001", "result": "FAIL", "detail": "Spoilage requires quarantine_inventory_sku"})
                warnings.append("SENT-001: Spoilage detected — quarantine_inventory_sku REQUIRED but missing")
                for i, tc in enumerate(tool_calls):
                    if "quarantine" not in (tc.get("tool") or "").lower():
                        blocked_indices.add(i)
        else:
            policy_checks.append({"policy": "SENT-001", "result": "PASS", "detail": "No spoilage detected"})

        # ── SENT-002: Human present → block hazardous tools ──────────
        if human_present:
            for i, tc in enumerate(tool_calls):
                tool_name = tc.get("tool", "")
                if tool_name in BLOCKED_WHEN_HUMAN or "clean" in tool_name.lower() or "shutdown" in tool_name.lower():
                    blocked_indices.add(i)
                    warnings.append(f"SENT-002: Blocked {tool_name} — human present in frame")
                    policy_checks.append({"policy": "SENT-002", "result": "FAIL", "detail": f"Blocked {tool_name} due to human presence"})
            if not any(pc["policy"] == "SENT-002" for pc in policy_checks):
                policy_checks.append({"policy": "SENT-002", "result": "PASS", "detail": "Human present but no hazardous tools requested"})
        else:
            policy_checks.append({"policy": "SENT-002", "result": "PASS", "detail": "No human detected in frame"})

        # ── SENT-003: System Shutdown requires 3+ anomalies ──────────
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get("tool", "")
            if "shutdown" in tool_name.lower():
                if self._anomaly_count < 3:
                    blocked_indices.add(i)
                    warnings.append(
                        f"SENT-003: System Shutdown requires 3+ anomalies (current: {self._anomaly_count})"
                    )
                    policy_checks.append({
                        "policy": "SENT-003", "result": "FAIL",
                        "detail": f"Shutdown blocked — only {self._anomaly_count} anomalies detected (need 3+)",
                    })
                else:
                    policy_checks.append({
                        "policy": "SENT-003", "result": "PASS",
                        "detail": f"Shutdown approved — {self._anomaly_count} anomalies verified",
                    })

        if not any(pc["policy"] == "SENT-003" for pc in policy_checks):
            policy_checks.append({"policy": "SENT-003", "result": "PASS", "detail": "No shutdown requested"})

        # If deterministic checks blocked everything, skip LLM
        if blocked_indices == set(range(len(tool_calls))):
            verdict = {
                "approved": False,
                "reason": "; ".join(warnings),
                "blocked_actions": sorted(blocked_indices),
                "warnings": warnings,
                "policy_checks": policy_checks,
            }
            self._append_safety_log(verdict, tool_calls, session_id, correlation_id)
            return verdict

        # ── LLM audit for remaining calls ────────────────────────────
        prompt = self._build_prompt(tool_calls, system_state)
        start = time.perf_counter()
        try:
            result = self.gemma.generate(
                prompt=prompt,
                system_prompt=AUDIT_SYSTEM_PROMPT,
                max_tokens=256,
                temperature=0.1,
                model_name=self.config.auditor.model_name,
            )
            llm_verdict = self._parse_verdict(result.get("text", ""))
        except Exception as e:
            llm_verdict = {
                "approved": True,
                "reason": f"LLM audit unavailable ({e}); deterministic checks applied",
                "blocked_actions": [],
                "warnings": [],
            }
        latency_ms = (time.perf_counter() - start) * 1000

        all_blocked = sorted(set(llm_verdict.get("blocked_actions", [])) | blocked_indices)
        all_warnings = warnings + llm_verdict.get("warnings", [])
        approved = len(all_blocked) == 0

        verdict = {
            "approved": approved,
            "reason": llm_verdict.get("reason", "") or "; ".join(all_warnings) or "All policies passed",
            "blocked_actions": all_blocked,
            "warnings": all_warnings,
            "policy_checks": policy_checks,
        }

        status = "success" if approved else "blocked"
        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="AUDIT",
            latency_ms=latency_ms,
            model=self.config.auditor.model_name,
            decision={
                "type": "audit",
                "status": status,
                "reason": verdict["reason"],
            },
            safety={"blocked": not approved, "policy": verdict["reason"]},
        )

        self._append_safety_log(verdict, tool_calls, session_id, correlation_id)
        return verdict

    def _append_safety_log(
        self,
        verdict: dict,
        tool_calls: list,
        session_id: str,
        correlation_id: str,
    ):
        record = {
            "timestamp": time.time(),
            "verdict": "APPROVED" if verdict["approved"] else "BLOCKED",
            "reason": verdict.get("reason", ""),
            "blocked_actions": verdict.get("blocked_actions", []),
            "warnings": verdict.get("warnings", []),
            "policy_checks": verdict.get("policy_checks", []),
            "session_id": session_id,
            "correlation_id": correlation_id,
            "tool_count": len(tool_calls),
            "tools": [tc.get("tool", "") for tc in tool_calls],
        }
        with open(self._safety_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _build_prompt(tool_calls: list[dict], system_state: dict) -> str:
        return (
            f"## Proposed Tool Calls\n{json.dumps(tool_calls, indent=2)}\n\n"
            f"## Current System State\n{json.dumps(system_state, indent=2, default=str)}\n\n"
            "Evaluate each action for safety and policy compliance. "
            'Output JSON: {"approved": bool, "reason": str, '
            '"blocked_actions": [indices], "warnings": [str], '
            '"policy_checks": [{"policy": "SENT-XXX", "result": "PASS|FAIL", "detail": str}]}'
        )

    @staticmethod
    def _parse_verdict(text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                verdict = json.loads(text[start:end])
                verdict.setdefault("approved", True)
                verdict.setdefault("reason", "")
                verdict.setdefault("blocked_actions", [])
                verdict.setdefault("warnings", [])
                return verdict
        except json.JSONDecodeError:
            pass
        return {
            "approved": True,
            "reason": "Could not parse audit response; deterministic checks applied",
            "blocked_actions": [],
            "warnings": [text[:200] if text else "empty response"],
        }
