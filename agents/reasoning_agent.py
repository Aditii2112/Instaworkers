"""REASON stage — analyzes observations and generates a structured action plan.

Uses deterministic logic when the VLM has already identified a clear anomaly
(leak/pest/spoilage). Falls back to Gemma 1B LLM for ambiguous situations.
The fine-tuned sovereign-sentinel model is vision-only — REASON/ACT use gemma3:1b.
"""

import json
import time

from config import AppConfig
from llm.gemma_runtime import GemmaRuntime
from llm.prompts import REASON_SYSTEM_PROMPT
from observability.event_trace import EventTracer

TEAM_ROUTING = {
    "leak": "Facilities",
    "spoilage": "Inventory",
    "pest": "Pest Control",
    "spill": "Facilities",
    "damage": "Facilities",
    "hazard": "Facilities",
    "fire": "Facilities",
    "smoke": "Facilities",
}


class ReasoningAgent:
    def __init__(self, config: AppConfig, gemma: GemmaRuntime, tracer: EventTracer):
        self.config = config
        self.gemma = gemma
        self.tracer = tracer

    def reason(
        self,
        observations: dict,
        context: dict,
        tool_descriptions: str,
        session_id: str = "",
        correlation_id: str = "",
        profile_override: dict | None = None,
        granularity_override: str | None = None,
    ) -> dict:
        start = time.perf_counter()

        anomaly = observations.get("anomaly", "none")
        severity = observations.get("severity", "low")
        location = observations.get("location", "unknown")
        human_present = observations.get("human_present", False)

        # Deterministic planning when VLM identified anomalies
        all_anomaly_types = observations.get("anomaly_types_seen", [])
        if not all_anomaly_types and anomaly != "none":
            all_anomaly_types = [anomaly]

        if all_anomaly_types:
            plan = self._deterministic_plan_multi(
                all_anomaly_types, severity, location, human_present
            )
            latency = (time.perf_counter() - start) * 1000
            self.tracer.create_event(
                session_id=session_id,
                correlation_id=correlation_id,
                stage="REASON",
                latency_ms=latency,
                decision={"type": "plan", "status": "success", "reason": "deterministic"},
            )
            return plan

        # LLM fallback for ambiguous situations — use gemma3:1b, NOT the vision model
        plan = self._llm_plan(observations, context, tool_descriptions, profile_override)
        latency = (time.perf_counter() - start) * 1000

        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="REASON",
            latency_ms=latency,
            decision={"type": "plan", "status": "success", "reason": "llm"},
        )
        return plan

    def _deterministic_plan_multi(
        self, anomaly_types: list[str], severity: str, location: str, human_present: bool
    ) -> dict:
        """Build a plan covering ALL detected anomaly types across video frames."""
        steps = []
        priority = 1
        teams_dispatched = set()
        analysis_parts = []

        for anomaly in anomaly_types:
            team = TEAM_ROUTING.get(anomaly.lower(), "Facilities")

            steps.append({
                "tool_name": "dispatch_visual_ticket",
                "parameters": {
                    "anomaly": anomaly,
                    "severity": severity,
                    "location": location,
                    "team": team,
                },
                "priority": priority,
            })
            priority += 1
            teams_dispatched.add(team)

            if anomaly == "spoilage":
                steps.append({
                    "tool_name": "quarantine_inventory_sku",
                    "parameters": {"sku": "*", "location": location},
                    "priority": priority,
                })
                priority += 1

            analysis_parts.append(f"{anomaly} → {team}")

        if human_present and severity in ("critical", "high"):
            for team in teams_dispatched:
                steps.append({
                    "tool_name": "broadcast_safety_alert",
                    "parameters": {"location": location, "team": team},
                    "priority": priority,
                })
                priority += 1

        anomaly_str = ", ".join(anomaly_types)
        analysis = f"Detected: {anomaly_str} at {location} (severity: {severity}). "
        analysis += "Dispatching: " + ", ".join(analysis_parts) + "."
        if human_present:
            analysis += " Worker present — safety alerts issued."

        return {
            "analysis": analysis,
            "plan": steps,
            "rationale": f"VLM identified {len(anomaly_types)} anomaly type(s): {anomaly_str}. "
                         f"Routing to respective teams per Sentinel protocol.",
            "delegate_to": "action_agent",
        }

    def _llm_plan(
        self, observations: dict, context: dict, tool_descriptions: str,
        profile_override: dict | None,
    ) -> dict:
        """Use Gemma 1B (not the vision model) for ambiguous situations."""
        profile = profile_override or self.config.matformer.profile
        prompt = self._build_prompt(observations, context, tool_descriptions)

        try:
            result = self.gemma.generate(
                prompt=prompt,
                system_prompt=REASON_SYSTEM_PROMPT,
                max_tokens=512,
                temperature=0.3,
                model_name=self.config.auditor.model_name,  # gemma3:1b
            )
            return self._parse_plan(result["text"])
        except Exception as e:
            return {
                "analysis": f"LLM reasoning unavailable: {e}",
                "plan": [],
                "rationale": "Fallback — no actions generated",
                "delegate_to": "action_agent",
            }

    def _build_prompt(
        self, observations: dict, context: dict, tool_descriptions: str
    ) -> str:
        obs_summary = {
            "anomaly": observations.get("anomaly", "none"),
            "severity": observations.get("severity", "low"),
            "location": observations.get("location", "unknown"),
            "human_present": observations.get("human_present", False),
            "source": observations.get("source", "none"),
        }

        parts = [
            "## Observations",
            json.dumps(obs_summary, indent=2),
            f"\n## Available Tools\n{tool_descriptions}",
            "\n## Instructions",
            "Analyze the situation and output ONLY valid JSON with keys: "
            "analysis, plan (list of {tool_name, parameters, priority}), rationale.",
            "Keep analysis to ONE sentence. No markdown fences.",
        ]

        return "\n".join(parts)

    @staticmethod
    def _parse_plan(text: str) -> dict:
        cleaned = (text or "").strip()

        for attempt in [cleaned, _strip_fences(cleaned), _extract_json(cleaned)]:
            if not attempt:
                continue
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    parsed.setdefault("analysis", text[:200])
                    parsed.setdefault("plan", [])
                    parsed.setdefault("rationale", "")
                    parsed.setdefault("delegate_to", "action_agent")
                    return parsed
            except json.JSONDecodeError:
                continue

        return {
            "analysis": text[:200] if text else "No analysis",
            "plan": [],
            "rationale": "Could not parse structured plan from LLM output",
            "delegate_to": "action_agent",
            "raw_output": text,
        }


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return ""


def _extract_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""
