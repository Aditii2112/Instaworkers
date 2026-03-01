"""ACT stage — converts plans into executable tool calls.

When the REASON stage produced deterministic steps (from VLM anomaly detection),
this agent converts them directly without LLM. Falls back to gemma3:1b for
ambiguous plans.
"""

import json
import time

from config import AppConfig
from llm.gemma_runtime import GemmaRuntime
from llm.prompts import ACT_SYSTEM_PROMPT
from observability.event_trace import EventTracer


class ActionAgent:
    def __init__(self, config: AppConfig, gemma: GemmaRuntime, tracer: EventTracer):
        self.config = config
        self.gemma = gemma
        self.tracer = tracer

    def act(
        self,
        plan: dict,
        tool_descriptions: str,
        session_id: str = "",
        correlation_id: str = "",
        profile_override: dict | None = None,
        granularity_override: str | None = None,
    ) -> list[dict]:
        start = time.perf_counter()

        steps = plan.get("plan", [])
        if steps:
            tool_calls = self._deterministic_convert(steps)
        else:
            tool_calls = self._llm_convert(plan, tool_descriptions)

        latency = (time.perf_counter() - start) * 1000

        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="ACT",
            latency_ms=latency,
            decision={
                "type": "tool_generation",
                "status": "success",
                "reason": f"Generated {len(tool_calls)} tool calls",
            },
        )

        return tool_calls

    @staticmethod
    def _deterministic_convert(steps: list[dict]) -> list[dict]:
        """Convert structured plan steps directly into tool calls."""
        calls = []
        for step in steps:
            tool_name = step.get("tool_name") or step.get("tool")
            if not tool_name:
                continue
            params = step.get("parameters") or step.get("params") or {}
            calls.append({"tool": tool_name, "params": params})
        return calls

    def _llm_convert(self, plan: dict, tool_descriptions: str) -> list[dict]:
        """Fallback: use gemma3:1b to generate tool calls from plan text."""
        prompt = (
            f"## Plan\n{json.dumps(plan, indent=2, default=str)}\n\n"
            f"## Available Tools\n{tool_descriptions}\n\n"
            'Convert into JSON array: [{"tool": "name", "params": {...}}]\n'
            "Output ONLY the JSON array. No markdown fences."
        )

        try:
            result = self.gemma.generate(
                prompt=prompt,
                system_prompt=ACT_SYSTEM_PROMPT,
                max_tokens=512,
                temperature=0.2,
                model_name=self.config.auditor.model_name,  # gemma3:1b
            )
            return self._parse_tool_calls(result["text"])
        except Exception:
            return []

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                calls = json.loads(text[start:end])
                if isinstance(calls, list):
                    return calls
        except json.JSONDecodeError:
            pass
        return []
