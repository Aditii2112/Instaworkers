"""Execute tools from the registry with timing and error handling.

Every tool execution is recorded as a ToolCallRecord for event tracing.
"""

import time
import traceback

from tools.tool_registry import ToolRegistry


class ToolRunner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_name: str, params: dict) -> dict:
        tool = self.registry.get(tool_name)
        if tool is None:
            return {
                "tool": tool_name,
                "status": "error",
                "error": f"Unknown tool: {tool_name}",
                "latency_ms": 0,
            }

        start = time.perf_counter()
        try:
            result = tool.handler(**params)
            latency = (time.perf_counter() - start) * 1000
            return {
                "tool": tool_name,
                "status": "ok",
                "result": result,
                "latency_ms": latency,
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return {
                "tool": tool_name,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "latency_ms": latency,
            }

    def execute_batch(self, tool_calls: list[dict]) -> list[dict]:
        return [
            self.execute(call.get("tool", ""), call.get("params", {}))
            for call in tool_calls
        ]

    @staticmethod
    def to_trace_records(results: list[dict]) -> list[dict]:
        """Convert execution results to event-trace tool_call records."""
        return [
            {
                "name": r["tool"],
                "status": r["status"],
                "latency_ms": r["latency_ms"],
            }
            for r in results
        ]
