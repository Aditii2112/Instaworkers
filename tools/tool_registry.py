"""Pluggable tool registry with JSON schema definitions.

Tools are registered at startup. When the problem statement arrives,
add new tools by calling registry.register(ToolDefinition(...)).
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Any]
    requires_approval: bool = False
    tags: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "requires_approval": t.requires_approval,
                "tags": t.tags,
            }
            for t in self._tools.values()
        ]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_schema_prompt(self) -> str:
        """Format all tools into a prompt-friendly string for the LLM."""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"\n- {t.name}: {t.description}")
            lines.append(f"  Parameters: {t.parameters}")
        return "\n".join(lines)
