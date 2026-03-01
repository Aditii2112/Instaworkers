"""Sovereign Sentinel — main entry point.

Initializes all subsystems (Gemma, InstaBrain, tools, observability),
verifies connectivity, and starts the interactive See → Reason → Act → Audit loop.
"""

import sys
from pathlib import Path

import httpx

from config import load_config, ROOT_DIR
from llm.gemma_runtime import GemmaRuntime
from llm.gemini_runtime import GeminiRuntime
from memory.embeddings import EmbeddingEngine
from memory.instabrain_db import InstaBrainDB
from observability.event_trace import EventTracer
from tools.tool_registry import ToolRegistry, ToolDefinition
from tools.tool_runner import ToolRunner
from tools.adapters import filesystem, http_client, shell_allowlist
from tools.industrial_tools import (
    dispatch_visual_ticket,
    quarantine_inventory_sku,
    broadcast_safety_alert,
)
from agents.orchestrator import RootOrchestrator


def register_industrial_tools(registry: ToolRegistry) -> None:
    """Register industrial warehouse tools: dispatch, quarantine, broadcast."""
    registry.register(ToolDefinition(
        name="dispatch_visual_ticket",
        description="Route anomaly to Facilities (leak), Inventory (spoilage), or Pest Control (pest). Captures the exact video frame as .jpg and saves ticket.json to /alerts/.",
        parameters={
            "type": "object",
            "properties": {
                "anomaly": {"type": "string", "description": "leak|pest|spoilage|spill|damage|hazard"},
                "severity": {"type": "string", "description": "critical|high|medium|low"},
                "location": {"type": "string", "description": "e.g. Aisle_4, Rack_04, Loading_Dock"},
                "image_path": {"type": "string", "description": "Path to captured frame"},
                "video_path": {"type": "string", "description": "Path to source video"},
                "frame_index": {"type": "integer", "description": "Frame index in video for exact capture"},
                "team": {"type": "string", "description": "Override team (optional)"},
            },
            "required": ["anomaly", "severity", "location"],
        },
        handler=lambda anomaly, severity, location, image_path=None, video_path=None, frame_index=None, team=None: dispatch_visual_ticket(
            anomaly=anomaly, severity=severity, location=location,
            image_path=image_path, video_path=video_path, frame_index=frame_index, team=team,
        ),
        tags=["industrial", "warehouse"],
    ))
    registry.register(ToolDefinition(
        name="quarantine_inventory_sku",
        description="Set inventory SKU status to QUARANTINED in inventory.db. Used when spoilage or contamination detected.",
        parameters={
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "location": {"type": "string", "description": "Optional filter by location"},
            },
            "required": ["sku"],
        },
        handler=lambda sku, location=None: quarantine_inventory_sku(sku=sku, location=location),
        tags=["industrial", "warehouse"],
    ))
    registry.register(ToolDefinition(
        name="broadcast_safety_alert",
        description="Announce hazard via MacOS say (TTS). Dispatched team and location are announced audibly.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "team": {"type": "string", "description": "Team dispatched"},
                "message": {"type": "string", "description": "Optional custom message"},
            },
            "required": ["location", "team"],
        },
        handler=lambda location, team, message=None: broadcast_safety_alert(
            location=location, team=team, message=message
        ),
        tags=["industrial", "safety"],
    ))


def register_baseline_tools(registry: ToolRegistry, config) -> None:
    """Register problem-agnostic baseline tools."""
    base_dir = config.tools.fs_base_dir

    registry.register(ToolDefinition(
        name="read_file",
        description="Read a file from the restricted data directory",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within data dir"},
            },
            "required": ["path"],
        },
        handler=lambda path: filesystem.read_file(path, base_dir=base_dir),
        tags=["filesystem"],
    ))

    registry.register(ToolDefinition(
        name="write_file",
        description="Write content to a file in the restricted data directory",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within data dir"},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["path", "content"],
        },
        handler=lambda path, content: filesystem.write_file(path, content, base_dir=base_dir),
        requires_approval=True,
        tags=["filesystem"],
    ))

    registry.register(ToolDefinition(
        name="list_directory",
        description="List contents of a directory in the restricted data directory",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path (default: '.')"},
            },
        },
        handler=lambda path=".": filesystem.list_directory(path, base_dir=base_dir),
        tags=["filesystem"],
    ))

    allowlist_domains = config.tools.http_allowlist

    registry.register(ToolDefinition(
        name="http_get",
        description="HTTP GET request (restricted to allowlisted domains)",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "headers": {"type": "object", "default": {}},
            },
            "required": ["url"],
        },
        handler=lambda url, headers=None: http_client.http_get(
            url, headers=headers, allowlist=allowlist_domains
        ),
        tags=["http"],
    ))

    registry.register(ToolDefinition(
        name="http_post",
        description="HTTP POST request (restricted to allowlisted domains)",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "body": {"type": "object", "default": {}},
                "headers": {"type": "object", "default": {}},
            },
            "required": ["url"],
        },
        handler=lambda url, body=None, headers=None: http_client.http_post(
            url, body=body, headers=headers, allowlist=allowlist_domains
        ),
        tags=["http"],
    ))

    shell_cmds = config.tools.shell_allowlist

    registry.register(ToolDefinition(
        name="run_shell",
        description="Run a shell command (restricted to allowlisted commands only)",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
        handler=lambda command: shell_allowlist.run_shell(
            command, allowlist=shell_cmds
        ),
        requires_approval=True,
        tags=["shell"],
    ))


def check_connectivity() -> str:
    try:
        resp = httpx.get("https://httpbin.org/get", timeout=5)
        return "online" if resp.status_code == 200 else "unknown"
    except Exception:
        return "offline"


def main():
    print("=" * 64)
    print("   Sovereign Sentinel  —  Facility Intelligence System")
    print("   PaliGemma 2 (3B) + Gemma 1B Auditor | On-Device")
    print("=" * 64)

    config = load_config()

    Path(config.instabrain.db_path).parent.mkdir(parents=True, exist_ok=True)

    print("\n[INIT] Connecting to LLM runtime...")
    gemma = GemmaRuntime(config)
    gemma.health_check()
    print(f"[INIT] LLM OK at {config.gemma.base_url}")
    gemini = GeminiRuntime(config)

    print(f"[INIT] Loading embedding model: {config.instabrain.embedding_model}")
    emb = EmbeddingEngine(config.instabrain.embedding_model)
    print(f"[INIT] Embeddings OK (dim={emb.dim})")

    print(f"[INIT] Opening InstaBrain: {config.instabrain.db_path}")
    db = InstaBrainDB(config.instabrain.db_path, emb)
    vec_mode = "sqlite-vec" if db._use_vec_ext else "numpy fallback"
    print(f"[INIT] InstaBrain OK ({vec_mode})")

    tracer = EventTracer(config.observability.events_path)
    print(f"[INIT] Event tracing -> {config.observability.events_path}")

    registry = ToolRegistry()
    runner = ToolRunner(registry)
    register_baseline_tools(registry, config)
    register_industrial_tools(registry)
    print(f"[INIT] Tools registered: {registry.list_names()}")

    orchestrator = RootOrchestrator(
        config=config, gemma=gemma, db=db,
        tracer=tracer, registry=registry, runner=runner, gemini=gemini,
    )

    print(f"\n[INIT] MatFormer granularity : {config.matformer.granularity}")
    print(f"[INIT] Video FPS target     : {config.video.fps_target}")
    print(f"[INIT] VLM model            : {config.vlm.model_path}")
    print(f"[INIT] Auditor (Watchdog)   : {'ON (' + config.auditor.model_name + ')' if config.auditor.enabled else 'OFF'}")
    print(f"[INIT] Hybrid reliability   : {'ON' if config.hybrid.enabled else 'OFF'}")
    print(f"[INIT] Gemini teacher       : {gemini.connectivity_state()}")

    conn = check_connectivity()
    print(f"[INIT] Connectivity         : {conn}")

    print("\n" + "=" * 64)
    print("  Sovereign Sentinel ready. Type commands or 'quit' to exit.")
    print("  Dashboard: http://localhost:3000")
    print("=" * 64 + "\n")

    try:
        while True:
            try:
                user_input = input(">>> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            cycle = orchestrator.run_cycle(text_input=user_input)

            print(f"\n--- Cycle {orchestrator._cycle_count} "
                  f"[{cycle['correlation_id'][:8]}] ---")

            obs = cycle.get("observations", {})
            if obs.get("anomaly", "none") != "none":
                print(f"  Anomaly  : {obs['anomaly']} ({obs.get('severity', '?')}) @ {obs.get('location', '?')}")
                print(f"  Human    : {obs.get('human_present', False)}")

            plan = cycle.get("plan", {})
            print(f"  Analysis : {plan.get('analysis', 'N/A')}")

            tool_calls = cycle.get("tool_calls", [])
            if tool_calls:
                print(f"  Tool calls ({len(tool_calls)}):")
                for tr in cycle.get("tool_results", []):
                    icon = "OK" if tr["status"] == "ok" else "ERR"
                    print(f"    [{icon}] {tr['tool']}")

            if "audit" in cycle:
                audit = cycle["audit"]
                verdict = "APPROVED" if audit["approved"] else "BLOCKED"
                print(f"  Audit: {verdict} — {audit.get('reason', '')}")
                for pc in audit.get("policy_checks", []):
                    print(f"    {pc['policy']}: {pc['result']} — {pc.get('detail', '')}")

            print()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        orchestrator.close()
        print("Sovereign Sentinel stopped.")


if __name__ == "__main__":
    main()
