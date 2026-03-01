"""
Hackathon Test Runner — end-to-end pipeline test with real Gemma.

What this does:
  1. Generates a synthetic test image (multimodal input)
  2. Seeds fake domain data into InstaBrain
  3. Registers domain-specific tools alongside baseline tools
  4. Runs the REAL See → Reason → Act → Audit pipeline
  5. Prints full cycle results

Usage:
  cd edge_agent_root
  python -m hackathon_test.run_test

At the hackathon, swap scenario.py with your actual problem and rerun.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from llm.gemma_runtime import GemmaRuntime
from memory.embeddings import EmbeddingEngine
from memory.instabrain_db import InstaBrainDB
from observability.event_trace import EventTracer
from tools.tool_registry import ToolRegistry
from tools.tool_runner import ToolRunner
from agents.orchestrator import RootOrchestrator

from hackathon_test.scenario import (
    FAKE_DATA,
    PROBLEM_INPUTS,
    register_domain_tools,
)
from hackathon_test.generate_test_image import generate_warehouse_image

# bring in baseline tools registration from main
from main import register_baseline_tools


def print_banner(text: str):
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def seed_fake_data(db: InstaBrainDB):
    """Seed all fake data records into InstaBrain for context retrieval."""
    print(f"\n[SEED] Inserting {len(FAKE_DATA)} fake data records into InstaBrain...")
    for i, record in enumerate(FAKE_DATA):
        row_id = db.insert_memory(record, metadata={"source": "hackathon_test", "index": i})
        preview = record[:60] + "..." if len(record) > 60 else record
        print(f"  [{i+1:2d}] id={row_id} | {preview}")
    print(f"[SEED] Done. {len(FAKE_DATA)} records seeded.\n")


def run():
    print_banner("HACKATHON TEST RUNNER — Real Pipeline")
    print("  Problem: Warehouse Safety Inspector")
    print("  Input:   Multimodal (image + text)")
    print("  LLM:     Real Gemma via llama.cpp")

    # ── Config ────────────────────────────────────────────────────────
    config = load_config()
    Path(config.instabrain.db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Generate test image ───────────────────────────────────────────
    print("\n[IMG] Generating synthetic warehouse image...")
    image_path = generate_warehouse_image()
    print(f"[IMG] Saved to: {image_path}")

    # ── Gemma runtime (REAL) ──────────────────────────────────────────
    print(f"\n[INIT] Connecting to Gemma at {config.gemma.base_url}...")
    gemma = GemmaRuntime(config)
    gemma.health_check()
    print("[INIT] Gemma OK")

    # ── Embeddings ────────────────────────────────────────────────────
    print(f"[INIT] Loading embeddings: {config.instabrain.embedding_model}")
    emb = EmbeddingEngine(config.instabrain.embedding_model)
    print(f"[INIT] Embeddings OK (dim={emb.dim})")

    # ── InstaBrain ────────────────────────────────────────────────────
    db = InstaBrainDB(config.instabrain.db_path, emb)
    vec_mode = "sqlite-vec" if db._use_vec_ext else "numpy fallback"
    print(f"[INIT] InstaBrain OK ({vec_mode})")

    # ── Seed fake data ────────────────────────────────────────────────
    seed_fake_data(db)

    # ── Observability ─────────────────────────────────────────────────
    tracer = EventTracer(config.observability.events_path)

    # ── Tool registry (baseline + domain) ─────────────────────────────
    registry = ToolRegistry()
    runner = ToolRunner(registry)
    register_baseline_tools(registry, config)
    register_domain_tools(registry)
    print(f"[INIT] All tools registered: {registry.list_names()}")

    # ── Orchestrator ──────────────────────────────────────────────────
    orchestrator = RootOrchestrator(
        config=config,
        gemma=gemma,
        db=db,
        tracer=tracer,
        registry=registry,
        runner=runner,
    )

    print(f"\n[INIT] MatFormer granularity : {config.matformer.granularity}")
    print(f"[INIT] Checkpointing        : {'ON' if config.checkpoint.enabled else 'OFF'}")
    print(f"[INIT] Auditor (Watchdog)    : {'ON' if config.auditor.enabled else 'OFF'}")

    # ── Run the cycle ─────────────────────────────────────────────────
    print_banner("RUNNING PIPELINE: See → Reason → Act → Audit")

    text_input = PROBLEM_INPUTS["text_input"]
    print(f"\n[INPUT] Text:  {text_input}")
    print(f"[INPUT] Image: {image_path}")

    cycle = orchestrator.run_cycle(
        image_path=image_path,
        text_input=text_input,
    )

    # ── Print results ─────────────────────────────────────────────────
    print_banner("RESULTS")

    # SEE
    obs = cycle.get("observations", {})
    print("\n── SEE (Observations) ──")
    print(f"  Source      : {obs.get('source', 'N/A')}")
    print(f"  Frame shape : {obs.get('frame_shape', 'N/A')}")
    print(f"  Brightness  : {obs.get('brightness', 'N/A')}")
    print(f"  Contrast    : {obs.get('contrast', 'N/A')}")
    print(f"  Faces       : {len(obs.get('faces', []))}")
    print(f"  Hands       : {len(obs.get('hands', []))}")
    print(f"  Text input  : {obs.get('text_input', 'N/A')[:80]}...")

    # RETRIEVE
    ctx = cycle.get("context", {})
    print("\n── RETRIEVE (Context from InstaBrain) ──")
    memories = ctx.get("memories", [])
    print(f"  Memories retrieved: {len(memories)}")
    for m in memories[:5]:
        content = m["content"][:70] + "..." if len(m["content"]) > 70 else m["content"]
        print(f"    - [{m.get('distance', 'N/A'):.3f}] {content}")
    checkpoints = ctx.get("checkpoints", [])
    print(f"  Checkpoints retrieved: {len(checkpoints)}")

    # REASON
    plan = cycle.get("plan", {})
    print("\n── REASON (Plan) ──")
    print(f"  Analysis  : {plan.get('analysis', 'N/A')}")
    print(f"  Rationale : {plan.get('rationale', 'N/A')}")
    plan_steps = plan.get("plan", [])
    if plan_steps:
        print(f"  Steps ({len(plan_steps)}):")
        for step in plan_steps:
            print(f"    - {step}")
    if plan.get("raw_output"):
        print(f"  Raw LLM output: {plan['raw_output'][:200]}...")

    # ACT
    tool_calls = cycle.get("tool_calls", [])
    print(f"\n── ACT (Tool Calls: {len(tool_calls)}) ──")
    for tc in tool_calls:
        print(f"  - {tc.get('tool', '?')}({json.dumps(tc.get('params', {}), default=str)})")

    # AUDIT
    if "audit" in cycle:
        audit = cycle["audit"]
        verdict = "APPROVED" if audit.get("approved") else "BLOCKED"
        print(f"\n── AUDIT ──")
        print(f"  Verdict : {verdict}")
        print(f"  Reason  : {audit.get('reason', 'N/A')}")
        if audit.get("warnings"):
            for w in audit["warnings"]:
                print(f"  Warning : {w}")

    # TOOL RESULTS
    tool_results = cycle.get("tool_results", [])
    if tool_results:
        print(f"\n── TOOL EXECUTION RESULTS ({len(tool_results)}) ──")
        for tr in tool_results:
            icon = "OK" if tr["status"] == "ok" else "ERR"
            print(f"  [{icon}] {tr['tool']} ({tr['latency_ms']:.1f}ms)")
            if tr["status"] == "ok":
                result_str = json.dumps(tr["result"], indent=4, default=str)
                for line in result_str.split("\n")[:10]:
                    print(f"       {line}")
                if len(result_str.split("\n")) > 10:
                    print("       ...")
            else:
                print(f"       Error: {tr.get('error', 'unknown')}")

    print_banner("TEST COMPLETE")
    print(f"  Correlation ID : {cycle['correlation_id']}")
    print(f"  Session ID     : {cycle['session_id']}")
    print(f"  Tool calls made: {len(tool_calls)}")
    print(f"  Tools executed : {len(tool_results)}")
    print()

    orchestrator.close()


if __name__ == "__main__":
    run()
