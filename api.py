"""Sovereign Sentinel — FastAPI backend.

Endpoints:
  POST /api/run          — full See → Reason → Act → Audit cycle
  GET  /api/run/stream   — SSE stream of cycle progress
  GET  /api/health       — system health
  GET  /api/tools        — registered tools
  POST /api/seed         — seed InstaBrain
  POST /api/session/new  — fresh session
  GET  /api/safety-log   — auditor verdicts
  GET  /api/alerts       — evidence gallery
  GET  /api/video/demo   — demo video feed
  GET  /api/events       — observability events
"""

import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from config import load_config, ROOT_DIR
from llm.gemma_runtime import GemmaRuntime
from llm.gemini_runtime import GeminiRuntime
from memory.embeddings import EmbeddingEngine
from memory.instabrain_db import InstaBrainDB
from observability.event_trace import EventTracer
from observability.metrics import MetricsCollector
from tools.tool_registry import ToolRegistry
from tools.tool_runner import ToolRunner
from agents.orchestrator import RootOrchestrator
from main import register_baseline_tools, register_industrial_tools

# ── globals ──────────────────────────────────────────────────────────
config = None
gemma = None
gemini = None
db = None
tracer = None
registry = None
runner = None
orchestrator = None
metrics = None
_domain_tools_registered = False

UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
ALERTS_DIR = ROOT_DIR / "data" / "alerts"
_last_uploaded_video: str | None = None


def _startup():
    global config, gemma, gemini, db, tracer, registry, runner, orchestrator, metrics

    config = load_config()
    Path(config.instabrain.db_path).parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    gemma = GemmaRuntime(config)
    if not gemma.is_available():
        print("[INIT] WARNING: Ollama/LLM not reachable at", config.gemma.base_url)
        print("       Start with: ollama serve")
        print("       API will run; /api/run may degrade until LLM is up.")
    else:
        print("[INIT] LLM OK at", config.gemma.base_url)
    gemini = GeminiRuntime(config)

    emb = EmbeddingEngine(config.instabrain.embedding_model)
    db = InstaBrainDB(config.instabrain.db_path, emb)

    tracer = EventTracer(config.observability.events_path)
    metrics = MetricsCollector(config.observability.events_path)

    registry = ToolRegistry()
    runner = ToolRunner(registry)
    register_baseline_tools(registry, config)
    register_industrial_tools(registry)

    try:
        from hackathon_test.scenario import register_domain_tools, FAKE_DATA
        register_domain_tools(registry)
        _domain_tools_registered = True
        for i, record in enumerate(FAKE_DATA):
            db.insert_memory(record, metadata={"source": "startup_seed", "index": i})
    except Exception as e:
        print(f"[INIT] Domain tools auto-register skipped: {e}")

    from config import ROOT_DIR as _rd
    inv_db = _rd / "data" / "inventory.db"
    if not inv_db.exists():
        from seed_db import seed_inventory
        seed_inventory()

    orchestrator = RootOrchestrator(
        config=config, gemma=gemma, db=db,
        tracer=tracer, registry=registry, runner=runner, gemini=gemini,
    )
    print("[INIT] Sovereign Sentinel ready")


@asynccontextmanager
async def lifespan(application):
    _startup()
    yield


app = FastAPI(title="Sovereign Sentinel", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    llm_available = gemma.is_available() if gemma else False
    return {
        "status": "ok" if llm_available else "degraded",
        "system": "Sovereign Sentinel",
        "llm": config.gemma.base_url if config else "not initialized",
        "llm_connected": llm_available,
        "vlm_model": config.vlm.model_path if config else "N/A",
        "matformer": config.matformer.granularity if config else "N/A",
        "tools": registry.list_names() if registry else [],
        "auditor": config.auditor.enabled if config else False,
        "auditor_model": config.auditor.model_name if config else "N/A",
        "video_fps": config.video.fps_target if config else 2.0,
        "hybrid_reliability": config.hybrid.enabled if config else False,
        "gemini_teacher": gemini.connectivity_state() if gemini else "offline",
    }


@app.get("/api/tools")
def list_tools():
    return {"tools": registry.list_tools() if registry else []}


@app.post("/api/seed")
def seed_data(records: list[str]):
    inserted = []
    for i, record in enumerate(records):
        row_id = db.insert_memory(record, metadata={"source": "api_seed", "index": i})
        inserted.append({"id": row_id, "preview": record[:80]})
    return {"seeded": len(inserted), "records": inserted}


@app.post("/api/register-domain-tools")
def register_domain_tools_endpoint():
    global _domain_tools_registered
    if _domain_tools_registered:
        return {"status": "already_registered", "tools": registry.list_names()}
    from hackathon_test.scenario import register_domain_tools
    register_domain_tools(registry)
    _domain_tools_registered = True
    return {"status": "registered", "tools": registry.list_names()}


@app.post("/api/session/new")
def new_session():
    if not orchestrator:
        return JSONResponse(status_code=503, content={"error": "Orchestrator not initialized"})
    previous_session_id = orchestrator.session_id
    orchestrator.new_session()

    # Clear stale data from previous session
    import shutil
    alerts_dir = ROOT_DIR / "data" / "alerts"
    if alerts_dir.exists():
        shutil.rmtree(alerts_dir)
    alerts_dir.mkdir(parents=True, exist_ok=True)
    safety_log = ROOT_DIR / "data" / "safety-log.jsonl"
    if safety_log.exists():
        safety_log.unlink()

    return {
        "status": "ok",
        "previous_session_id": previous_session_id,
        "session_id": orchestrator.session_id,
        "message": "Started a new session with fresh context",
    }


# ── Main Run Cycle ───────────────────────────────────────────────────

def _save_upload(upload: UploadFile, ext_default: str) -> str | None:
    if not upload or not upload.filename:
        return None
    ext = Path(upload.filename).suffix or ext_default
    filename = f"{uuid.uuid4().hex}{ext}"
    path = str(UPLOAD_DIR / filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@app.post("/api/run")
async def run_cycle(
    text: str = Form(""),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
):
    """Run one full See → Reason → Act → Audit cycle."""
    # Purge ALL stale data so the dashboard only shows results from THIS run
    for purge_dir in [ALERTS_DIR, UPLOAD_DIR / "frames", UPLOAD_DIR]:
        if purge_dir.exists():
            shutil.rmtree(purge_dir)
        purge_dir.mkdir(parents=True, exist_ok=True)
    safety_log = ROOT_DIR / "data" / "safety-log.jsonl"
    if safety_log.exists():
        safety_log.unlink()

    global _last_uploaded_video
    image_path = _save_upload(image, ".png")
    video_path = _save_upload(video, ".mp4")
    if video_path:
        _last_uploaded_video = video_path
    text_input = text.strip() if text.strip() else None

    print(f"[API] run_cycle: text={bool(text_input)} image={bool(image_path)} video={video_path}")

    if not text_input and not image_path and not video_path:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide at least text, an image, or a video"},
        )

    cycle = orchestrator.run_cycle(
        image_path=image_path,
        text_input=text_input,
        video_path=video_path,
    )

    return {
        "correlation_id": cycle["correlation_id"],
        "session_id": cycle["session_id"],
        "stages_completed": cycle.get("stages_completed", []),
        "observations": cycle.get("observations", {}),
        "video_info": cycle.get("video_info"),
        "frames_extracted": cycle.get("frames_extracted"),
        "inventory_at_location": cycle.get("inventory_at_location"),
        "context": {
            "memories": [
                {"content": m["content"][:200], "distance": m.get("distance")}
                for m in cycle.get("context", {}).get("memories", [])
            ],
            "checkpoints": cycle.get("context", {}).get("checkpoints", []),
        },
        "matformer": cycle.get("matformer"),
        "plan": cycle.get("plan", {}),
        "tool_calls": cycle.get("tool_calls", []),
        "tool_calls_after_validation": cycle.get("tool_calls_after_validation", []),
        "audit": cycle.get("audit"),
        "validation": cycle.get("validation"),
        "pending_reconcile": cycle.get("pending_reconcile"),
        "safety_alert_triggered": cycle.get("safety_alert_triggered"),
        "tool_results": [
            {
                "tool": r["tool"],
                "status": r["status"],
                "result": r.get("result"),
                "error": r.get("error"),
                "latency_ms": r["latency_ms"],
            }
            for r in cycle.get("tool_results", [])
        ],
    }


# ── SSE Stream ───────────────────────────────────────────────────────

@app.post("/api/run/stream")
async def run_cycle_stream(
    text: str = Form(""),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
):
    """SSE stream of the pipeline for real-time dashboard updates."""
    image_path = _save_upload(image, ".png")
    video_path = _save_upload(video, ".mp4")
    text_input = text.strip() if text.strip() else None

    if not text_input and not image_path and not video_path:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide at least text, an image, or a video"},
        )

    def event_generator():
        for event in orchestrator.run_cycle_stream(
            image_path=image_path,
            text_input=text_input,
            video_path=video_path,
        ):
            data = json.dumps(event["data"], default=str)
            yield f"event: {event['stage']}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Observability Endpoints ──────────────────────────────────────────

@app.get("/api/metrics/summary")
def metrics_summary(window_minutes: int = 60):
    window_minutes = max(1, min(window_minutes, 24 * 60))
    return {
        "window_minutes": window_minutes,
        "system": metrics.system_metrics() if metrics else {},
        "stage_latency": metrics.stage_latency(window_minutes) if metrics else {},
        "end_to_end": metrics.end_to_end_latency(window_minutes) if metrics else {},
        "granularity": metrics.granularity_distribution(window_minutes) if metrics else {},
        "safety": metrics.safety_stats(window_minutes) if metrics else {},
        "error_rate": metrics.error_rate(window_minutes) if metrics else {},
        "retrieval": metrics.retrieval_stats(window_minutes) if metrics else {},
        "checkpoints": metrics.checkpoint_stats(window_minutes) if metrics else {},
        "connectivity": metrics.connectivity_stats(window_minutes) if metrics else {},
        "pending_validation_queue": db.pending_validation_counts() if db else {},
    }


@app.get("/api/metrics/timeseries")
def metrics_timeseries(window_minutes: int = 60, bucket_seconds: int = 60):
    window_minutes = max(1, min(window_minutes, 24 * 60))
    bucket_seconds = max(10, min(bucket_seconds, 600))
    return {
        "window_minutes": window_minutes,
        "bucket_seconds": bucket_seconds,
        "points": metrics.timeseries(window_minutes, bucket_seconds) if metrics else [],
    }


@app.get("/api/safety-log")
def get_safety_log(limit: int = 100):
    log_path = ROOT_DIR / "data" / "safety-log.jsonl"
    if not log_path.exists():
        return {"entries": []}
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    entries = entries[-limit:][::-1]
    return {"entries": entries}


@app.get("/api/alerts/{filename}")
def serve_alert_image(filename: str):
    path = ALERTS_DIR / filename
    if not path.exists() or not path.is_file():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/video/demo")
def serve_demo_video():
    if _last_uploaded_video and Path(_last_uploaded_video).exists():
        return FileResponse(_last_uploaded_video, media_type="video/mp4")
    from config import load_config as _lc
    cfg = _lc()
    video_path = Path(cfg.video.demo_path)
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"error": "Demo video not found. Add data/demo.mp4 or set VIDEO_DEMO_PATH"})
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/alerts")
def list_alerts():
    if not ALERTS_DIR.exists():
        return {"alerts": []}
    alerts = []
    for f in sorted(ALERTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        if f.suffix == ".json" and "_ticket" in f.name:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    data["_file"] = f.name
                    alerts.append(data)
            except (json.JSONDecodeError, IOError):
                pass
    return {"alerts": alerts}


@app.get("/api/events")
def list_events(limit: int = 200, stage: str | None = None, status: str | None = None):
    limit = max(1, min(limit, 1000))
    return {
        "limit": limit,
        "events": tracer.load_events(limit=limit, stage=stage, status=status) if tracer else [],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
