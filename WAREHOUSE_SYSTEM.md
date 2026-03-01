# Autonomous Multi-Agent Warehouse Intelligence System

This document describes the refactored Sovereign Sentinel system for live video processing and industrial tooling.

## Quick Start

1. **Seed inventory** (creates `data/inventory.db`):
   ```bash
   python seed_db.py
   ```

2. **Add demo video** (or set `VIDEO_DEMO_PATH` in `.env`):
   ```bash
   cp /path/to/your/demo.mp4 data/demo.mp4
   ```

3. **Start Ollama** (Terminal 1):
   ```bash
   ollama serve
   ```

4. **Start API** (Terminal 2):
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

5. **Start frontend** (Terminal 3):
   ```bash
   cd frontend && npm run dev
   ```

6. Open the app, click **Warehouse Dashboard**, upload a video or image, and run the agent.

## Architecture

### Stage 1: Vision Agent (`agents/vision_agent.py`)
- **OpenCV** `cv2.VideoCapture` loop at 2–5 FPS
- **MediaPipe** face/hand detection → `human_present: true` when a person is in frame
- **PaliGemma 2** via Ollama for VLM anomaly analysis
- Output: `{anomaly, severity, location, human_present, timestamp}`

### Stage 2: Orchestrator (`agents/orchestrator.py`)
On anomaly detection:
- **Inventory Agent**: Queries `inventory.db` for SKUs at the visual location
- **Safety Agent**: If `human_present` and `severity == "critical"` → immediately calls `broadcast_safety_alert`
- **Action Agent**: Generates tool calls from vision + inventory context

### Stage 3: Industrial Tools (`tools/industrial_tools.py`)
- `dispatch_visual_ticket`: Routes to Facilities (leak), Inventory (spoilage), Pest Control (pest); saves frame as `.jpg` and `ticket.json` in `/data/alerts/`
- `quarantine_inventory_sku`: SQL `UPDATE` on `inventory.db` to set status `QUARANTINED`
- `broadcast_safety_alert`: MacOS `say` command to announce the hazard

### Stage 4: Auditor Agent (`agents/auditor_agent.py`)
- Uses **Gemma 2B** via Ollama (`AUDITOR_MODEL=gemma:2b`)
- **SENT-001**: Spoilage must trigger quarantine
- **SENT-002**: Block industrial cleaning tools if `human_present`
- **SENT-003**: Only approve "System Shutdown" if 3+ video-verified anomalies
- Logs verdicts to `data/safety-log.jsonl`

### Stage 5: Dashboard (`frontend/src/components/WarehouseDashboard.jsx`)
- Live video feed (from `/api/video/demo`)
- Agent flow: SEE → REASON → ACT → AUDIT
- Team dispatch icons (Facilities, Inventory, Safety) pulse when a ticket is routed
- Visual evidence panel (latest captured frame)
- Live safety log (Auditor verdicts)

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/video/demo` | Demo video for live feed |
| `GET /api/alerts` | List tickets and evidence |
| `GET /api/alerts/{filename}` | Serve alert image |
| `GET /api/safety-log` | Auditor verdicts |
| `POST /api/run` | Run full cycle (text, image, video) |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `VIDEO_DEMO_PATH` | `data/demo.mp4` | Path to demo video |
| `VIDEO_FPS` | `3.0` | Target FPS for video processing |
| `AUDITOR_MODEL` | `gemma:2b` | Ollama model for Auditor |
