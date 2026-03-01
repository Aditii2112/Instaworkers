# Edge AI Agent — InstaWorkers

Production-grade **edge AI agent platform** with a **See -> Reason -> Act** core loop, local **Gemma LLM** (via llama.cpp), vector memory (**InstaBrain**), and real-time observability dashboard (**InstaControl View**).

Built as a problem-agnostic platform — plug in domain-specific tools when the problem statement arrives.

## Architecture

See [`architecture.md`](architecture.md) for the full Mermaid diagram.

| Layer | What it does |
|-------|-------------|
| **SEE** | Vision Specialist — OpenCV + MediaPipe, outputs structured observations |
| **REASON** | Root Orchestrator — retrieves context from InstaBrain, plans via Gemma |
| **ACT** | Action Specialist — generates executable tool calls (JSON) via Gemma |
| **AUDIT** | State Watchdog — validates actions before execution (safety + policy) |
| **InstaBrain** | SQLite + sqlite-vec vector store for agent memory & checkpoints |
| **InstaControl** | JSONL event tracing -> metrics aggregation -> Streamlit dashboard |
| **MatFormer** | Elastic scaling profiles (S/M/L/XL) for resource-adaptive inference |

### Reliability Extras (implemented)

- **Contextual Checkpointing**: every N cycles, Gemma summarizes session state into a compact checkpoint stored in InstaBrain; relevant checkpoints are rehydrated into context automatically.
- **Worker + Auditor (State Watchdog)**: the Action Agent proposes tool calls; the Auditor (Gemma with a constrained safety prompt) validates each action vs. system state before execution — can approve, block, or queue.

## Prerequisites

- **Python 3.11+**
- **llama.cpp** (for local Gemma inference)
- **Gemma weights** in GGUF format

## Setup

### 1. Python Environment

```bash
cd edge_agent_root
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Install & Run Gemma via llama.cpp

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j$(nproc)

# Download Gemma weights (GGUF format)
# Option A: from HuggingFace
#   https://huggingface.co/google/gemma-2b-GGUF
#   Download gemma-2b.Q4_K_M.gguf (or similar quantization)
#
# Option B: use huggingface-cli
#   pip install huggingface_hub
#   huggingface-cli download google/gemma-2b-GGUF gemma-2b.Q4_K_M.gguf --local-dir models/

# Start the llama.cpp server
./llama-server \
  -m models/gemma-2b.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  --threads $(nproc)
```

Verify it's running:

```bash
curl http://localhost:8080/v1/models
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if your llama.cpp server is on a different host/port
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA_BASE_URL` | `http://localhost:8080` | llama.cpp server URL |
| `GEMMA_MODEL_NAME` | `gemma-2b` | Model identifier |
| `MATFORMER_GRANULARITY` | `M` | Elastic scaling profile (S/M/L/XL) |
| `CHECKPOINT_ENABLED` | `true` | Enable contextual checkpointing |
| `CHECKPOINT_INTERVAL_CALLS` | `10` | Cycles between checkpoints |
| `AUDITOR_ENABLED` | `true` | Enable State Watchdog |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

### 4. Run the Agent

```bash
python main.py
```

On first run, the embedding model (~80 MB) downloads automatically. The agent verifies Gemma connectivity and exits with clear instructions if the server is unreachable.

### 5. Run the Dashboard

```bash
streamlit run observability/dashboard.py
```

Opens InstaControl View at `http://localhost:8501` with:
- Real-time CPU/RAM/temperature gauges
- Stage latency time series (SEE, REASON, ACT, AUDIT)
- End-to-end latency tracking
- MatFormer granularity distribution pie chart
- Safety counters (blocked, auditor rejections, queued, HITL prompts)
- Error rates by stage
- Memory & retrieval stats
- Filterable event log (latest 200 events)

## Project Structure

```
edge_agent_root/
├── main.py                          # Entry point — agent loop
├── config.py                        # Configuration from .env
├── architecture.md                  # Mermaid architecture diagram
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── orchestrator.py              # Root Orchestrator (See->Reason->Act->Audit)
│   ├── vision_agent.py              # SEE — OpenCV + MediaPipe
│   ├── retrieval_agent.py           # Context retrieval from InstaBrain
│   ├── reasoning_agent.py           # REASON — Gemma planning
│   ├── action_agent.py              # ACT — Gemma tool-call generation
│   └── auditor_agent.py             # AUDIT — State Watchdog
│
├── llm/
│   ├── gemma_runtime.py             # Gemma HTTP client (llama.cpp)
│   └── prompts.py                   # System prompts per stage
│
├── memory/
│   ├── embeddings.py                # sentence-transformers embeddings
│   └── instabrain_db.py             # SQLite + sqlite-vec vector store
│
├── tools/
│   ├── tool_registry.py             # Pluggable tool definitions
│   ├── tool_runner.py               # Execution + logging
│   └── adapters/
│       ├── filesystem.py            # Restricted file R/W
│       ├── http_client.py           # Allowlisted HTTP
│       └── shell_allowlist.py       # Allowlisted shell commands
│
├── observability/
│   ├── event_trace.py               # JSONL structured event logging
│   ├── metrics.py                   # Rolling aggregation + psutil
│   └── dashboard.py                 # Streamlit InstaControl View
│
└── data/
    ├── instabrain.db                # Created at runtime
    └── events.jsonl                 # Created at runtime
```

## Adding Tools (When Problem Statement Arrives)

The tool registry is fully pluggable. To add a new tool:

### 1. Create an adapter

```python
# tools/adapters/my_sensor.py

def read_sensor(sensor_id: str, metric: str = "temperature") -> dict:
    """Read from a real sensor — implement your hardware interface here."""
    # Your real implementation
    value = hardware_api.read(sensor_id, metric)
    return {"sensor_id": sensor_id, "metric": metric, "value": value}
```

### 2. Register it in main.py

```python
from tools.tool_registry import ToolDefinition
from tools.adapters import my_sensor

registry.register(ToolDefinition(
    name="read_sensor",
    description="Read a value from a hardware sensor",
    parameters={
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string", "description": "Sensor identifier"},
            "metric": {"type": "string", "enum": ["temperature", "humidity", "pressure"]},
        },
        "required": ["sensor_id"],
    },
    handler=my_sensor.read_sensor,
    tags=["sensor", "hardware"],
))
```

The tool is immediately available to the REASON and ACT agents — they see it in the tool descriptions prompt and can generate calls to it.

## Event Trace Schema

Every agent stage emits one structured event to `data/events.jsonl`:

```json
{
  "ts": "2026-02-26T12:00:00+00:00",
  "correlation_id": "uuid",
  "session_id": "uuid",
  "stage": "SEE|REASON|ACT|AUDIT",
  "latency_ms": 42.5,
  "model": "gemma-2b",
  "model_granularity": "S|M|L|XL",
  "tokens_in": 128,
  "tokens_out": 64,
  "tool_calls": [{"name": "read_file", "status": "ok", "latency_ms": 5.2}],
  "decision": {"type": "plan", "status": "success", "reason": "..."},
  "safety": {"blocked": false, "policy": null},
  "connectivity": {"state": "online"}
}
```

## Dashboard Metrics

| Category | Metrics |
|----------|---------|
| **Health** | CPU%, RAM%, temperature, avg latency per stage, end-to-end latency |
| **Errors** | Error rate per stage, total errors |
| **MatFormer** | % time in S/M/L/XL granularity profiles |
| **Memory** | Retrieval call count, avg retrieval latency, checkpoints/hour |
| **Safety** | Actions blocked by policy, auditor rejections, queued, HITL prompts |
| **Connectivity** | Online/offline/unknown status distribution |

## Future Extensions

- **Offline Buffer**: queue actions when connectivity drops; replay when back online
- **Local-First Escalation**: attempt local resolution before any cloud fallback
- **Multi-Device Federation**: edge mesh for distributed agent coordination
- **Auto-Scaling MatFormer**: dynamically adjust granularity based on real-time CPU/RAM utilization
- **Secure Enclave**: isolate sensitive tool execution in a sandboxed subprocess
