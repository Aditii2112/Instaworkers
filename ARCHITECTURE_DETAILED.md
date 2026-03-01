# Instaworkers — Detailed Architecture

This document describes the full architecture of the **Instaworkers** edge AI agent platform: components, models, data flow, configuration, and dependencies.

---

## 1. Overview

**Instaworkers** is an **edge AI agent platform** that implements a **See → Reason → Act → Audit** loop with:

| Layer | Technology |
|-------|------------|
| **Primary LLM/VLM** | **Sovereign Sentinel** — PaliGemma 2 (3B) fine-tuned via MLX LoRA, served by **Ollama** (or llama.cpp) |
| **Vision (SEE)** | OpenCV + MediaPipe (face/hand) + optional VLM analysis via the same model |
| **Memory** | **InstaBrain** — SQLite + vector search (sqlite-vec or NumPy fallback), embeddings via **sentence-transformers** |
| **Observability** | JSONL event log → **InstaControl** (Streamlit dashboard + React metrics view) |
| **Reliability** | Local **Auditor** (Gemma) + **Hybrid Reliability** (optional Gemini “teacher” for high-risk actions) |

The repo is a **monorepo** combining: **ML (training + inference)**, **agent orchestration**, **REST API**, and **web frontend**.

---

## 2. Directory Structure

```
Instaworkers/
└── edge_agent_root/                    # Project root
    ├── main.py                         # CLI entry — interactive See→Reason→Act loop
    ├── api.py                          # FastAPI REST API entry
    ├── config.py                       # Env-based configuration (dataclasses)
    ├── requirements.txt                # Python dependencies
    ├── .env.example / .env             # Environment variables
    ├── lora_config.yaml                # LoRA fine-tuning config (rank, alpha, keys)
    ├── train_sentinel.py               # Data generation + MLX LoRA training script
    ├── deploy_to_ollama.sh             # Fuse adapters → GGUF → Ollama model
    ├── download_datasets.py            # HuggingFace dataset download/organization
    ├── architecture.md                 # Mermaid architecture diagram
    ├── README.md
    │
    ├── agents/                         # Agent pipeline
    │   ├── orchestrator.py             # RootOrchestrator — full cycle
    │   ├── vision_agent.py             # SEE — OpenCV + MediaPipe + VLM
    │   ├── video_processor.py          # Video → key-frame extraction
    │   ├── retrieval_agent.py          # InstaBrain RAG
    │   ├── reasoning_agent.py          # REASON — Gemma planning
    │   ├── action_agent.py             # ACT — Gemma tool-call generation
    │   └── auditor_agent.py            # AUDIT — State Watchdog (Gemma)
    │
    ├── llm/                            # LLM clients
    │   ├── gemma_runtime.py            # Ollama / OpenAI-compat (llama.cpp)
    │   ├── gemini_runtime.py           # Optional Gemini teacher validator
    │   └── prompts.py                  # System prompts (SEE, REASON, ACT, AUDIT, CHECKPOINT)
    │
    ├── memory/
    │   ├── embeddings.py               # sentence-transformers EmbeddingEngine
    │   └── instabrain_db.py            # SQLite + sqlite-vec (or NumPy) vector store
    │
    ├── tools/
    │   ├── tool_registry.py            # ToolDefinition + ToolRegistry
    │   ├── tool_runner.py              # Execute tools with timing/errors
    │   └── adapters/
    │       ├── filesystem.py           # Restricted read/write under TOOL_FS_BASE_DIR
    │       ├── http_client.py          # Allowlisted HTTP GET/POST
    │       └── shell_allowlist.py      # Allowlisted shell commands
    │
    ├── observability/
    │   ├── event_trace.py              # JSONL AgentEvent logging
    │   ├── metrics.py                  # MetricsCollector (stage latency, errors, safety)
    │   └── dashboard.py                # Streamlit InstaControl View
    │
    ├── reliability/
    │   └── hybrid_reliability.py       # Local confidence + optional Gemini teacher
    │
    ├── hackathon_test/                 # Demo / E2E test
    │   ├── scenario.py                 # FAKE_DATA, domain tools, PROBLEM_INPUTS
    │   ├── run_test.py                 # Full pipeline test with real Gemma
    │   └── generate_test_image.py      # Synthetic warehouse image
    │
    ├── frontend/                       # React SPA
    │   ├── package.json                # React 19, Vite 7, Tailwind 4
    │   ├── vite.config.js              # Dev proxy /api → localhost:8000
    │   └── src/
    │       ├── main.jsx, App.jsx
    │       ├── components/
    │       │   ├── InputPanel.jsx
    │       │   ├── ResultsPanel.jsx
    │       │   ├── PipelineView.jsx
    │       │   └── ObservabilityDashboard.jsx
    │       └── index.css
    │
    ├── models/
    │   └── paligemma2-base/            # Base PaliGemma 2 (config, tokenizer, weights)
    │       ├── config.json
    │       ├── tokenizer.json, tokenizer_config.json
    │       └── (safetensors)
    │
    ├── adapters/                       # Output: LoRA adapter weights (after training)
    ├── data/                           # Runtime: instabrain.db, events.jsonl, uploads/, models/
    ├── finetune_data/                  # train.jsonl, valid.jsonl (from train_sentinel)
    ├── train.jsonl                     # Flat training records (from train_sentinel)
    └── datasets/                       # Raw datasets (from download_datasets.py)
        ├── package_damage/
        ├── ip102/
        └── mvtec/
```

---

## 3. Components & Modules

### 3.1 Entry Points

| Entry | Purpose |
|-------|--------|
| **`main.py`** | Interactive CLI: loads config, Gemma, InstaBrain, tools, orchestrator; runs loop with `input(">>> ")` and `orchestrator.run_cycle(text_input=...)`. |
| **`api.py`** | FastAPI app: `POST /api/run`, `GET /api/health`, `GET /api/tools`, `POST /api/seed`, `POST /api/session/new`, `GET /api/metrics/*`, `GET /api/events`. Served with `uvicorn api:app` (e.g. port 8000). |
| **`train_sentinel.py`** | Builds `train.jsonl` / `finetune_data/{train,valid}.jsonl` from datasets, then runs `mlx_vlm.lora` training. CLI: `--generate-only`, `--train-only`, `--datasets-dir`, `--model-dir`, `--adapter-dir`, `--resume`. |
| **`deploy_to_ollama.sh`** | Fuses LoRA into base model (`mlx_vlm.fuse`), converts to GGUF, writes Modelfile, runs `ollama create sovereign-sentinel`. |
| **`download_datasets.py`** | Fetches/organizes package-damage, IP102, MVTec into `datasets/` layout for `train_sentinel.py`. |
| **`observability/dashboard.py`** | Streamlit dashboard: `streamlit run observability/dashboard.py` (e.g. port 8501). |
| **`hackathon_test/run_test.py`** | E2E: generate image, seed InstaBrain, register domain tools, run one orchestrator cycle. |

### 3.2 Agents (`agents/`)

| Agent | File | Responsibility |
|-------|------|----------------|
| **RootOrchestrator** | `orchestrator.py` | Runs one cycle: optional video frame extraction → SEE → MatFormer selection → Retrieval → REASON → ACT → AUDIT → Hybrid reliability → tool execution → optional checkpoint. Uses VideoProcessor, VisionAgent, RetrievalAgent, ReasoningAgent, ActionAgent, AuditorAgent, HybridReliabilityManager, ToolRunner. |
| **VisionAgent** | `vision_agent.py` | **SEE.** Input: frame array, `image_path`, or text. Uses OpenCV, MediaPipe (face/hand from `data/models/`), and optionally Gemma with image for VLM analysis (SEE/REASON/ACT-style output). Writes events via EventTracer. |
| **VideoProcessor** | `video_processor.py` | Opens video with OpenCV, samples frames (e.g. 1 fps, max 30), saves as JPEGs; used by orchestrator for video input. |
| **RetrievalAgent** | `retrieval_agent.py` | Calls `InstaBrainDB.query_memories` and `query_checkpoints` with `top_k` from MatFormer profile; logs retrieval event. |
| **ReasoningAgent** | `reasoning_agent.py` | Builds prompt from observations + context + tool list; calls `GemmaRuntime.generate` with `REASON_SYSTEM_PROMPT`; parses JSON plan (analysis, plan[], rationale, delegate_to). |
| **ActionAgent** | `action_agent.py` | Converts plan to tool-call list via Gemma with `ACT_SYSTEM_PROMPT`; parses JSON array of `{tool, params}`. |
| **AuditorAgent** | `auditor_agent.py` | If enabled: runs Gemma with `AUDIT_SYSTEM_PROMPT` on proposed tool calls + system state; returns approve/block/warnings; blocks by index. |

**Internal dependencies:** `config`, `llm.gemma_runtime`, `llm.prompts`, `memory.instabrain_db`, `observability.event_trace`, `reliability.hybrid_reliability`, `tools.tool_registry` / `tool_runner`.

### 3.3 LLM (`llm/`)

| Module | Responsibility |
|--------|----------------|
| **GemmaRuntime** (`gemma_runtime.py`) | HTTP client to Ollama (`/api/generate`, optional image base64) or OpenAI-compatible `/v1/chat/completions`. Used for SEE VLM, REASON, ACT, AUDIT, checkpoint summarization. |
| **GeminiRuntime** (`gemini_runtime.py`) | Optional. Calls Google `generativelanguage.googleapis.com` for action validation (structured JSON: approved, confidence, reason); used by HybridReliabilityManager. |
| **prompts.py** | Defines `REASON_SYSTEM_PROMPT`, `ACT_SYSTEM_PROMPT`, `AUDIT_SYSTEM_PROMPT`, `CHECKPOINT_SYSTEM_PROMPT`, `SEE_SYSTEM_PROMPT`. |

**External:** Ollama (or llama.cpp), Google Gemini API when enabled.

### 3.4 Memory (`memory/`)

| Module | Responsibility |
|--------|----------------|
| **EmbeddingEngine** (`embeddings.py`) | Wraps `SentenceTransformer(embedding_model)`; `embed()` / `embed_batch()`; dimension from model (e.g. 384 for all-MiniLM-L6-v2). |
| **InstaBrainDB** (`instabrain_db.py`) | SQLite DB with tables `memories`, `checkpoints`, `pending_validations`; optional sqlite-vec for vector search, else NumPy cosine similarity. Methods: `insert_memory`, `query_memories`, `insert_checkpoint`, `get_recent_checkpoints`, `query_checkpoints`; for hybrid: `insert_pending_validation`, `get_pending_validations`, `update_pending_validation`, `pending_validation_counts`. |

**External:** sentence-transformers (downloads model from HuggingFace on first use).

### 3.5 Tools (`tools/`)

| Module | Responsibility |
|--------|----------------|
| **ToolRegistry** (`tool_registry.py`) | Registers `ToolDefinition` (name, description, parameters schema, handler, requires_approval, tags); `get_schema_prompt()` for LLM. |
| **ToolRunner** (`tool_runner.py`) | `execute(tool_name, params)` and `execute_batch(tool_calls)`; returns `{tool, status, result|error, latency_ms}`. |
| **Adapters** | **filesystem:** read_file, write_file, list_directory under `TOOL_FS_BASE_DIR`. **http_client:** http_get / http_post restricted to `TOOL_HTTP_ALLOWLIST`. **shell_allowlist:** run_shell restricted to `TOOL_SHELL_ALLOWLIST`. |

Baseline tools registered in `main.register_baseline_tools()`; domain tools in `hackathon_test.scenario.register_domain_tools()` (and optionally on API startup).

### 3.6 Observability (`observability/`)

| Module | Responsibility |
|--------|----------------|
| **EventTracer** (`event_trace.py`) | Appends `AgentEvent` (ts, correlation_id, session_id, stage, latency_ms, model, decision, safety, etc.) to JSONL; `load_events(limit, stage, status)` for reading. |
| **MetricsCollector** (`metrics.py`) | Reads JSONL; computes per-stage latency, e2e latency, error rate, granularity distribution, safety stats, retrieval/checkpoint/connectivity; `system_metrics()` uses psutil (CPU, RAM, temperatures). |
| **dashboard.py** | Streamlit UI: system health, stage latency, e2e, MatFormer pie, safety counters, error rate, retrieval/checkpoints, event list. |

### 3.7 Reliability (`reliability/`)

| Module | Responsibility |
|--------|----------------|
| **HybridReliabilityManager** (`hybrid_reliability.py`) | Per tool call: local assessment (tool existence, requires_approval, params, budget, risky keywords). If high-risk (e.g. budget > threshold) → block and insert into `pending_validations`. Else if below confidence threshold and Gemini online → teacher validate. Else queue for retry. `reconcile_pending()` processes queued validations when teacher is back. Emits AUDIT-stage events. |

**Internal:** `config`, `llm.gemini_runtime`, `memory.instabrain_db`, `observability.event_trace`, `tools.tool_registry`.

### 3.8 Hackathon Test (`hackathon_test/`)

| File | Responsibility |
|------|----------------|
| **scenario.py** | Defines `FAKE_DATA` (zone/worker/safety/equipment/incident text), `PROBLEM_INPUTS`, and `register_domain_tools(registry)` for warehouse tools: `lookup_zone`, `check_incidents`, `lookup_safety_code`, `flag_violation`, `generate_report` (in-memory DBs). |
| **run_test.py** | Loads config, Gemma, InstaBrain, embeddings, tracer, registry (baseline + domain), orchestrator; seeds FAKE_DATA; generates test image; runs one `run_cycle(image_path=..., text_input=...)`. |
| **generate_test_image.py** | Produces synthetic warehouse image for testing. |

### 3.9 Frontend (`frontend/`)

- **Stack:** React 19, Vite 7, Tailwind 4. Dev server port 3000; `/api` proxied to `http://localhost:8000`.
- **App.jsx:** View toggle (pipeline vs observability); health fetch; optional domain-tool registration; `handleSubmit` → `POST /api/run` (FormData: text, image); `handleNewSession` → `POST /api/session/new`; renders InputPanel, PipelineView, ResultsPanel, or ObservabilityDashboard.
- **InputPanel:** Text + image (and optionally video) input and submit.
- **ResultsPanel:** Displays cycle result (observations, context, plan, tool calls, audit, validation, tool results, video info).
- **PipelineView:** Progress through see → reason → act → audit.
- **ObservabilityDashboard:** Fetches `/api/metrics/summary`, `/api/metrics/timeseries`, `/api/events` and visualizes.

**External:** Backend at localhost:8000 (via proxy in dev).

---

## 4. Models

| Model | Role | Where defined/loaded | Framework / config |
|-------|------|----------------------|--------------------|
| **Sovereign Sentinel** | Primary VLM/LLM (SEE, REASON, ACT, AUDIT, checkpoint) | Served by Ollama; client in `llm/gemma_runtime.py` | Fine-tuned PaliGemma 2 (3B); name from `GEMMA_MODEL_NAME` (default `sovereign-sentinel`). |
| **PaliGemma 2 (base)** | Base for fine-tuning | `models/paligemma2-base/` (config, tokenizer, safetensors); referenced in `train_sentinel.py` and `deploy_to_ollama.sh` | `config.json`: PaliGemmaForConditionalGeneration, Gemma2 text (2304d, 26L), SigLIP vision (256 image tokens). Training: `train_sentinel.py` → `mlx_vlm.lora`. |
| **LoRA adapters** | Vision-language fine-tuning | Output to `adapters/` by `train_sentinel.py`; fused in `deploy_to_ollama.sh` via `mlx_vlm.fuse` | **`lora_config.yaml`:** rank 16, alpha 32, dropout 0.05, scale 2.0, keys `q_proj`, `v_proj`. |
| **all-MiniLM-L6-v2** | InstaBrain embeddings | `memory/embeddings.py` — `SentenceTransformer(embedding_model)` | Config: `EMBEDDING_MODEL` (default all-MiniLM-L6-v2), 384-dim. |
| **MediaPipe face** | SEE face detection | `vision_agent.py`; path from `_ensure_model("face_detector")` → `data/models/` | URL: `storage.googleapis.com/.../blaze_face_short_range.tflite`. |
| **MediaPipe hand** | SEE hand landmarks | Same; `_ensure_model("hand_landmarker")` | URL: `.../hand_landmarker.task`. |
| **Gemini** (optional) | Teacher validator | `llm/gemini_runtime.py` | `GEMINI_MODEL_NAME` (default gemini-1.5-pro), REST API. |

**Training pipeline:**  
`download_datasets.py` → `datasets/` → `train_sentinel.py` (builds `train.jsonl` / `finetune_data/train.jsonl`, `valid.jsonl`) → `python -m mlx_vlm.lora --model-path models/paligemma2-base --dataset finetune_data --output-path adapters ...` → `deploy_to_ollama.sh` (fuse → GGUF → Ollama).

---

## 5. Configuration

All behavior is driven by **environment variables** (and `.env`), loaded in **`config.py`** via `dotenv`. `load_config()` returns an **`AppConfig`** dataclass.

### 5.1 Config dataclasses (`config.py`)

| Config | Env vars | Purpose |
|--------|----------|--------|
| **GemmaConfig** | `GEMMA_BASE_URL` (default `http://localhost:11434`), `GEMMA_MODEL_NAME` (default `sovereign-sentinel`) | Ollama/LLM endpoint and model name. |
| **MatFormerConfig** | `MATFORMER_GRANULARITY` (S/M/L/XL, default `M`) | Profile selection; `MATFORMER_PROFILES` in code define S/M/L/XL (max_tokens, context_k, temperature). |
| **InstaBrainConfig** | `INSTABRAIN_DB_PATH`, `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`), `embedding_dim` (384) | DB path and embedding model. |
| **ObservabilityConfig** | `EVENTS_JSONL_PATH` | JSONL event log path. |
| **ToolsConfig** | `TOOL_HTTP_ALLOWLIST`, `TOOL_SHELL_ALLOWLIST`, `TOOL_FS_BASE_DIR` | Allowlists and filesystem root for tools. |
| **CheckpointConfig** | `CHECKPOINT_ENABLED` (default true), `CHECKPOINT_INTERVAL_CALLS` (default 10) | Contextual checkpointing. |
| **AuditorConfig** | `AUDITOR_ENABLED` (default true) | Enable/disable AuditorAgent. |
| **GeminiConfig** | `GEMINI_ENABLED` (default false), `GEMINI_API_KEY`, `GEMINI_MODEL_NAME` (default `gemini-1.5-pro`), `GEMINI_TIMEOUT_SECONDS` (20) | Optional Gemini teacher. |
| **HybridReliabilityConfig** | `HYBRID_RELIABILITY_ENABLED` (default true), `LOCAL_CONFIDENCE_THRESHOLD` (0.8), `HIGH_RISK_BUDGET_USD` (500), `PENDING_RETRY_SECONDS` (30), `MAX_PENDING_PROCESS_PER_CYCLE` (5) | Hybrid reliability gate. |

### 5.2 LoRA training

- **`lora_config.yaml`:** `fine_tune_type: lora`, `lora_parameters`: rank 16, alpha 32, dropout 0.05, scale 2.0, keys `q_proj`, `v_proj`.
- **CLI (`train_sentinel.py`):** `--generate-only`, `--train-only`, `--datasets-dir`, `--model-dir`, `--adapter-dir`, `--seed`, `--resume`. Actual training args in `run_lora_training()` (e.g. `--lora-rank 16`, `--lora-alpha 32`, `--epochs 3`, `--batch-size 1`, `--image-resize-shape 224 224`).

### 5.3 API & frontend

- API: port/host from uvicorn (e.g. 8000). No separate API config file.
- Frontend: dev proxy target in `frontend/vite.config.js` (`/api` → `http://localhost:8000`).

---

## 6. Data Flow

1. **Input:** Text and/or image and/or video (CLI `main.py` or `POST /api/run`). Video → `VideoProcessor.extract_and_save` → list of frame paths; first (and optionally others) sent to SEE.
2. **SEE:** `VisionAgent.observe(frame, image_path, text_input)` → OpenCV + MediaPipe (faces, hands, basic stats); if `image_path` and Gemma, VLM analysis (SEE/REASON/ACT format). Output: observations dict. Event written to JSONL.
3. **MatFormer:** Orchestrator selects S/M/L/XL from config or query; profile sets max_tokens, context_k, temperature.
4. **Retrieve:** `RetrievalAgent.retrieve(query, top_k)` → InstaBrain memories + checkpoints → context dict. Event written.
5. **REASON:** `ReasoningAgent.reason(observations, context, tool_descriptions)` → Gemma → JSON plan (analysis, plan[], rationale). Event written.
6. **ACT:** `ActionAgent.act(plan, tool_descriptions)` → Gemma → list of `{tool, params}`. Event written.
7. **AUDIT:** If enabled, `AuditorAgent.audit(tool_calls, system_state)` → Gemma → approve/block/warnings; tool_calls filtered by blocked indices. Event written.
8. **Hybrid reliability:** `HybridReliabilityManager.evaluate_actions(tool_calls)` → allow / block / queue (pending_validations); optional Gemini teacher. Event written. Only allowed_calls proceed.
9. **Execute:** `ToolRunner.execute_batch(allowed_calls)` → handler invocations; results returned; cycle summary stored with `db.insert_memory(...)`.
10. **Checkpoint:** Every `CHECKPOINT_INTERVAL_CALLS` cycles, orchestrator builds checkpoint prompt, Gemma generates summary JSON, `db.insert_checkpoint(...)`.
11. **Reconcile:** Start of next cycle can run `reliability.reconcile_pending()` to re-validate queued actions when Gemini is available.
12. **Observability:** All stages emit to JSONL; MetricsCollector and dashboard read JSONL; API exposes `/api/metrics/*`, `/api/events`.

**End-to-end:** **User input → SEE → Retrieve → REASON → ACT → AUDIT → Hybrid gate → Tools → InstaBrain + JSONL.**

---

## 7. Dependencies & External Services

### 7.1 Python (`requirements.txt`)

| Category | Packages |
|----------|----------|
| Core | httpx, python-dotenv, pydantic |
| Vision (SEE) | opencv-python, mediapipe |
| Embeddings | sentence-transformers |
| Vector store | sqlite-vec |
| Observability | psutil, streamlit, plotly, pandas |
| MLX fine-tuning (Apple Silicon) | mlx, mlx-lm, mlx-vlm |
| Utilities | numpy |

*Note: FastAPI and uvicorn are used by `api.py`; add them to requirements if not already present (e.g. `fastapi`, `uvicorn`).*

### 7.2 External services

| Service | Purpose |
|---------|---------|
| **Ollama** | Default LLM server (port 11434); serves `sovereign-sentinel`. Fallback: llama.cpp OpenAI-compatible server (e.g. port 8080). |
| **Google Gemini** | Optional: `generativelanguage.googleapis.com` for hybrid reliability teacher. |
| **HuggingFace** | Datasets (e.g. package-damage, IP102, MVTec) via `download_datasets.py`; sentence-transformers model download. |
| **Google Cloud Storage** | MediaPipe model files (`storage.googleapis.com/...`) into `data/models/`. |
| **Tool allowlist** | Example: `httpbin.org`, `api.github.com` for HTTP; shell: ls, cat, head, tail, wc, date, uptime, df, whoami. |

### 7.3 Deploy pipeline

- **llama.cpp** repo (for `convert_hf_to_gguf.py`), **ollama** CLI.
- Python **mlx_vlm** for fuse and (in training) `mlx_vlm.lora`.

---

## 8. High-Level Diagram (from `architecture.md`)

The existing **`architecture.md`** contains a Mermaid flowchart for:

- INPUTS → SEE → REASON (with InstaBrain, MatFormer, Checkpointing) → ACT → AUDIT → TOOLS
- INSTABRAIN (SQLite + sqlite-vec, Embeddings, Checkpoint Store)
- INSTACONTROL (Event Trace → Metrics → Streamlit Dashboard)

This detailed document aligns with that diagram and fills in implementation details, file paths, config, and models.
