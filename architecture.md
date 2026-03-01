# Edge AI Agent — Architecture

```mermaid
flowchart TB
    subgraph INPUTS["📥 INPUTS"]
        direction LR
        CAM["Camera / Frames"]
        TXT["Text / Commands"]
        DOCS["Documents"]
        TEL["Telemetry"]
    end

    subgraph SEE["👁 SEE — Vision Specialist"]
        CV["OpenCV + MediaPipe<br/>Real CV Pipeline"]
        STRUCT_OBS["Structured Observations<br/>(JSON)"]
        CV --> STRUCT_OBS
    end

    subgraph REASON["🧠 REASON — Root Orchestrator"]
        CTX_ASM["Context Assembly<br/>(InstaBrain RAG)"]
        MF["MatFormer Profile<br/>S / M / L / XL"]
        PLANNER["Task Planner<br/>(Local Gemma LLM)"]
        CKPT["Contextual<br/>Checkpointing"]
        CTX_ASM --> MF --> PLANNER
        PLANNER -.->|"every N cycles"| CKPT
    end

    subgraph ACT["⚡ ACT — Action Specialist"]
        TOOL_GEN["Generate Tool Calls<br/>(Gemma → JSON)"]
        TOOL_EXEC["Tool Runner<br/>(Allowlisted)"]
        TOOL_GEN --> TOOL_EXEC
    end

    subgraph AUDIT["🛡 AUDIT — State Watchdog"]
        VALIDATE["Validate Action<br/>vs System State"]
        GATE{"Approve /<br/>Block /<br/>Queue"}
        VALIDATE --> GATE
    end

    subgraph INSTABRAIN["🧠 InstaBrain — Memory"]
        SQLITE[("SQLite + sqlite-vec")]
        VEC["Vector Embeddings<br/>(sentence-transformers)"]
        CKPT_STORE["Checkpoint Store"]
        SQLITE --- VEC
        SQLITE --- CKPT_STORE
    end

    subgraph TOOLS["🔧 Tool Registry"]
        direction LR
        FS["Filesystem<br/>R / W"]
        HTTP["HTTP<br/>GET / POST"]
        SH["Shell<br/>(Allowlist)"]
        PLUG["+ Your<br/>Tools"]
    end

    subgraph INSTACONTROL["📊 InstaControl — Observability"]
        TRACE["Event Trace<br/>(JSONL)"]
        METRICS["Metrics Collector<br/>(psutil + aggregation)"]
        DASH["InstaControl View<br/>(Streamlit Dashboard)"]
        TRACE --> METRICS --> DASH
    end

    INPUTS --> SEE
    SEE -->|"observations"| REASON
    REASON -->|"plan"| ACT
    ACT -->|"proposed actions"| AUDIT
    AUDIT -->|"approved"| TOOLS
    AUDIT -.->|"blocked / queued"| REASON
    TOOLS -->|"results"| REASON

    REASON <-->|"read / write"| INSTABRAIN
    CKPT -->|"store checkpoint"| CKPT_STORE
    INSTABRAIN -->|"context"| CTX_ASM

    SEE -->|"event"| TRACE
    REASON -->|"event"| TRACE
    ACT -->|"event"| TRACE
    AUDIT -->|"event"| TRACE

    classDef inputStyle fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef seeStyle fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef reasonStyle fill:#fff3e0,stroke:#e65100,color:#bf360c
    classDef actStyle fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef auditStyle fill:#fce4ec,stroke:#c62828,color:#b71c1c
    classDef memStyle fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef toolStyle fill:#f1f8e9,stroke:#558b2f,color:#33691e
    classDef obsStyle fill:#fff8e1,stroke:#f9a825,color:#f57f17

    class CAM,TXT,DOCS,TEL inputStyle
    class CV,STRUCT_OBS seeStyle
    class CTX_ASM,MF,PLANNER,CKPT reasonStyle
    class TOOL_GEN,TOOL_EXEC actStyle
    class VALIDATE,GATE auditStyle
    class SQLITE,VEC,CKPT_STORE memStyle
    class FS,HTTP,SH,PLUG toolStyle
    class TRACE,METRICS,DASH obsStyle
```

## Data Flow

1. **Inputs** (camera, text, docs, telemetry) enter the **SEE** stage
2. **SEE** runs real OpenCV + MediaPipe pipelines, outputs structured JSON observations
3. **REASON** retrieves context from **InstaBrain** (RAG), selects MatFormer profile, plans via Gemma
4. **ACT** converts the plan into executable tool calls (Gemma → JSON)
5. **AUDIT** (State Watchdog) validates each action for safety/policy before execution
6. Approved actions run through the **Tool Registry** (filesystem, HTTP, shell — all allowlisted)
7. Results flow back to REASON and are stored in InstaBrain
8. **Contextual Checkpointing** periodically summarizes session state into compact checkpoints
9. **InstaControl** traces every stage as JSONL events → aggregates metrics → renders the Streamlit dashboard

## Future Extensions

- **Offline Buffer**: queue actions when connectivity drops; replay when back online
- **Local-First Escalation**: attempt local resolution before cloud fallback
- **Multi-Device Federation**: edge mesh for distributed agent coordination
- **Auto-Scaling MatFormer**: dynamically adjust granularity based on real-time CPU/RAM
- **Secure Enclave**: isolate sensitive tool execution in a sandboxed subprocess
