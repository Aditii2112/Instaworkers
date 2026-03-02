# Sovereign Sentinel: The On-Device Industrial Nervous System

**Sovereign Sentinel** is a privacy-first, autonomous multi-agent orchestration platform designed by Team **Instaworkers**. Built for "Connectivity Deserts," the system utilizes a three-model ensemble—anchored by a **surgically fine-tuned PaliGemma 2 (3B)**—to move beyond simple object detection into **Autonomous Facility Governance**.

The system processes 4K video feeds entirely at the edge, identifies anomalies, coordinates with local inventory databases, and audits every action against a safety constitution—**100% offline.**

Demo: https://drive.google.com/file/d/1iWbdTB_gN6dmG4eRcic3kqNkFH-fCl2d/view?usp=sharing
---

##  The $160B Problem Statement

Global supply chains lose approximately **$160 Billion annually** to undetected "Micro-Escalations"—small leaks, pest intrusions, or structural faults that snowball into total batch loss.

* **The "Faraday Cage" Paradox:** 90% of high-risk industrial environments (cold storage, underground basements) are Wi-Fi dead zones where cloud AI is physically non-viable.
* **The Privacy Barrier:** Critical infrastructure policy often prohibits streaming internal security footage to the cloud.
* **Team Mission:** **Instaworkers** set out to build a solution that provides "intelligence at the edge," ensuring safety and efficiency without a single byte leaving the local network.

---

##  The Multi-Agent Architecture (SEE → REASON → ACT → AUDIT)

Our system uses a **Synchronous Handshake** between specialized agents to ensure data integrity and safety.

### 1. The Autonomous Workforce

| Agent | Model | Responsibility |
| --- | --- | --- |
| **Vision (The Eyes)** | **PaliGemma 2 (3B)** | **Fine-Tuned.** Performs multi-class detection (Humans, Pests, Leaks) and spatial reasoning via `mlx-vlm`. |
| **Inventory (Context)** | **Logic-Based (SQL)** | Reconciles visual coordinates with `inventory.db` to identify at-risk SKUs. |
| **Action (The Worker)** | **FunctionGemma (270M)** | Generates Python tool calls for visual dispatch, ticketing, and database updates. |
| **Auditor (Watchdog)** | **Gemma (1B)** | **Policy Enforcement.** Intercepts tool calls to validate against Safety Policies (SENT-001/002/003). |

---

##  Technical Deep Dive: The Fine-Tuned Brain

The core of our "SEE" layer is a **surgically fine-tuned PaliGemma 2 (3B)** developed by **Instaworkers**.

* **Training:** Fine-tuned with LoRA (Low-Rank Adaptation) on a curated dataset of industrial hazards using the `mlx-vlm` framework on Apple Silicon.
* **Surgical Alignment:** Unlike base VLMs, our model is aligned to a strict **Industrial JSON Schema**. It doesn't just describe the scene; it outputs actionable structured data.
* **Sensor Fusion:** Integrated with **MediaPipe** for sub-millisecond human detection, ensuring the `human_present` flag is verified twice for maximum safety.

---

## The Sentinel Workflow

1. **Ingestion:** Processes a video stream at 2–5 FPS via OpenCV.
2. **Detection:** Vision Agent identifies an anomaly (e.g., a "pest" near "Aisle 4").
3. **Cross-Reference:** Inventory Agent identifies "Rack 04: Fresh Produce" is in the contamination zone.
4. **Action:** Action Agent captures the specific video frame as evidence and generates a `ticket.json`.
5. **Audit:** The Auditor Agent checks for human presence. If a worker is in the frame, it **blocks** automated chemical deployment and triggers a **Voice Alert** via MacOS `say`.

---

##  Dashboard: The 'Nervous System' UI

The **Sovereign Sentinel Dashboard** (React/Tailwind) provides a high-stakes command center for warehouse operators:

* **Live Feed:** Real-time video with bounding box overlays for anomalies and workers.
* **Agent Connectivity Widget:** A pulsing brain icon visualizing the data flow between agents.
* **Visual Evidence Gallery:** Instant display of screenshots captured by the agents for human verification.
* **Safety Log:** A scrolling terminal showing the Auditor’s policy checks (e.g., `APPROVED: SENT-002`).

---

##  Industrial Tooling Suite

The agents have access to a specialized registry of tools:

* `dispatch_visual_ticket`: Routes frame-captures to Facilities, Inventory, or Safety teams.
* `quarantine_inventory_sku`: Autonomously updates the SQL database to lock contaminated stock.
* `broadcast_safety_alert`: System-level TTS to warn workers on the floor in real-time.

---

##  Impact & Business Value

* **Efficiency:** Reduces response latency from **minutes to milliseconds**.
* **Edge Native:** Functions in environments with **zero connectivity**.
* **Privacy First:** Zero bytes of video data ever leave the local MacBook Pro.

---

### 🏁 Getting Started

To launch the Sovereign Sentinel ecosystem locally on your MacBook Pro:

```bash
# Seed the local inventory database
python seed_db.py

# Launch the Agentic Workforce & Dashboard
chmod +x start_sentinel.sh
./start_sentinel.sh

```

---

**Team:** Instaworkers

**Technologies:** PaliGemma 2, Gemma, FunctionGemma, mlx-vlm, Ollama, React, OpenCV.
