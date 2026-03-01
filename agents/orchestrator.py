"""Root Orchestrator — Sovereign Sentinel See → Reason → Act → Audit loop.

Coordinates all specialist agents through the autonomous event loop:
  1. SEE: VisionAgent processes video at 2 FPS via PaliGemma 2 + MediaPipe
  2. REASON: Retrieval → Reasoning (with inventory context injection)
  3. ACT: ActionAgent generates tool calls → ToolRunner executes
  4. AUDIT: AuditorAgent enforces SENT-001/002/003 policies

Supports SSE streaming for real-time dashboard updates.
"""

import json
import time
import uuid
from pathlib import Path

from config import AppConfig, MATFORMER_PROFILES
from llm.gemma_runtime import GemmaRuntime
from llm.gemini_runtime import GeminiRuntime
from llm.prompts import CHECKPOINT_SYSTEM_PROMPT
from memory.instabrain_db import InstaBrainDB
from observability.event_trace import EventTracer
from reliability.hybrid_reliability import HybridReliabilityManager
from tools.tool_registry import ToolRegistry
from tools.tool_runner import ToolRunner

from agents.video_processor import VideoProcessor
from agents.vision_agent import VisionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.action_agent import ActionAgent
from agents.auditor_agent import AuditorAgent
from agents.inventory_agent import query_skus_at_location


class RootOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        gemma: GemmaRuntime,
        db: InstaBrainDB,
        tracer: EventTracer,
        registry: ToolRegistry,
        runner: ToolRunner,
        gemini: GeminiRuntime | None = None,
    ):
        self.config = config
        self.gemma = gemma
        self.db = db
        self.tracer = tracer
        self.registry = registry
        self.runner = runner
        self.gemini = gemini

        self.video_processor = VideoProcessor(
            sample_fps=config.video.fps_target,
            max_frames=config.video.max_frames,
        )
        self.vision = VisionAgent(config, tracer, gemma=gemma)
        self.retrieval = RetrievalAgent(config, db, tracer)
        self.reasoning = ReasoningAgent(config, gemma, tracer)
        self.action = ActionAgent(config, gemma, tracer)
        self.auditor = (
            AuditorAgent(config, gemma, tracer)
            if config.auditor.enabled
            else None
        )
        self.reliability = HybridReliabilityManager(
            config=config,
            db=db,
            tracer=tracer,
            registry=registry,
            gemini=gemini,
        )

        self._cycle_count = 0
        self._session_id = str(uuid.uuid4())

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self):
        self._session_id = str(uuid.uuid4())
        self._cycle_count = 0

    # ── Main Cycle ───────────────────────────────────────────────────

    def run_cycle(
        self,
        frame=None,
        image_path: str | None = None,
        text_input: str | None = None,
        video_path: str | None = None,
    ) -> dict:
        """Execute one full See → Reason → Act → Audit cycle.

        When video_path is provided, processes the video at 2 FPS and aggregates
        observations from all frames. The worst anomaly drives the pipeline.
        """
        correlation_id = str(uuid.uuid4())
        result: dict = {
            "correlation_id": correlation_id,
            "session_id": self._session_id,
            "stages_completed": [],
        }

        reconcile = self.reliability.reconcile_pending(
            session_id=self._session_id,
            correlation_id=correlation_id,
        )
        if reconcile["processed"] > 0:
            result["pending_reconcile"] = reconcile

        # ── VIDEO PROCESSING ─────────────────────────────────────────
        all_frame_observations: list[dict] = []
        if video_path and VideoProcessor.is_video(video_path):
            result["video_info"] = self.video_processor.get_video_info(video_path)

            frames_dir = str(Path(video_path).parent / "frames")
            saved_frames = self.video_processor.extract_and_save(video_path, frames_dir)
            result["frames_extracted"] = len(saved_frames)

            for sf in saved_frames:
                obs = self.vision.observe(
                    image_path=sf["path"],
                    text_input=text_input,
                    session_id=self._session_id,
                    correlation_id=correlation_id,
                )
                obs["timestamp"] = sf["timestamp_s"]
                obs["frame_path"] = sf["path"]
                all_frame_observations.append(obs)

            if saved_frames:
                image_path = saved_frames[0]["path"]

        # ── SEE ──────────────────────────────────────────────────────
        if all_frame_observations:
            observations = self._aggregate_observations(all_frame_observations)
            observations["all_frames"] = all_frame_observations
        else:
            observations = self.vision.observe(
                frame=frame,
                image_path=image_path,
                text_input=text_input,
                session_id=self._session_id,
                correlation_id=correlation_id,
            )

        if text_input:
            observations["text_input"] = text_input

        result["observations"] = observations
        result["stages_completed"].append("SEE")

        anomaly = observations.get("anomaly", "none")
        human_present = observations.get("human_present", False)
        severity = observations.get("severity", "low")
        location = observations.get("location", "unknown")

        # ── INVENTORY CHECK (on anomaly) ─────────────────────────────
        inventory_context = []
        if anomaly and anomaly != "none" and location != "unknown":
            skus = query_skus_at_location(location)
            inventory_context = skus
            result["inventory_at_location"] = skus

        # ── SAFETY BROADCAST (critical + human) ─────────────────────
        if human_present and severity == "critical" and anomaly != "none":
            try:
                team = self._route_team(anomaly)
                br = self.runner.execute(
                    "broadcast_safety_alert",
                    {"location": location, "team": team},
                )
                result["safety_alert_triggered"] = br
            except Exception as e:
                result["safety_alert_error"] = str(e)

        # ── MATFORMER GRANULARITY ────────────────────────────────────
        query = self._observations_to_query(observations)
        granularity, granularity_reason = self._select_granularity(observations, query)
        profile = MATFORMER_PROFILES.get(granularity, MATFORMER_PROFILES["M"])
        result["matformer"] = {
            "selected_granularity": granularity,
            "reason": granularity_reason,
            "profile": {
                "max_tokens": profile["max_tokens"],
                "context_k": profile["context_k"],
                "temperature": profile["temperature"],
            },
        }

        # ── RETRIEVE (part of REASON) ────────────────────────────────
        # For video-based analysis, skip InstaBrain retrieval entirely.
        # Old memories from previous sessions are irrelevant to fresh video frames.
        # Only retrieve context for text-only queries (no video uploaded).
        has_video = bool(video_path) or bool(all_frame_observations)
        if has_video:
            context = {"memories": [], "checkpoints": []}
            if inventory_context:
                context["memories"].append(
                    {"content": f"Inventory at {location}: {json.dumps(inventory_context, default=str)}", "distance": 0}
                )
        else:
            context = self.retrieval.retrieve(
                query=query,
                session_id=self._session_id,
                correlation_id=correlation_id,
                top_k=profile["context_k"],
            )
            if inventory_context:
                mems = list(context.get("memories", []))
                mems.insert(
                    0,
                    {"content": f"Inventory at {location}: {json.dumps(inventory_context, default=str)}", "distance": 0},
                )
                context = {**context, "memories": mems}
        result["context"] = context

        # ── REASON ───────────────────────────────────────────────────
        tool_descriptions = self.registry.get_schema_prompt()
        plan = self.reasoning.reason(
            observations=observations,
            context=context,
            tool_descriptions=tool_descriptions,
            session_id=self._session_id,
            correlation_id=correlation_id,
            profile_override=profile,
            granularity_override=granularity,
        )
        result["plan"] = plan
        result["stages_completed"].append("REASON")

        # ── ACT ──────────────────────────────────────────────────────
        tool_calls = self.action.act(
            plan=plan,
            tool_descriptions=tool_descriptions,
            session_id=self._session_id,
            correlation_id=correlation_id,
            profile_override=profile,
            granularity_override=granularity,
        )

        evidence_path = image_path
        if all_frame_observations:
            anomaly_frame = self._find_anomaly_frame(all_frame_observations)
            if anomaly_frame:
                evidence_path = anomaly_frame.get("frame_path", evidence_path)

        for tc in tool_calls:
            if tc.get("tool") == "dispatch_visual_ticket":
                params = tc.setdefault("params", {})
                if not params.get("image_path") and evidence_path:
                    params["image_path"] = evidence_path
                if video_path:
                    params["video_path"] = video_path

        result["tool_calls"] = tool_calls
        result["stages_completed"].append("ACT")

        # ── AUDIT ────────────────────────────────────────────────────
        if self.auditor and tool_calls:
            anomaly_count = sum(
                1 for o in all_frame_observations
                if o.get("anomaly", "none") != "none"
            ) if all_frame_observations else (1 if anomaly != "none" else 0)

            self.auditor.set_anomaly_count(anomaly_count)

            system_state = {
                "source": observations.get("source", "unknown"),
                "text_input": text_input or "",
                "plan_analysis": plan.get("analysis", ""),
                "plan_rationale": plan.get("rationale", ""),
                "available_tools": self.registry.list_names(),
                "session_cycle": self._cycle_count + 1,
                "human_present": human_present,
                "anomaly": anomaly,
                "severity": severity,
                "location": location,
                "anomaly_count": anomaly_count,
                "observations": observations,
            }
            verdict = self.auditor.audit(
                tool_calls=tool_calls,
                system_state=system_state,
                session_id=self._session_id,
                correlation_id=correlation_id,
            )
            result["audit"] = verdict
            result["stages_completed"].append("AUDIT")

            if not verdict["approved"]:
                blocked_indices = set(verdict.get("blocked_actions", []))
                tool_calls = [
                    tc for i, tc in enumerate(tool_calls)
                    if i not in blocked_indices
                ]
                result["tool_calls_after_audit"] = tool_calls

        # ── HYBRID RELIABILITY GATE ──────────────────────────────────
        gate = self.reliability.evaluate_actions(
            tool_calls=tool_calls,
            session_id=self._session_id,
            correlation_id=correlation_id,
        )
        result["validation"] = {
            "records": gate["validation_records"],
            "queued": gate["queued_actions"],
            "blocked": gate["blocked_calls"],
            "connectivity": gate["connectivity"],
        }
        allowed_calls = gate["allowed_calls"]
        result["tool_calls_after_validation"] = allowed_calls

        # ── EXECUTE TOOLS ────────────────────────────────────────────
        if allowed_calls:
            exec_results = self.runner.execute_batch(allowed_calls)
            result["tool_results"] = exec_results

            summary = json.dumps({
                "query": query,
                "anomaly": anomaly,
                "severity": severity,
                "location": location,
                "tool_results": [
                    {"tool": r["tool"], "status": r["status"]}
                    for r in exec_results
                ],
            }, default=str)
            self.db.insert_memory(
                summary, metadata={"correlation_id": correlation_id}
            )

        # ── CHECKPOINT ───────────────────────────────────────────────
        self._cycle_count += 1
        if (
            self.config.checkpoint.enabled
            and self._cycle_count % self.config.checkpoint.interval_calls == 0
        ):
            self._do_checkpoint(correlation_id)

        return result

    # ── SSE Stream ───────────────────────────────────────────────────

    def run_cycle_stream(
        self,
        frame=None,
        image_path: str | None = None,
        text_input: str | None = None,
        video_path: str | None = None,
    ):
        """Generator that yields SSE events as the cycle progresses.

        Yields dicts: {"stage": str, "data": dict}
        Dashboard can consume these in real time.
        """
        correlation_id = str(uuid.uuid4())

        yield {"stage": "START", "data": {"correlation_id": correlation_id, "session_id": self._session_id}}

        # Video frame extraction
        all_frame_obs = []
        if video_path and VideoProcessor.is_video(video_path):
            video_info = self.video_processor.get_video_info(video_path)
            yield {"stage": "VIDEO_INFO", "data": video_info}

            frames_dir = str(Path(video_path).parent / "frames")
            saved_frames = self.video_processor.extract_and_save(video_path, frames_dir)

            for i, sf in enumerate(saved_frames):
                obs = self.vision.observe(
                    image_path=sf["path"],
                    text_input=text_input,
                    session_id=self._session_id,
                    correlation_id=correlation_id,
                )
                obs["timestamp"] = sf["timestamp_s"]
                obs["frame_path"] = sf["path"]
                all_frame_obs.append(obs)

                yield {
                    "stage": "SEE_FRAME",
                    "data": {
                        "frame_index": i,
                        "total_frames": len(saved_frames),
                        "timestamp_s": sf["timestamp_s"],
                        "anomaly": obs.get("anomaly", "none"),
                        "severity": obs.get("severity", "low"),
                        "location": obs.get("location", "unknown"),
                        "human_present": obs.get("human_present", False),
                    },
                }

            if saved_frames:
                image_path = saved_frames[0]["path"]

        if all_frame_obs:
            observations = self._aggregate_observations(all_frame_obs)
        else:
            observations = self.vision.observe(
                frame=frame,
                image_path=image_path,
                text_input=text_input,
                session_id=self._session_id,
                correlation_id=correlation_id,
            )

        if text_input:
            observations["text_input"] = text_input

        yield {"stage": "SEE", "data": observations}

        anomaly = observations.get("anomaly", "none")
        location = observations.get("location", "unknown")

        if anomaly != "none" and location != "unknown":
            skus = query_skus_at_location(location)
            yield {"stage": "INVENTORY", "data": {"location": location, "skus": skus}}

        yield {"stage": "REASON", "data": {"status": "planning"}}
        yield {"stage": "ACT", "data": {"status": "executing"}}
        yield {"stage": "AUDIT", "data": {"status": "checking"}}
        yield {"stage": "DONE", "data": {"correlation_id": correlation_id}}

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_observations(frame_observations: list[dict]) -> dict:
        """Aggregate per-frame observations into a single summary.

        Collects ALL distinct anomaly types seen across all frames,
        picks the highest-severity frame as the base.
        """
        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        worst = frame_observations[0].copy()
        worst_rank = severity_rank.get(worst.get("severity", "low"), 1)
        anomaly_types = set()
        total_faces = 0
        total_hands = 0
        any_human = False

        for obs in frame_observations:
            a = obs.get("anomaly", "none")
            if a != "none":
                anomaly_types.add(a)
            # Also merge per-frame multi-anomaly lists
            for af in obs.get("all_anomalies_in_frame", []):
                anomaly_types.add(af)
            rank = severity_rank.get(obs.get("severity", "low"), 1)
            if rank > worst_rank:
                worst = obs.copy()
                worst_rank = rank
            total_faces += obs.get("face_count", 0)
            total_hands += obs.get("hand_count", 0)
            if obs.get("human_present"):
                any_human = True
            if obs.get("vlm_human_detected"):
                any_human = True

        worst["human_present"] = any_human
        worst["anomaly_types_seen"] = sorted(anomaly_types)
        worst["frames_analyzed"] = len(frame_observations)
        worst["total_faces_across_frames"] = total_faces
        worst["total_hands_across_frames"] = total_hands
        return worst

    @staticmethod
    def _find_anomaly_frame(frame_observations: list[dict]) -> dict | None:
        """Find the frame with the highest-severity anomaly for evidence capture."""
        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        best = None
        best_rank = 0
        for obs in frame_observations:
            if obs.get("anomaly", "none") == "none":
                continue
            rank = severity_rank.get(obs.get("severity", "low"), 1)
            if rank > best_rank:
                best = obs
                best_rank = rank
        return best

    @staticmethod
    def _route_team(anomaly: str) -> str:
        routing = {
            "leak": "Facilities",
            "spoilage": "Inventory",
            "pest": "Pest Control",
            "spill": "Facilities",
            "damage": "Facilities",
            "hazard": "Facilities",
            "fire": "Facilities",
            "smoke": "Facilities",
        }
        return routing.get(anomaly.lower(), "Facilities")

    @staticmethod
    def _observations_to_query(obs: dict) -> str:
        parts: list[str] = []
        if obs.get("text_input"):
            parts.append(obs["text_input"])
        anomaly = obs.get("anomaly", "none")
        if anomaly != "none":
            parts.append(f"Anomaly detected: {anomaly} ({obs.get('severity', 'unknown')} severity) at {obs.get('location', 'unknown')}")
        if obs.get("human_present"):
            parts.append("Worker present in frame")
        if obs.get("face_count", 0) > 0:
            parts.append(f"{obs['face_count']} face(s) detected")
        return "; ".join(parts) if parts else "general observation"

    def _select_granularity(self, observations: dict, query: str) -> tuple[str, str]:
        configured = (self.config.matformer.granularity or "M").upper()
        if configured in MATFORMER_PROFILES and configured != "AUTO":
            return configured, "Manual MATFORMER_GRANULARITY override"

        text = (query or "").lower()
        text_len = len(text)
        has_image = observations.get("source") in {"file", "frame"}
        has_video = observations.get("frames_analyzed", 0) > 0
        memory_pressure = self._cycle_count >= 10

        if has_video or (has_image and text_len > 180):
            return "XL", "Video/multimodal + complex request"
        if has_image or text_len > 220 or memory_pressure:
            return "L", "High context/complexity request"
        if text_len < 80:
            return "S", "Short low-complexity request"
        return "M", "Default medium complexity"

    def _do_checkpoint(self, correlation_id: str):
        recent = self.db.get_recent_checkpoints(self._session_id, limit=3)
        history = (
            "; ".join(c["summary"] for c in recent)
            if recent
            else "No prior checkpoints."
        )

        prompt = (
            f"Session cycle count: {self._cycle_count}\n"
            f"Recent checkpoint summaries: {history}\n"
            "Produce a compact JSON checkpoint of the current session state."
        )

        start = time.perf_counter()
        result = self.gemma.generate(
            prompt=prompt,
            system_prompt=CHECKPOINT_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.3,
        )
        latency = (time.perf_counter() - start) * 1000

        ckpt_data = self._parse_checkpoint(result["text"])
        self.db.insert_checkpoint(
            session_id=self._session_id,
            summary=ckpt_data.get("summary", result["text"]),
            key_entities=ckpt_data.get("key_entities", []),
            pending_tasks=ckpt_data.get("pending_tasks", []),
            decisions_made=ckpt_data.get("decisions_made", []),
        )

        self.tracer.create_event(
            session_id=self._session_id,
            correlation_id=correlation_id,
            stage="REASON",
            latency_ms=latency,
            model=result.get("model", ""),
            model_granularity=self.config.matformer.granularity,
            tokens_in=result.get("tokens_in"),
            tokens_out=result.get("tokens_out"),
            decision={
                "type": "checkpoint",
                "status": "success",
                "reason": f"Checkpoint at cycle {self._cycle_count}",
            },
        )

    @staticmethod
    def _parse_checkpoint(text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return {"summary": text}

    def close(self):
        self.vision.close()
        self.db.close()
        self.gemma.close()
        if self.gemini:
            self.gemini.close()
