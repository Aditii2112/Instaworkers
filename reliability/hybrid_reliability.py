"""Hybrid reliability layer: local-first validation + optional teacher fallback."""

from __future__ import annotations

import json
import math
import time

from config import AppConfig
from llm.gemini_runtime import GeminiRuntime
from memory.instabrain_db import InstaBrainDB
from observability.event_trace import EventTracer
from tools.tool_registry import ToolRegistry


class HybridReliabilityManager:
    def __init__(
        self,
        config: AppConfig,
        db: InstaBrainDB,
        tracer: EventTracer,
        registry: ToolRegistry,
        gemini: GeminiRuntime | None = None,
    ):
        self.config = config
        self.db = db
        self.tracer = tracer
        self.registry = registry
        self.gemini = gemini

    def evaluate_actions(
        self,
        tool_calls: list[dict],
        session_id: str,
        correlation_id: str,
    ) -> dict:
        """Gate tool calls using confidence threshold + teacher escalation."""
        if not self.config.hybrid.enabled or not tool_calls:
            return {
                "allowed_calls": tool_calls,
                "blocked_calls": [],
                "queued_actions": [],
                "validation_records": [],
                "connectivity": {"state": "disabled"},
            }

        start = time.perf_counter()
        connectivity_state = self._connectivity_state()
        allowed_calls: list[dict] = []
        blocked_calls: list[dict] = []
        queued_actions: list[dict] = []
        records: list[dict] = []

        for idx, call in enumerate(tool_calls):
            assessment = self._local_assessment(call)
            payload = {
                "tool_call": call,
                "local_assessment": assessment,
                "policy": {
                    "confidence_threshold": self.config.hybrid.confidence_threshold,
                    "high_risk_budget_usd": self.config.hybrid.high_risk_budget_usd,
                },
            }

            if assessment["high_risk"]:
                pending_id = self.db.insert_pending_validation(
                    session_id=session_id,
                    correlation_id=correlation_id,
                    action_index=idx,
                    tool_name=call.get("tool", ""),
                    payload=payload,
                    local_confidence=assessment["confidence"],
                    local_reason=assessment["reason"],
                    state="blocked_human_review",
                )
                blocked_calls.append(
                    {
                        "index": idx,
                        "tool": call.get("tool", ""),
                        "reason": (
                            "High-risk action exceeds board responsibility limit; "
                            "requires human review."
                        ),
                    }
                )
                records.append(
                    {
                        "index": idx,
                        "tool": call.get("tool", ""),
                        "state": "blocked_human_review",
                        "pending_id": pending_id,
                        "local_confidence": assessment["confidence"],
                        "reason": assessment["reason"],
                    }
                )
                continue

            if not assessment["needs_teacher"]:
                allowed_calls.append(call)
                records.append(
                    {
                        "index": idx,
                        "tool": call.get("tool", ""),
                        "state": "local_only",
                        "local_confidence": assessment["confidence"],
                        "reason": assessment["reason"],
                    }
                )
                continue

            if connectivity_state == "online" and self.gemini:
                teacher_verdict = self._teacher_validate(payload)
                if teacher_verdict["approved"] and not teacher_verdict.get(
                    "requires_human_review", False
                ):
                    allowed_calls.append(call)
                    records.append(
                        {
                            "index": idx,
                            "tool": call.get("tool", ""),
                            "state": "teacher_validated",
                            "local_confidence": assessment["confidence"],
                            "teacher_confidence": teacher_verdict.get("confidence"),
                            "reason": teacher_verdict.get("reason"),
                        }
                    )
                else:
                    pending_id = self.db.insert_pending_validation(
                        session_id=session_id,
                        correlation_id=correlation_id,
                        action_index=idx,
                        tool_name=call.get("tool", ""),
                        payload=payload,
                        local_confidence=assessment["confidence"],
                        local_reason=teacher_verdict.get("reason", assessment["reason"]),
                        state=(
                            "blocked_human_review"
                            if teacher_verdict.get("requires_human_review", False)
                            else "rejected"
                        ),
                    )
                    blocked_calls.append(
                        {
                            "index": idx,
                            "tool": call.get("tool", ""),
                            "reason": teacher_verdict.get(
                                "reason", "Teacher validator rejected action"
                            ),
                        }
                    )
                    records.append(
                        {
                            "index": idx,
                            "tool": call.get("tool", ""),
                            "state": (
                                "blocked_human_review"
                                if teacher_verdict.get("requires_human_review", False)
                                else "rejected"
                            ),
                            "pending_id": pending_id,
                            "local_confidence": assessment["confidence"],
                            "teacher_confidence": teacher_verdict.get("confidence"),
                            "reason": teacher_verdict.get("reason"),
                        }
                    )
            else:
                pending_id = self.db.insert_pending_validation(
                    session_id=session_id,
                    correlation_id=correlation_id,
                    action_index=idx,
                    tool_name=call.get("tool", ""),
                    payload=payload,
                    local_confidence=assessment["confidence"],
                    local_reason=assessment["reason"],
                    state="pending_teacher",
                    next_retry_seconds=self.config.hybrid.pending_retry_seconds,
                )
                queued_actions.append(
                    {
                        "index": idx,
                        "tool": call.get("tool", ""),
                        "pending_id": pending_id,
                        "reason": "Low confidence and teacher unavailable; queued.",
                    }
                )
                records.append(
                    {
                        "index": idx,
                        "tool": call.get("tool", ""),
                        "state": "pending_teacher",
                        "pending_id": pending_id,
                        "local_confidence": assessment["confidence"],
                        "reason": assessment["reason"],
                    }
                )

        latency = (time.perf_counter() - start) * 1000
        overall_status = "success"
        if queued_actions:
            overall_status = "queued"
        if blocked_calls:
            overall_status = "blocked"

        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="AUDIT",
            latency_ms=latency,
            decision={
                "type": "hybrid_validation",
                "status": overall_status,
                "reason": (
                    f"allowed={len(allowed_calls)} "
                    f"queued={len(queued_actions)} blocked={len(blocked_calls)}"
                ),
            },
            safety={
                "blocked": bool(blocked_calls),
                "policy": "hybrid_reliability",
            },
            connectivity={"state": connectivity_state},
        )
        return {
            "allowed_calls": allowed_calls,
            "blocked_calls": blocked_calls,
            "queued_actions": queued_actions,
            "validation_records": records,
            "connectivity": {"state": connectivity_state},
        }

    def reconcile_pending(self, session_id: str, correlation_id: str) -> dict:
        """Try to validate queued actions once teacher connectivity is back."""
        if not self.config.hybrid.enabled:
            return {"processed": 0, "validated": 0, "rejected": 0, "still_pending": 0}
        if self._connectivity_state() != "online" or not self.gemini:
            return {"processed": 0, "validated": 0, "rejected": 0, "still_pending": 0}

        pending = self.db.get_pending_validations(
            state="pending_teacher",
            limit=self.config.hybrid.max_pending_process_per_cycle,
        )
        if not pending:
            return {"processed": 0, "validated": 0, "rejected": 0, "still_pending": 0}

        processed = 0
        validated = 0
        rejected = 0
        still_pending = 0
        for row in pending:
            processed += 1
            try:
                verdict = self._teacher_validate(row["payload"])
                if verdict["approved"] and not verdict.get("requires_human_review", False):
                    self.db.update_pending_validation(
                        row_id=row["id"],
                        state="teacher_validated",
                        teacher_verdict=verdict,
                    )
                    validated += 1
                else:
                    self.db.update_pending_validation(
                        row_id=row["id"],
                        state=(
                            "blocked_human_review"
                            if verdict.get("requires_human_review", False)
                            else "rejected"
                        ),
                        teacher_verdict=verdict,
                    )
                    rejected += 1
            except Exception as e:
                self.db.update_pending_validation(
                    row_id=row["id"],
                    state="pending_teacher",
                    error_text=f"{type(e).__name__}: {e}",
                    next_retry_seconds=self.config.hybrid.pending_retry_seconds,
                )
                still_pending += 1

        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="AUDIT",
            latency_ms=0.0,
            decision={
                "type": "teacher_reconcile",
                "status": "success",
                "reason": (
                    f"processed={processed} validated={validated} "
                    f"rejected={rejected} pending={still_pending}"
                ),
            },
            connectivity={"state": "online"},
        )
        return {
            "processed": processed,
            "validated": validated,
            "rejected": rejected,
            "still_pending": still_pending,
        }

    def _teacher_validate(self, payload: dict) -> dict:
        if not self.gemini:
            return {
                "approved": False,
                "confidence": 0.0,
                "reason": "Teacher unavailable",
                "requires_human_review": True,
            }
        try:
            return self.gemini.validate_action(payload)
        except Exception as e:
            return {
                "approved": False,
                "confidence": 0.0,
                "reason": f"Teacher validation failed: {type(e).__name__}: {e}",
                "requires_human_review": True,
            }

    def _connectivity_state(self) -> str:
        return self.gemini.connectivity_state() if self.gemini else "offline"

    def _local_assessment(self, call: dict) -> dict:
        tool_name = call.get("tool", "")
        params = call.get("params", {}) or {}
        tool = self.registry.get(tool_name)

        confidence = 0.9
        reasons: list[str] = []

        if tool is None:
            confidence -= 0.6
            reasons.append("Unknown tool")
        else:
            if tool.requires_approval:
                confidence -= 0.15
                reasons.append("Tool requires approval")

        if not isinstance(params, dict):
            confidence -= 0.35
            reasons.append("Invalid parameter object")
            params = {}
        elif not params:
            confidence -= 0.05
            reasons.append("Empty params")

        max_budget = self._extract_max_budget_value(params)
        high_risk = max_budget is not None and max_budget > self.config.hybrid.high_risk_budget_usd
        if high_risk:
            confidence -= 0.35
            reasons.append(
                f"Budget {max_budget:.2f} exceeds {self.config.hybrid.high_risk_budget_usd:.2f}"
            )

        risky_tokens = ("delete", "shutdown", "terminate", "purchase", "payment")
        if any(tok in tool_name.lower() for tok in risky_tokens):
            confidence -= 0.15
            reasons.append("Risky action keyword")

        confidence = max(0.0, min(1.0, confidence))
        needs_teacher = confidence < self.config.hybrid.confidence_threshold

        return {
            "confidence": float(confidence),
            "needs_teacher": bool(needs_teacher),
            "high_risk": bool(high_risk),
            "estimated_budget": max_budget,
            "reason": "; ".join(reasons) if reasons else "Local validation looks safe",
        }

    def _extract_max_budget_value(self, params: dict) -> float | None:
        values: list[float] = []

        def _walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    key = str(k).lower()
                    if isinstance(v, (int, float)) and any(
                        token in key for token in ("amount", "cost", "price", "budget", "total")
                    ):
                        if not math.isnan(float(v)):
                            values.append(float(v))
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(params)
        return max(values) if values else None

    def queue_counts(self) -> dict:
        return self.db.pending_validation_counts()
