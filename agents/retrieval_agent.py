"""Retrieval agent — queries InstaBrain for relevant context.

Fetches top-k memories and checkpoints based on semantic similarity
to the current query. The value of k is governed by the active
MatFormer granularity profile.
"""

import time

from config import AppConfig
from memory.instabrain_db import InstaBrainDB
from observability.event_trace import EventTracer


class RetrievalAgent:
    def __init__(self, config: AppConfig, db: InstaBrainDB, tracer: EventTracer):
        self.config = config
        self.db = db
        self.tracer = tracer

    def retrieve(
        self,
        query: str,
        session_id: str = "",
        correlation_id: str = "",
        top_k: int | None = None,
    ) -> dict:
        if top_k is None:
            top_k = self.config.matformer.profile["context_k"]

        start = time.perf_counter()
        memories = self.db.query_memories(query, top_k=top_k)
        checkpoints = self.db.query_checkpoints(query, top_k=min(3, top_k))
        latency = (time.perf_counter() - start) * 1000

        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="REASON",
            latency_ms=latency,
            tool_calls=[
                {"name": "retrieval_query", "status": "ok", "latency_ms": latency},
            ],
            decision={
                "type": "retrieval",
                "status": "success",
                "reason": (
                    f"Retrieved {len(memories)} memories, "
                    f"{len(checkpoints)} checkpoints"
                ),
            },
        )

        return {
            "memories": memories,
            "checkpoints": checkpoints,
            "query": query,
            "latency_ms": latency,
        }
