"""InstaBrain: SQLite + vector store for agent memory and checkpoints.

Uses sqlite-vec for hardware-accelerated KNN when available.
Falls back to NumPy cosine similarity otherwise.
"""

import json
import sqlite3
import struct
from pathlib import Path

import numpy as np

from memory.embeddings import EmbeddingEngine


class InstaBrainDB:
    def __init__(self, db_path: str, embedding_engine: EmbeddingEngine):
        self.db_path = db_path
        self.emb = embedding_engine
        self.dim = self.emb.dim
        self._use_vec_ext = False

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_entities TEXT DEFAULT '[]',
                pending_tasks TEXT DEFAULT '[]',
                decisions_made TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_ckpt_session
                ON checkpoints(session_id);
            CREATE TABLE IF NOT EXISTS pending_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                action_index INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                local_confidence REAL NOT NULL,
                local_reason TEXT DEFAULT '',
                state TEXT NOT NULL DEFAULT 'pending_teacher',
                teacher_verdict_json TEXT,
                error_text TEXT,
                next_retry_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_pending_state_retry
                ON pending_validations(state, next_retry_at);
        """)

        try:
            import sqlite_vec  # noqa: F811
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories "
                f"USING vec0(embedding float[{self.dim}])"
            )
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_checkpoints "
                f"USING vec0(embedding float[{self.dim}])"
            )
            self._use_vec_ext = True
        except (ImportError, Exception):
            self._use_vec_ext = False

        self.conn.commit()

    # ── serialization ────────────────────────────────────────────────

    def _pack(self, vec: np.ndarray) -> bytes:
        return struct.pack(f"{self.dim}f", *vec.tolist())

    def _unpack(self, blob: bytes) -> np.ndarray:
        return np.array(struct.unpack(f"{self.dim}f", blob), dtype=np.float32)

    # ── memories ─────────────────────────────────────────────────────

    def insert_memory(self, content: str, metadata: dict | None = None) -> int:
        vec = self.emb.embed(content)
        vec_bytes = self._pack(vec)

        cur = self.conn.execute(
            "INSERT INTO memories (content, metadata, embedding) VALUES (?, ?, ?)",
            (content, json.dumps(metadata or {}), vec_bytes),
        )
        row_id = cur.lastrowid

        if self._use_vec_ext:
            self.conn.execute(
                "INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)",
                (row_id, vec_bytes),
            )

        self.conn.commit()
        return row_id

    def query_memories(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.emb.embed(query)

        if self._use_vec_ext:
            return self._vec_search_memories(query_vec, top_k)
        return self._fallback_search("memories", query_vec, top_k)

    def _vec_search_memories(self, vec: np.ndarray, top_k: int) -> list[dict]:
        vec_bytes = self._pack(vec)
        rows = self.conn.execute(
            "SELECT rowid, distance FROM vec_memories "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (vec_bytes, top_k),
        ).fetchall()

        results = []
        for row in rows:
            mem = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (row["rowid"],)
            ).fetchone()
            if mem:
                results.append({
                    "id": mem["id"],
                    "content": mem["content"],
                    "metadata": json.loads(mem["metadata"]),
                    "created_at": mem["created_at"],
                    "distance": row["distance"],
                })
        return results

    # ── checkpoints ──────────────────────────────────────────────────

    def insert_checkpoint(
        self,
        session_id: str,
        summary: str,
        key_entities: list | None = None,
        pending_tasks: list | None = None,
        decisions_made: list | None = None,
    ) -> int:
        vec = self.emb.embed(summary)
        vec_bytes = self._pack(vec)

        cur = self.conn.execute(
            "INSERT INTO checkpoints "
            "(session_id, summary, key_entities, pending_tasks, decisions_made, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                summary,
                json.dumps(key_entities or []),
                json.dumps(pending_tasks or []),
                json.dumps(decisions_made or []),
                vec_bytes,
            ),
        )
        row_id = cur.lastrowid

        if self._use_vec_ext:
            self.conn.execute(
                "INSERT INTO vec_checkpoints (rowid, embedding) VALUES (?, ?)",
                (row_id, vec_bytes),
            )

        self.conn.commit()
        return row_id

    def get_recent_checkpoints(self, session_id: str, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "summary": r["summary"],
                "key_entities": json.loads(r["key_entities"]),
                "pending_tasks": json.loads(r["pending_tasks"]),
                "decisions_made": json.loads(r["decisions_made"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def query_checkpoints(self, query: str, top_k: int = 3) -> list[dict]:
        vec = self.emb.embed(query)

        if self._use_vec_ext:
            return self._vec_search_checkpoints(vec, top_k)
        return self._fallback_search_checkpoints(vec, top_k)

    def _vec_search_checkpoints(self, vec: np.ndarray, top_k: int) -> list[dict]:
        vec_bytes = self._pack(vec)
        rows = self.conn.execute(
            "SELECT rowid, distance FROM vec_checkpoints "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (vec_bytes, top_k),
        ).fetchall()

        results = []
        for row in rows:
            ckpt = self.conn.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (row["rowid"],)
            ).fetchone()
            if ckpt:
                results.append({
                    "id": ckpt["id"],
                    "summary": ckpt["summary"],
                    "session_id": ckpt["session_id"],
                    "distance": row["distance"],
                })
        return results

    # ── fallback (no sqlite-vec) ─────────────────────────────────────

    def _fallback_search(
        self, table: str, query_vec: np.ndarray, top_k: int
    ) -> list[dict]:
        rows = self.conn.execute(
            f"SELECT * FROM {table} WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for row in rows:
            stored_vec = self._unpack(row["embedding"])
            similarity = float(np.dot(query_vec, stored_vec))
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, row in scored[:top_k]:
            entry = {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "distance": 1.0 - sim,
            }
            if "metadata" in row.keys():
                entry["metadata"] = json.loads(row["metadata"])
            results.append(entry)
        return results

    def _fallback_search_checkpoints(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM checkpoints WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for row in rows:
            stored_vec = self._unpack(row["embedding"])
            similarity = float(np.dot(query_vec, stored_vec))
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r["id"],
                "summary": r["summary"],
                "session_id": r["session_id"],
                "distance": 1.0 - s,
            }
            for s, r in scored[:top_k]
        ]

    # ── hybrid reliability pending validations ───────────────────────

    def insert_pending_validation(
        self,
        session_id: str,
        correlation_id: str,
        action_index: int,
        tool_name: str,
        payload: dict,
        local_confidence: float,
        local_reason: str,
        state: str = "pending_teacher",
        next_retry_seconds: int | None = None,
    ) -> int:
        next_retry_expr = (
            f"datetime('now', '+{int(next_retry_seconds)} seconds')"
            if next_retry_seconds
            else "datetime('now')"
        )
        cur = self.conn.execute(
            f"""
            INSERT INTO pending_validations (
                session_id, correlation_id, action_index, tool_name, payload_json,
                local_confidence, local_reason, state, next_retry_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, {next_retry_expr})
            """,
            (
                session_id,
                correlation_id,
                action_index,
                tool_name,
                json.dumps(payload),
                float(local_confidence),
                local_reason,
                state,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_pending_validations(
        self, state: str = "pending_teacher", limit: int = 20
    ) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM pending_validations
            WHERE state = ?
              AND datetime(COALESCE(next_retry_at, datetime('now'))) <= datetime('now')
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (state, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "correlation_id": r["correlation_id"],
                "action_index": r["action_index"],
                "tool_name": r["tool_name"],
                "payload": json.loads(r["payload_json"]),
                "local_confidence": r["local_confidence"],
                "local_reason": r["local_reason"],
                "state": r["state"],
                "teacher_verdict": (
                    json.loads(r["teacher_verdict_json"])
                    if r["teacher_verdict_json"]
                    else None
                ),
                "error_text": r["error_text"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def update_pending_validation(
        self,
        row_id: int,
        state: str,
        teacher_verdict: dict | None = None,
        error_text: str | None = None,
        next_retry_seconds: int | None = None,
    ) -> None:
        next_retry_expr = (
            f", next_retry_at = datetime('now', '+{int(next_retry_seconds)} seconds')"
            if next_retry_seconds
            else ", next_retry_at = datetime('now')"
        )
        self.conn.execute(
            f"""
            UPDATE pending_validations
            SET state = ?,
                teacher_verdict_json = ?,
                error_text = ?,
                updated_at = datetime('now')
                {next_retry_expr}
            WHERE id = ?
            """,
            (
                state,
                json.dumps(teacher_verdict) if teacher_verdict is not None else None,
                error_text,
                row_id,
            ),
        )
        self.conn.commit()

    def pending_validation_counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT state, COUNT(*) AS count FROM pending_validations GROUP BY state"
        ).fetchall()
        return {r["state"]: r["count"] for r in rows}

    def close(self):
        self.conn.close()
