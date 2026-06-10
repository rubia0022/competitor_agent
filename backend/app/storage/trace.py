"""
SQLite trace 存储：每个 Agent 的调用都落库，支持端到端可追溯。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    industry TEXT,
    competitors_json TEXT,
    status TEXT,
    rounds INTEGER DEFAULT 0,
    created_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    msg_id TEXT PRIMARY KEY,
    task_id TEXT,
    from_agent TEXT,
    to_agent TEXT,
    intent TEXT,
    payload_json TEXT,
    parent_msg_id TEXT,
    round_no INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    agent TEXT,
    round_no INTEGER,
    prompt TEXT,
    input_payload TEXT,
    output_payload TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    status TEXT,
    error_msg TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS task_states (
    task_id TEXT PRIMARY KEY,
    state_json TEXT,
    failed_node TEXT,
    created_at TEXT
);
"""


class TraceStore:
    """轻量 SQLite 封装。线程安全（演示场景够用）。"""

    def __init__(self, db_path: Path, langfuse=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.langfuse = langfuse
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    # ---- Task ----
    def create_task(self, task_id: str, industry: str, competitors: List[str]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks (task_id, industry, competitors_json, status, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (task_id, industry, json.dumps(competitors, ensure_ascii=False), "running", datetime.utcnow().isoformat()),
            )
        if self.langfuse:
            self.langfuse.trace_task(task_id, industry, competitors)

    def finish_task(self, task_id: str, status: str, rounds: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, rounds=?, finished_at=? WHERE task_id=?",
                (status, rounds, datetime.utcnow().isoformat(), task_id),
            )
        if self.langfuse:
            self.langfuse.flush()

    # ---- Message ----
    def write_message(self, msg: Dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO messages "
                "(msg_id, task_id, from_agent, to_agent, intent, payload_json, parent_msg_id, round_no, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    msg["msg_id"],
                    msg["task_id"],
                    msg["from_agent"],
                    msg["to_agent"],
                    msg["intent"],
                    json.dumps(msg.get("payload", {}), ensure_ascii=False, default=str),
                    msg.get("parent_msg_id"),
                    msg.get("round_no", 0),
                    msg.get("created_at", datetime.utcnow().isoformat()),
                ),
            )

    # ---- Trace ----
    def write(
        self,
        *,
        task_id: str,
        agent: str,
        round_no: int,
        prompt: str,
        input_payload: str,
        output_payload: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        status: str,
        error_msg: Optional[str] = None,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO traces "
                "(task_id, agent, round_no, prompt, input_payload, output_payload,"
                " tokens_in, tokens_out, latency_ms, status, error_msg, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    agent,
                    round_no,
                    prompt,
                    input_payload,
                    output_payload,
                    tokens_in,
                    tokens_out,
                    latency_ms,
                    status,
                    error_msg,
                    datetime.utcnow().isoformat(),
                ),
            )
        if self.langfuse:
            self.langfuse.span_agent(
                task_id=task_id, agent=agent, round_no=round_no,
                input_payload=input_payload, output_payload=output_payload,
                status=status, latency_ms=latency_ms,
            )

    # ---- Query ----
    def list_traces(self, task_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM traces WHERE task_id=? ORDER BY id ASC", (task_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_messages(self, task_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE task_id=? ORDER BY created_at ASC", (task_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- State (人工介入) ----
    def save_state(self, task_id: str, state: Dict[str, Any], failed_node: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO task_states (task_id, state_json, failed_node, created_at)"
                " VALUES (?, ?, ?, ?)",
                (task_id, json.dumps(state, ensure_ascii=False, default=str), failed_node, datetime.utcnow().isoformat()),
            )

    def load_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM task_states WHERE task_id=?", (task_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_state(self, task_id: str, patches: Dict[str, Any]) -> None:
        saved = self.load_state(task_id)
        if not saved:
            return
        state = json.loads(saved["state_json"])
        state.update(patches)
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE task_states SET state_json=? WHERE task_id=?",
                (json.dumps(state, ensure_ascii=False, default=str), task_id),
            )