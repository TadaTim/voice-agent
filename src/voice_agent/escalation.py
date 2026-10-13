"""Human escalation queue.

When the agent escalates, the full transcript of the call is attached so
the human picking it up has complete context. Stored locally in SQLite --
no external ticketing system is required.
"""

from __future__ import annotations

import json
import sqlite3


class EscalationQueue:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS escalations(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts TEXT NOT NULL,
                       caller_id TEXT NOT NULL,
                       session_id TEXT NOT NULL,
                       reason TEXT NOT NULL,
                       priority TEXT NOT NULL DEFAULT 'normal',
                       transcript_json TEXT NOT NULL,
                       status TEXT NOT NULL DEFAULT 'pending')"""
            )

    def close(self) -> None:
        self._conn.close()

    def enqueue(
        self,
        *,
        ts: str,
        caller_id: str,
        session_id: str,
        reason: str,
        priority: str = "normal",
        transcript: list[dict],
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO escalations(ts, caller_id, session_id, reason,"
                " priority, transcript_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    caller_id,
                    session_id,
                    reason,
                    priority,
                    json.dumps(transcript, indent=1),
                ),
            )
        return cur.lastrowid

    def list_pending(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM escalations WHERE status = 'pending' ORDER BY id"
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["transcript"] = json.loads(item.pop("transcript_json"))
            out.append(item)
        return out

    def count_pending(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM escalations WHERE status = 'pending'"
        ).fetchone()
        return row["n"]
