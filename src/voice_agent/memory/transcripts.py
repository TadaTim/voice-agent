"""Transcript store with a token index.

Every turn of every call is recorded here. Transcripts are what gets
attached to human escalations -- and they are in scope for deletion: when
a caller deletes their data, transcripts and every index entry derived
from them are removed, and the deletion scan verifies it.
"""

from __future__ import annotations

import re
import sqlite3

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _index_terms(text: str) -> set[str]:
    terms = {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2}
    digits = "".join(c for c in text if c.isdigit())
    if len(digits) >= 7:
        terms.add(digits)  # phone-number-shaped terms are indexed whole
    return terms


class TranscriptStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS transcripts(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       session_id TEXT NOT NULL,
                       caller_id TEXT NOT NULL,
                       ts TEXT NOT NULL,
                       role TEXT NOT NULL,
                       text TEXT NOT NULL)"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS transcript_index(
                       term TEXT NOT NULL,
                       transcript_id INTEGER NOT NULL,
                       caller_id TEXT NOT NULL)"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcripts_caller"
                " ON transcripts(caller_id)"
            )

    def close(self) -> None:
        self._conn.close()

    def append(self, *, session_id: str, caller_id: str, role: str, text: str, ts: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO transcripts(session_id, caller_id, ts, role, text)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, caller_id, ts, role, text),
            )
            tid = cur.lastrowid
            for term in _index_terms(text):
                self._conn.execute(
                    "INSERT INTO transcript_index(term, transcript_id, caller_id)"
                    " VALUES (?, ?, ?)",
                    (term, tid, caller_id),
                )
        return tid

    def for_session(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM transcripts WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def for_caller(self, caller_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM transcripts WHERE caller_id = ? ORDER BY id",
            (caller_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_caller(self, caller_id: str) -> dict[str, int]:
        with self._conn:
            cur_idx = self._conn.execute(
                "DELETE FROM transcript_index WHERE caller_id = ?", (caller_id,)
            )
            cur_tx = self._conn.execute(
                "DELETE FROM transcripts WHERE caller_id = ?", (caller_id,)
            )
        return {"transcripts": cur_tx.rowcount, "transcript_index": cur_idx.rowcount}

    def count_identifier_matches(self, *, caller_id: str, name: str, phone_digits: str) -> int:
        """Residual scan: transcript rows or index entries still linked to
        the caller or still containing their name / phone digits."""
        n = 0
        name_l = name.strip().lower()
        for row in self._conn.execute("SELECT * FROM transcripts").fetchall():
            if row["caller_id"] == caller_id:
                n += 1
                continue
            text_l = row["text"].lower()
            digits = "".join(c for c in text_l if c.isdigit())
            if (name_l and name_l in text_l) or (phone_digits and phone_digits in digits):
                n += 1
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_index WHERE caller_id = ?",
            (caller_id,),
        ).fetchone()
        n += row["n"]
        if phone_digits:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM transcript_index WHERE term = ?",
                (phone_digits,),
            ).fetchone()
            n += row["n"]
        return n
