"""Per-caller memory store with governance rules.

Rules enforced here (not left to the dialogue layer):

- **Consent gate**: nothing is persisted for a caller without a consent
  record. Assertions made before consent are held by the dialogue layer
  as session state only.
- **Provisional until read-back**: facts asserted by a caller are stored
  as ``provisional``. Only a read-back confirmation promotes a fact to
  ``confirmed``.
- **No silent overwrite**: asserting a different value for a key that has
  a confirmed value flags a *conflict*. The confirmed value stays in
  force until the caller explicitly resolves the conflict; the change is
  then recorded in an audit history.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

STATUS_PROVISIONAL = "provisional"
STATUS_CONFIRMED = "confirmed"


@dataclass
class AssertResult:
    kind: str  # stored_confirmed | stored_provisional | conflict_flagged |
    #            not_stored_no_consent | unchanged
    key: str
    value: str
    existing_value: str | None = None


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS consent(
                       caller_id TEXT PRIMARY KEY,
                       granted INTEGER NOT NULL,
                       updated_ts TEXT NOT NULL)"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memory_facts(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       caller_id TEXT NOT NULL,
                       key TEXT NOT NULL,
                       value TEXT NOT NULL,
                       status TEXT NOT NULL,
                       source TEXT NOT NULL,
                       created_ts TEXT NOT NULL,
                       updated_ts TEXT NOT NULL,
                       UNIQUE(caller_id, key, status))"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memory_conflicts(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       caller_id TEXT NOT NULL,
                       key TEXT NOT NULL,
                       existing_value TEXT NOT NULL,
                       asserted_value TEXT NOT NULL,
                       ts TEXT NOT NULL,
                       resolved INTEGER NOT NULL DEFAULT 0)"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memory_history(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       caller_id TEXT NOT NULL,
                       key TEXT NOT NULL,
                       old_value TEXT NOT NULL,
                       new_value TEXT NOT NULL,
                       ts TEXT NOT NULL,
                       reason TEXT NOT NULL)"""
            )

    def close(self) -> None:
        self._conn.close()

    # -- consent -----------------------------------------------------------

    def get_consent(self, caller_id: str) -> bool:
        row = self._conn.execute(
            "SELECT granted FROM consent WHERE caller_id = ?", (caller_id,)
        ).fetchone()
        return bool(row["granted"]) if row else False

    def set_consent(self, caller_id: str, granted: bool, ts: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO consent(caller_id, granted, updated_ts) VALUES (?, ?, ?)"
                " ON CONFLICT(caller_id) DO UPDATE SET granted = excluded.granted,"
                " updated_ts = excluded.updated_ts",
                (caller_id, 1 if granted else 0, ts),
            )

    # -- facts ---------------------------------------------------------------

    def _get(self, caller_id: str, key: str, status: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM memory_facts WHERE caller_id = ? AND key = ? AND status = ?",
            (caller_id, key, status),
        ).fetchone()
        return dict(row) if row else None

    def _upsert(self, caller_id: str, key: str, value: str, status: str, source: str, ts: str) -> None:
        self._conn.execute(
            "INSERT INTO memory_facts(caller_id, key, value, status, source, created_ts, updated_ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(caller_id, key, status) DO UPDATE SET value = excluded.value,"
            " source = excluded.source, updated_ts = excluded.updated_ts",
            (caller_id, key, value, status, source, ts, ts),
        )

    def assert_fact(
        self,
        caller_id: str,
        key: str,
        value: str,
        *,
        via_readback: bool = False,
        source: str = "caller_assertion",
        ts: str,
    ) -> AssertResult:
        """Record a caller-asserted fact under the governance rules."""
        if not self.get_consent(caller_id):
            return AssertResult("not_stored_no_consent", key, value)
        confirmed = self._get(caller_id, key, STATUS_CONFIRMED)
        if confirmed and confirmed["value"] == value:
            return AssertResult("unchanged", key, value, existing_value=value)
        if confirmed and confirmed["value"] != value:
            # Never silently overwrite: keep confirmed, park the new value
            # as provisional, and flag the conflict for clarification.
            with self._conn:
                self._upsert(caller_id, key, value, STATUS_PROVISIONAL, source, ts)
                self._conn.execute(
                    "INSERT INTO memory_conflicts(caller_id, key, existing_value,"
                    " asserted_value, ts) VALUES (?, ?, ?, ?, ?)",
                    (caller_id, key, confirmed["value"], value, ts),
                )
            return AssertResult(
                "conflict_flagged", key, value, existing_value=confirmed["value"]
            )
        if via_readback:
            with self._conn:
                self._upsert(caller_id, key, value, STATUS_CONFIRMED, source, ts)
                self._conn.execute(
                    "DELETE FROM memory_facts WHERE caller_id = ? AND key = ?"
                    " AND status = ?",
                    (caller_id, key, STATUS_PROVISIONAL),
                )
            return AssertResult("stored_confirmed", key, value)
        with self._conn:
            self._upsert(caller_id, key, value, STATUS_PROVISIONAL, source, ts)
        return AssertResult("stored_provisional", key, value)

    def confirm_fact(self, caller_id: str, key: str, *, ts: str) -> bool:
        """Promote the provisional value for ``key`` to confirmed."""
        provisional = self._get(caller_id, key, STATUS_PROVISIONAL)
        if provisional is None:
            return False
        with self._conn:
            self._upsert(
                caller_id, key, provisional["value"], STATUS_CONFIRMED,
                provisional["source"], ts,
            )
            self._conn.execute(
                "DELETE FROM memory_facts WHERE caller_id = ? AND key = ? AND status = ?",
                (caller_id, key, STATUS_PROVISIONAL),
            )
        return True

    def resolve_conflict(self, caller_id: str, key: str, *, accept_new: bool, ts: str) -> bool:
        """Resolve a flagged conflict. Accepting archives the old confirmed
        value to history (an explicit, caller-confirmed change); rejecting
        discards the provisional value."""
        provisional = self._get(caller_id, key, STATUS_PROVISIONAL)
        confirmed = self._get(caller_id, key, STATUS_CONFIRMED)
        if provisional is None or confirmed is None:
            return False
        with self._conn:
            if accept_new:
                self._conn.execute(
                    "INSERT INTO memory_history(caller_id, key, old_value, new_value,"
                    " ts, reason) VALUES (?, ?, ?, ?, ?, 'caller_confirmed_update')",
                    (caller_id, key, confirmed["value"], provisional["value"], ts),
                )
                self._upsert(
                    caller_id, key, provisional["value"], STATUS_CONFIRMED,
                    provisional["source"], ts,
                )
            self._conn.execute(
                "DELETE FROM memory_facts WHERE caller_id = ? AND key = ? AND status = ?",
                (caller_id, key, STATUS_PROVISIONAL),
            )
            self._conn.execute(
                "UPDATE memory_conflicts SET resolved = 1 WHERE caller_id = ?"
                " AND key = ? AND resolved = 0",
                (caller_id, key),
            )
        return True

    def discard_provisional(self, caller_id: str, key: str) -> bool:
        """Drop a provisional value (caller declined to confirm it)."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM memory_facts WHERE caller_id = ? AND key = ? AND status = ?",
                (caller_id, key, STATUS_PROVISIONAL),
            )
        return cur.rowcount > 0

    def get_facts(self, caller_id: str, *, include_provisional: bool = True) -> list[dict]:
        if include_provisional:
            rows = self._conn.execute(
                "SELECT * FROM memory_facts WHERE caller_id = ? ORDER BY key, status",
                (caller_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_facts WHERE caller_id = ? AND status = ?"
                " ORDER BY key",
                (caller_id, STATUS_CONFIRMED),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_confirmed(self, caller_id: str, key: str) -> str | None:
        row = self._get(caller_id, key, STATUS_CONFIRMED)
        return row["value"] if row else None

    def pending_conflicts(self, caller_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM memory_conflicts WHERE caller_id = ? AND resolved = 0"
            " ORDER BY id",
            (caller_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- deletion support ----------------------------------------------------

    _TABLES = ("memory_facts", "memory_conflicts", "memory_history", "consent")

    def delete_caller(self, caller_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._conn:
            for table in self._TABLES:
                cur = self._conn.execute(
                    f"DELETE FROM {table} WHERE caller_id = ?", (caller_id,)
                )
                counts[table] = cur.rowcount
        return counts

    def count_identifier_matches(self, *, caller_id: str, name: str, phone_digits: str) -> int:
        """Residual scan over every memory table: rows linked to the caller,
        or still carrying the caller's name/phone in any value column."""
        n = 0
        name_l = name.strip().lower()
        for table in self._TABLES:
            rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()
            for row in rows:
                values = [str(v) for v in dict(row).values()]
                if caller_id in values:
                    n += 1
                    continue
                joined = " ".join(values).lower()
                digits = "".join(c for c in joined if c.isdigit())
                if (name_l and name_l in joined) or (phone_digits and phone_digits in digits):
                    n += 1
        return n
