"""Local booking service backed by SQLite.

This is the agent's only tool for appointments. The dialogue layer may
state availability, prices and confirmation details *only* from results
returned here; if a question cannot be answered from this service, the
dialogue layer must escalate rather than invent an answer.

Deterministic by design: fixed clinic hours, a fixed 30-minute slot grid,
and a fixed alternative-search order, so eval results are reproducible.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta

OPEN_TIME = time(8, 0)
CLOSE_TIME = time(17, 0)
GRID_MINUTES = 30
ALTERNATIVE_HORIZON_DAYS = 5

DEFAULT_SERVICES: dict[str, dict] = {
    "Checkup": {"duration_min": 30, "price_cents": 8000},
    "Physical": {"duration_min": 45, "price_cents": 15000},
    "Vaccination": {"duration_min": 15, "price_cents": 4000},
    "Blood Work": {"duration_min": 20, "price_cents": 6000},
    "Follow-up": {"duration_min": 30, "price_cents": 7000},
}

# Caller-facing synonyms -> canonical service names.
SERVICE_SYNONYMS: dict[str, str] = {
    "checkup": "Checkup",
    "check-up": "Checkup",
    "check up": "Checkup",
    "physical": "Physical",
    "annual physical": "Physical",
    "vaccination": "Vaccination",
    "vaccine": "Vaccination",
    "flu shot": "Vaccination",
    "shot": "Vaccination",
    "blood work": "Blood Work",
    "bloodwork": "Blood Work",
    "blood test": "Blood Work",
    "labs": "Blood Work",
    "follow-up": "Follow-up",
    "follow up": "Follow-up",
    "followup": "Follow-up",
}


class SlotUnavailable(Exception):
    """Raised when a booking is attempted on a slot that is not bookable."""


def canonical_service(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    if key in SERVICE_SYNONYMS:
        return SERVICE_SYNONYMS[key]
    for canonical in DEFAULT_SERVICES:
        if canonical.lower() == key:
            return canonical
    return None


@dataclass
class AvailabilityResult:
    available: bool
    reason: str | None
    alternatives: list[datetime]


class BookingService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS services(
                       name TEXT PRIMARY KEY,
                       duration_min INTEGER NOT NULL,
                       price_cents INTEGER NOT NULL)"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS appointments(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       confirmation_code TEXT UNIQUE,
                       caller_id TEXT NOT NULL,
                       patient_name TEXT NOT NULL,
                       phone TEXT NOT NULL,
                       service TEXT NOT NULL,
                       start_ts TEXT NOT NULL,
                       end_ts TEXT NOT NULL,
                       status TEXT NOT NULL DEFAULT 'booked')"""
            )
            for name, info in DEFAULT_SERVICES.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO services(name, duration_min, price_cents)"
                    " VALUES (?, ?, ?)",
                    (name, info["duration_min"], info["price_cents"]),
                )

    def close(self) -> None:
        self._conn.close()

    # -- service facts -----------------------------------------------------

    def service_info(self, service: str) -> dict | None:
        canonical = canonical_service(service)
        if canonical is None:
            return None
        row = self._conn.execute(
            "SELECT name, duration_min, price_cents FROM services WHERE name = ?",
            (canonical,),
        ).fetchone()
        return dict(row) if row else None

    def get_price_cents(self, service: str) -> int | None:
        info = self.service_info(service)
        return info["price_cents"] if info else None

    def service_names(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM services ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    # -- availability ------------------------------------------------------

    def _duration(self, service: str) -> timedelta:
        info = self.service_info(service)
        if info is None:
            raise ValueError(f"unknown service: {service}")
        return timedelta(minutes=info["duration_min"])

    def _is_bookable(self, start: datetime, service: str) -> tuple[bool, str | None]:
        if start.weekday() >= 5:
            return False, "clinic_closed_weekend"
        end = start + self._duration(service)
        if start.time() < OPEN_TIME or end.time() > CLOSE_TIME or end.date() != start.date():
            return False, "outside_clinic_hours"
        if start.minute % GRID_MINUTES != 0 or start.second or start.microsecond:
            return False, "off_grid"
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM appointments WHERE status = 'booked'"
            " AND start_ts < ? AND end_ts > ?",
            (end.isoformat(), start.isoformat()),
        ).fetchone()
        if row["n"] > 0:
            return False, "slot_taken"
        return True, None

    def check_availability(self, start: datetime, service: str) -> AvailabilityResult:
        """Check a slot; if unbookable, deterministically find alternatives."""
        ok, reason = self._is_bookable(start, service)
        if ok:
            return AvailabilityResult(available=True, reason=None, alternatives=[])
        alternatives: list[datetime] = []
        cursor = self._next_grid_point(start)
        horizon = start + timedelta(days=ALTERNATIVE_HORIZON_DAYS)
        while cursor <= horizon and len(alternatives) < 3:
            bookable, _ = self._is_bookable(cursor, service)
            if bookable:
                alternatives.append(cursor)
            cursor = self._next_grid_point(cursor)
        return AvailabilityResult(
            available=False, reason=reason, alternatives=alternatives
        )

    @staticmethod
    def _next_grid_point(dt: datetime) -> datetime:
        base = dt.replace(second=0, microsecond=0)
        remainder = base.minute % GRID_MINUTES
        if remainder == 0 and base == dt.replace(second=0, microsecond=0) and dt.second == 0:
            return base + timedelta(minutes=GRID_MINUTES)
        return base + timedelta(minutes=GRID_MINUTES - remainder)

    # -- booking -----------------------------------------------------------

    def book(
        self,
        *,
        caller_id: str,
        patient_name: str,
        phone: str,
        service: str,
        start: datetime,
    ) -> dict:
        """Book a slot. Re-validates availability at write time."""
        canonical = canonical_service(service)
        if canonical is None:
            raise ValueError(f"unknown service: {service}")
        ok, reason = self._is_bookable(start, canonical)
        if not ok:
            raise SlotUnavailable(f"{start.isoformat()} not bookable: {reason}")
        end = start + self._duration(canonical)
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO appointments(caller_id, patient_name, phone, service,"
                " start_ts, end_ts) VALUES (?, ?, ?, ?, ?, ?)",
                (caller_id, patient_name, phone, canonical, start.isoformat(), end.isoformat()),
            )
            appt_id = cur.lastrowid
            digest = hashlib.sha256(
                f"{appt_id}|{caller_id}|{start.isoformat()}".encode()
            ).hexdigest()[:4].upper()
            code = f"VA-{appt_id:04d}-{digest}"
            self._conn.execute(
                "UPDATE appointments SET confirmation_code = ? WHERE id = ?",
                (code, appt_id),
            )
        return self._get_appointment(appt_id)

    def force_insert(
        self,
        *,
        caller_id: str,
        patient_name: str,
        phone: str,
        service: str,
        start: datetime,
    ) -> dict:
        """Insert without availability checks. Fixture/eval seeding only --
        the agent itself never calls this."""
        canonical = canonical_service(service) or service
        end = start + timedelta(
            minutes=DEFAULT_SERVICES.get(canonical, {"duration_min": 30})["duration_min"]
        )
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO appointments(caller_id, patient_name, phone, service,"
                " start_ts, end_ts) VALUES (?, ?, ?, ?, ?, ?)",
                (caller_id, patient_name, phone, canonical, start.isoformat(), end.isoformat()),
            )
            appt_id = cur.lastrowid
            code = f"VA-{appt_id:04d}-SEED"
            self._conn.execute(
                "UPDATE appointments SET confirmation_code = ? WHERE id = ?",
                (code, appt_id),
            )
        return self._get_appointment(appt_id)

    def _get_appointment(self, appt_id: int) -> dict:
        row = self._conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appt_id,)
        ).fetchone()
        return dict(row)

    def appointments_for_caller(self, caller_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM appointments WHERE caller_id = ? AND status = 'booked'"
            " ORDER BY start_ts",
            (caller_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- deletion support ----------------------------------------------------

    def anonymize_caller(self, caller_id: str) -> int:
        """Strip identifiers from a caller's appointment records.

        The slot itself is kept (the clinic still needs to know a patient is
        coming) but name, phone and caller link are replaced with a shared
        tombstone, so no residual identifier remains."""
        with self._conn:
            cur = self._conn.execute(
                "UPDATE appointments SET patient_name = 'REDACTED',"
                " phone = 'REDACTED', caller_id = 'deleted-caller'"
                " WHERE caller_id = ?",
                (caller_id,),
            )
        return cur.rowcount

    def count_identifier_matches(self, *, caller_id: str, name: str, phone_digits: str) -> int:
        """Count appointment rows still carrying any of the identifiers."""
        n = 0
        for row in self._conn.execute(
            "SELECT caller_id, patient_name, phone FROM appointments"
        ).fetchall():
            if row["caller_id"] == caller_id:
                n += 1
            elif name and row["patient_name"].strip().lower() == name.strip().lower():
                n += 1
            elif phone_digits and "".join(c for c in row["phone"] if c.isdigit()) == phone_digits:
                n += 1
        return n
