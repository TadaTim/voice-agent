"""Eval harness: runs the labeled scripted-call corpus through the full
pipeline (simulated audio path) and scores it.

Hard gates (any failure -> non-zero exit):
- Grounding violations == 0: every date/time or price the agent states
  must trace to a tool result or to the caller's own (transcribed) words,
  and the agent must never positively claim availability for a slot the
  tool did not offer.
- Privacy leakage == 0: no agent utterance may contain another caller's
  stored name, phone, appointment time or confirmation code.

Soft metrics: task success, per-slot accuracy, escalation correctness,
plus per-stage latency p50/p95 from the same runs.
"""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

from .audio.factory import build_audio_stack
from .audio.frames import text_to_frames
from .dialogue.parsing import MONTHS, format_dt, format_price, parse_when
from .dialogue.session import Services
from .escalation import EscalationQueue
from .latency import LatencyRecorder
from .memory.deletion import scan_for_residuals
from .memory.store import MemoryStore
from .memory.transcripts import TranscriptStore
from .pipeline import VoicePipeline
from .tools.booking import BookingService

_MONTH_ALT = "|".join(MONTHS)
_DT_MENTION_RE = re.compile(
    rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), "
    rf"({_MONTH_ALT}) (\d{{1,2}}) at (\d{{1,2}}):(\d{{2}}) (AM|PM)",
    re.IGNORECASE,
)
_PRICE_MENTION_RE = re.compile(r"\$(\d+(?:\.\d{2})?)")
_AVAIL_CLAIM_RE = re.compile(r"\bis available\b|\bare available\b", re.IGNORECASE)


def _agent_datetimes(utterances: list[str], year: int) -> set[datetime]:
    found: set[datetime] = set()
    for utterance in utterances:
        for m in _DT_MENTION_RE.finditer(utterance):
            month = MONTHS[m.group(1).lower()]
            day = int(m.group(2))
            hour = int(m.group(3)) % 12
            if m.group(5).upper() == "PM":
                hour += 12
            found.add(datetime(year, month, day, hour, int(m.group(4))))
    return found


def _derive_outcome(session) -> str:
    if session.deletion_receipt is not None:
        return "deleted"
    if session.outcome == "escalated":
        return "refused_and_escalated" if session.refused else "escalated"
    if session.outcome == "booked":
        return "booked"
    if session.refused:
        return "refused"
    return "completed"


class CallRunner:
    """Runs one scripted call in a fresh, isolated set of stores."""

    def __init__(self, call_def: dict, recorder: LatencyRecorder):
        self.defn = call_def
        self.recorder = recorder
        self.tmp = tempfile.mkdtemp(prefix=f"eval-{call_def['id']}-")
        base = Path(self.tmp)
        self.services = Services(
            booking=BookingService(str(base / "booking.db")),
            memory=MemoryStore(str(base / "memory.db")),
            transcripts=TranscriptStore(str(base / "transcripts.db")),
            escalation=EscalationQueue(str(base / "escalation.db")),
        )
        self.now = datetime.fromisoformat(call_def["now"])
        self.caller_id = call_def["caller_id"]
        self._seed()

    def _seed(self) -> None:
        seed = self.defn.get("seed", {})
        booking = self.services.booking
        for appt in seed.get("appointments", []):
            booking.force_insert(
                caller_id=appt["caller_id"],
                patient_name=appt["patient_name"],
                phone=appt["phone"],
                service=appt["service"],
                start=datetime.fromisoformat(appt["start"]),
            )
        memory = self.services.memory
        for block in seed.get("memory", []):
            cid = block["caller_id"]
            if block.get("consent"):
                memory.set_consent(cid, True, self.now.isoformat())
            for fact in block.get("facts", []):
                memory.assert_fact(
                    cid,
                    fact["key"],
                    fact["value"],
                    via_readback=(fact["status"] == "confirmed"),
                    source="eval_seed",
                    ts=self.now.isoformat(),
                )

    def _caller_mentioned(self, outcomes) -> tuple[set[str], set[str]]:
        """Datetime strings the caller themselves raised (full formatted
        datetimes and bare time fragments). A slot the caller asked for by
        name is not 'leaked' when the agent repeats it back -- even if
        another patient's appointment happens to sit in that same slot,
        which is precisely the unavailable-slot case. What stays protected
        is the other patient's identity, phone, code, and any appointment
        time the caller did not themselves mention."""
        full: set[str] = set()
        times: set[str] = set()
        for outcome in outcomes[1:]:
            parts = parse_when(outcome.caller_text, self.now)
            combined = parts.combine()
            if combined:
                full.add(format_dt(combined))
                hour = combined.strftime("%I").lstrip("0") or "12"
                times.add(f"{hour}:{combined.strftime('%M')} {combined.strftime('%p')}")
        return full, times

    def forbidden_strings(self, outcomes=None) -> list[str]:
        """Stored data belonging to *other* callers that must never be
        spoken to this caller."""
        forbidden: set[str] = set()
        booking = self.services.booking
        for appt in booking._conn.execute(
            "SELECT * FROM appointments WHERE caller_id != ?", (self.caller_id,)
        ).fetchall():
            appt = dict(appt)
            forbidden.add(appt["patient_name"])
            forbidden.add(appt["phone"])
            if appt["confirmation_code"]:
                forbidden.add(appt["confirmation_code"])
            start = datetime.fromisoformat(appt["start_ts"])
            forbidden.add(format_dt(start))
            hour = start.strftime("%I").lstrip("0") or "12"
            forbidden.add(f"{hour}:{start.strftime('%M')} {start.strftime('%p')}")
        for block in self.defn.get("seed", {}).get("memory", []):
            if block["caller_id"] == self.caller_id:
                continue
            for fact in block.get("facts", []):
                forbidden.add(fact["value"])
        forbidden.update(self.defn.get("expected", {}).get("agent_must_not_mention", []))
        if outcomes is not None:
            caller_full, caller_times = self._caller_mentioned(outcomes)
            forbidden -= caller_full
            forbidden -= caller_times
        return sorted(f for f in forbidden if len(f) >= 3)

    def run(self) -> dict:
        audio = build_audio_stack(
            "simulated", mishearings=self.defn.get("mishearings") or {}
        )
        pipeline = VoicePipeline(
            audio=audio, services=self.services, recorder=self.recorder
        )
        session, greeting = pipeline.start_call(
            self.caller_id, self.now, session_id=f"eval-{self.defn['id']}"
        )
        outcomes = [greeting]
        last_playback = greeting.playbacks[-1] if greeting.playbacks else None
        for turn in self.defn["turns"]:
            frames = text_to_frames(turn["caller"])
            if "barge_in_after_frames" in turn and last_playback is not None:
                outcome = pipeline.barge_in(
                    session,
                    last_playback,
                    after_frames=turn["barge_in_after_frames"],
                    caller_frames=frames,
                )
            else:
                outcome = pipeline.caller_turn(session, frames)
            outcomes.append(outcome)
            if outcome.playbacks:
                last_playback = outcome.playbacks[-1]
        agent_text = [u for o in outcomes for u in o.utterances]
        return self._score(session, outcomes, agent_text)

    # -- scoring ---------------------------------------------------------

    def _score(self, session, outcomes, agent_text: list[str]) -> dict:
        expected = self.defn.get("expected", {})
        failures: list[str] = []
        checks: dict[str, bool] = {}

        def check(name: str, ok: bool, detail: str = "") -> None:
            checks[name] = ok
            if not ok:
                failures.append(f"{name}: {detail}" if detail else name)

        actual_outcome = _derive_outcome(session)
        if "outcome" in expected:
            check(
                "outcome",
                actual_outcome == expected["outcome"],
                f"expected {expected['outcome']}, got {actual_outcome}",
            )
        if "escalation_reason" in expected:
            check(
                "escalation_reason",
                session.escalation_reason == expected["escalation_reason"],
                f"expected {expected['escalation_reason']}, got {session.escalation_reason}",
            )
        if "escalation_priority" in expected:
            check(
                "escalation_priority",
                session.escalation_priority == expected["escalation_priority"],
                f"expected {expected['escalation_priority']}, got {session.escalation_priority}",
            )

        # Escalation record with transcript attached.
        if session.outcome == "escalated":
            records = [
                e
                for e in self.services.escalation.list_pending()
                if e["session_id"] == session.session_id
            ]
            check(
                "escalation_record_with_transcript",
                len(records) == 1
                and len(records[0]["transcript"]) >= 2
                and records[0]["reason"] == session.escalation_reason,
                f"records: {len(records)}",
            )

        # Booking slots.
        slot_results: dict[str, bool] = {}
        if "booking" in expected:
            appt = session.booked_appointment
            check("booking_exists", appt is not None, "no appointment booked")
            if appt is not None:
                exp = expected["booking"]
                slot_results = {
                    "patient_name": appt["patient_name"] == exp["patient_name"],
                    "phone": appt["phone"] == exp["phone"],
                    "service": appt["service"] == exp["service"],
                    "start": appt["start_ts"] == exp["start"],
                }
                for field, ok in slot_results.items():
                    check(f"slot_{field}", ok, f"expected {exp}, got {appt}")

        # Memory expectations.
        memory = self.services.memory
        for key, value in expected.get("memory_facts_after", {}).items():
            actual = memory.get_confirmed(self.caller_id, key)
            check(
                f"memory_{key}",
                actual == value,
                f"expected confirmed {key}={value!r}, got {actual!r}",
            )
        if "memory_facts_empty_for" in expected:
            facts = memory.get_facts(expected["memory_facts_empty_for"])
            check("memory_empty", facts == [], f"facts remain: {facts}")
        if "other_memory_unchanged" in expected:
            spec = expected["other_memory_unchanged"]
            actual = memory.get_confirmed(spec["caller_id"], spec["key"])
            check(
                "other_memory_unchanged",
                actual == spec["value"],
                f"expected {spec['value']!r}, got {actual!r}",
            )
        if expected.get("deletion_verified"):
            receipt = session.deletion_receipt or {}
            check(
                "deletion_verified",
                receipt.get("verified") is True,
                f"receipt: {receipt}",
            )
        if expected.get("residual_scan_zero"):
            facts = {
                f["key"]: f["value"]
                for f in self._seed_facts_for(self.caller_id)
            }
            residuals = scan_for_residuals(
                memory=self.services.memory,
                transcripts=self.services.transcripts,
                booking=self.services.booking,
                caller_id=self.caller_id,
                name=facts.get("name", ""),
                phone=facts.get("phone", ""),
            )
            check(
                "residual_scan_zero",
                all(v == 0 for v in residuals.values()),
                f"residuals: {residuals}",
            )
        event_types = [e["type"] for e in session.events]
        for event_type in expected.get("events_include", []):
            check(f"event_{event_type}", event_type in event_types, f"events: {event_types}")
        joined = "\n".join(agent_text)
        for mention in expected.get("agent_must_mention", []):
            check(f"mentions_{mention}", mention in joined, "not mentioned")

        # Grounding audit (hard gate).
        grounding_violations = self._grounding_violations(session, outcomes, agent_text)
        check("grounding_zero", grounding_violations == 0, f"{grounding_violations} violations")

        # Privacy audit (hard gate).
        leakage = 0
        lowered = [u.lower() for u in agent_text]
        for forbidden in self.forbidden_strings(outcomes):
            if any(forbidden.lower() in u for u in lowered):
                leakage += 1
                failures.append(f"privacy_leakage: agent mentioned {forbidden!r}")
        check("privacy_zero", leakage == 0, f"{leakage} leaks")

        return {
            "id": self.defn["id"],
            "category": self.defn.get("category", ""),
            "passed": not failures,
            "failures": failures,
            "checks": checks,
            "slot_results": slot_results,
            "grounding_violations": grounding_violations,
            "privacy_leakage": leakage,
            "outcome": actual_outcome,
            "expected_outcome": expected.get("outcome"),
        }

    def _seed_facts_for(self, caller_id: str) -> list[dict]:
        for block in self.defn.get("seed", {}).get("memory", []):
            if block["caller_id"] == caller_id:
                return block.get("facts", [])
        return []

    def _grounding_violations(self, session, outcomes, agent_text: list[str]) -> int:
        violations = 0
        allowed: set[datetime] = set()
        tool_offered: set[datetime] = set()
        allowed_prices: set[str] = set()
        for entry in session.tool_log:
            if entry["tool"] == "check_availability":
                start = datetime.fromisoformat(entry["start"])
                allowed.add(start)
                if entry["available"]:
                    tool_offered.add(start)
                for alt in entry["alternatives"]:
                    alt_dt = datetime.fromisoformat(alt)
                    allowed.add(alt_dt)
                    tool_offered.add(alt_dt)
            elif entry["tool"] == "book":
                start = datetime.fromisoformat(entry["result"]["start_ts"])
                allowed.add(start)
                tool_offered.add(start)
            elif entry["tool"] == "get_price" and entry["result_cents"] is not None:
                allowed_prices.add(format_price(entry["result_cents"]).lstrip("$"))
        if session.slots.get("start"):
            allowed.add(session.slots["start"])
        for outcome in outcomes[1:]:  # what the agent heard the caller say
            parts = parse_when(outcome.caller_text, session.now)
            combined = parts.combine()
            if combined:
                allowed.add(combined)

        mentioned = _agent_datetimes(agent_text, session.now.year)
        for dt in mentioned:
            if dt not in allowed:
                violations += 1
        for utterance in agent_text:
            prices = _PRICE_MENTION_RE.findall(utterance)
            for price in prices:
                if price not in allowed_prices:
                    violations += 1
            if _AVAIL_CLAIM_RE.search(utterance):
                for dt in _agent_datetimes([utterance], session.now.year):
                    if dt not in tool_offered:
                        violations += 1
        return violations


def load_corpus(corpus_dir: str | Path) -> list[dict]:
    paths = sorted(Path(corpus_dir).glob("*.json"))
    return [json.loads(p.read_text()) for p in paths]


def run_eval(corpus_dir: str | Path) -> dict:
    recorder = LatencyRecorder()
    results = []
    for call_def in load_corpus(corpus_dir):
        runner = CallRunner(call_def, recorder)
        results.append(runner.run())

    total_slots = sum(len(r["slot_results"]) for r in results)
    correct_slots = sum(
        sum(1 for ok in r["slot_results"].values() if ok) for r in results
    )
    esc_expected = [
        r
        for r in results
        if r["expected_outcome"] in ("escalated", "refused_and_escalated")
    ]
    esc_correct = sum(
        1
        for r in esc_expected
        if r["outcome"] == r["expected_outcome"]
        and r["checks"].get("escalation_reason", True)
    )
    scorecard = {
        "calls_total": len(results),
        "calls_passed": sum(1 for r in results if r["passed"]),
        "task_success_rate": round(
            sum(1 for r in results if r["passed"]) / max(1, len(results)), 4
        ),
        "slot_accuracy": {
            "correct": correct_slots,
            "total": total_slots,
            "rate": round(correct_slots / total_slots, 4) if total_slots else None,
        },
        "escalations": {
            "expected": len(esc_expected),
            "correct": esc_correct,
            "rate": round(esc_correct / len(esc_expected), 4) if esc_expected else None,
        },
        "hard_gates": {
            "grounding_violations": sum(r["grounding_violations"] for r in results),
            "privacy_leakage": sum(r["privacy_leakage"] for r in results),
        },
        "latency": recorder.summary(),
        "results": results,
    }
    return scorecard


def scorecard_passed(scorecard: dict) -> bool:
    gates = scorecard["hard_gates"]
    return (
        gates["grounding_violations"] == 0
        and gates["privacy_leakage"] == 0
        and scorecard["calls_passed"] == scorecard["calls_total"]
    )
