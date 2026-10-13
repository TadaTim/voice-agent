"""Deterministic dialogue manager for clinic appointment booking.

Grounding discipline (enforced structurally, audited by the eval harness):

- Availability, prices, service lists and confirmation details are stated
  *only* from booking-tool results, which are logged per session.
- A question the tools cannot answer escalates to a human. The agent never
  invents an answer.

Memory discipline:

- Name/phone are confirmed through a mandatory read-back before booking.
- Facts reach the governed memory store only under its rules (consent,
  provisional-until-confirmed, conflicts flagged -- see memory/store.py).
"""

from __future__ import annotations

import re
from datetime import datetime

from ..memory.deletion import delete_caller_data
from ..tools.booking import SlotUnavailable
from .guardrails import GuardrailHit, check_caller_text
from .parsing import (
    extract_name,
    extract_phone,
    extract_phone_last4,
    extract_service,
    format_dt,
    format_price,
    parse_when,
)
from .session import CLINIC_NAME, ConfirmItem, DialogueSession, Services

_AFFIRM = (
    "yes", "yeah", "yep", "correct", "that's right", "that is right",
    "sounds good", "looks good", "confirm", "sure", "okay", "ok",
    "please do", "go ahead", "perfect", "that's fine", "that is fine",
)
_NEGATE = (
    "no", "nope", "wrong", "actually", "incorrect", "not right",
    "isn't right", "is not right", "don't", "do not",
)
_DELETE_CUES = (
    "delete my", "forget me", "forget everything", "erase my",
    "remove my information", "remove my data", "delete everything about me",
)
_INSPECT_CUES = (
    "what do you remember about me", "what do you know about me",
    "what information do you have about me", "what do you have on file for me",
    "what have you saved about me", "show me my information",
)
_BOOKING_CUES = (
    "book", "appointment", "schedule", "see a doctor", "see someone",
    "make an appointment", "need to see", "want to see",
)
_PRICE_CUES = ("how much", "price", "cost", "charge for", "what do you charge")
_PREF_RE = re.compile(
    r"\bi\s+(?:prefer|would prefer|'d prefer|like)\s+"
    r"(morning|afternoon|evening)\s+appointments?\b"
)
_EMAIL_RE = re.compile(r"my email is\s+([\w.+-]+@[\w-]+\.[\w.]+)")
_OTHER_ASSERT_RE = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s|\s+his|\s+her)\s+"
    r"(?:phone|number|email|address)\s+is\s+",
)
_PRICE_PHRASE_RE = re.compile(
    r"(?:how much is|how much does|price of|cost of|what is the price of|"
    r"what's the price of)\s+(?:a\s+|an\s+|the\s+)?"
    r"([a-z][a-z ]{1,30}?)(?:\s+session|\s+appointment|\?|$)"
)


def _is_affirmation(lower: str) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", lower) for p in _AFFIRM)


def _is_negation(lower: str) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", lower) for p in _NEGATE)


class DialogueManager:
    def __init__(self, services: Services):
        self.svc = services

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------

    def open_call(self, session: DialogueSession) -> list[str]:
        greeting = f"Hello. Thank you for calling {CLINIC_NAME}."
        name = self._known_name(session)
        if name:
            greeting = f"Hello, {name.split()[0]}. Thank you for calling {CLINIC_NAME}."
        return [greeting + " How can I help you today?"]

    def handle(self, session: DialogueSession, text: str) -> list[str]:
        session.turns += 1
        lower = text.lower()

        hit = check_caller_text(text, caller_name=self._known_name(session))
        if hit is not None:
            return self._apply_guardrail(session, hit)

        if any(cue in lower for cue in _DELETE_CUES):
            return self._do_delete(session)
        if any(cue in lower for cue in _INSPECT_CUES):
            return self._do_inspect(session)

        price_response = self._maybe_price_answer(session, text)
        if price_response is not None:
            return price_response

        if self._other_person_assertion(session, text):
            session.refused = True
            session.events.append({"type": "refusal", "kind": "third_party_fact"})
            return [
                "I can only save information about you, the caller. "
                "I can't store details about other people."
            ]

        self._capture_assertions(session, text)

        handler = getattr(self, f"_state_{session.state}", self._state_done)
        return handler(session, text)

    # ------------------------------------------------------------------
    # guardrails / escalation / refusal
    # ------------------------------------------------------------------

    def _apply_guardrail(self, session: DialogueSession, hit: GuardrailHit) -> list[str]:
        if hit.kind == "distress":
            utterance = (
                "I'm really glad you told me. I want to make sure you get the "
                "right help right away, so I'm connecting you to a member of "
                "our care team now. If this is an emergency, please hang up "
                "and call 911."
            )
            return [utterance] + self._escalate(session, "distress", "high")
        if hit.kind == "medical_advice":
            utterance = (
                "That's a medical question I can't answer safely. I'm "
                "connecting you to a member of our care team who can help."
            )
            return [utterance] + self._escalate(session, "medical_advice", "normal")
        if hit.kind == "social_engineering":
            session.refused = True
            session.events.append({"type": "refusal", "kind": "social_engineering"})
            utterance = (
                "I can't provide that information. I'm also flagging this "
                "call for our team to review."
            )
            return [utterance] + self._escalate(session, "social_engineering", "high")
        # privacy_other_info
        session.refused = True
        session.events.append({"type": "refusal", "kind": "privacy_other_info"})
        return [
            "I can't share information about other patients. I'm only able "
            "to access your own information. Is there something I can help "
            "you with for yourself?"
        ]

    def _escalate(self, session: DialogueSession, reason: str, priority: str) -> list[str]:
        transcript = self.svc.transcripts.for_session(session.session_id)
        esc_id = self.svc.escalation.enqueue(
            ts=session.now.isoformat(),
            caller_id=session.caller_id,
            session_id=session.session_id,
            reason=reason,
            priority=priority,
            transcript=transcript,
        )
        session.outcome = "escalated"
        session.escalation_reason = reason
        session.escalation_priority = priority
        session.state = "done"
        session.events.append(
            {"type": "escalation", "reason": reason, "priority": priority, "id": esc_id}
        )
        return []

    def _escalate_with_notice(
        self, session: DialogueSession, reason: str, notice: str
    ) -> list[str]:
        return [notice] + self._escalate(session, reason, "normal")

    # ------------------------------------------------------------------
    # memory commands
    # ------------------------------------------------------------------

    def _known_name(self, session: DialogueSession) -> str | None:
        if session.slots.get("name"):
            return session.slots["name"]
        if self.svc.memory.get_consent(session.caller_id):
            return self.svc.memory.get_confirmed(session.caller_id, "name")
        return None

    def _describe_fact(self, key: str, value: str) -> str:
        if key == "preferred_time":
            return f"you prefer {value} appointments"
        if key == "email":
            return f"your email is {value}"
        if key == "phone":
            return f"your phone number is {value}"
        if key == "name":
            return f"your name is {value}"
        return f"{key} is {value}"

    def _do_inspect(self, session: DialogueSession) -> list[str]:
        session.events.append({"type": "memory_inspection"})
        facts = []
        if self.svc.memory.get_consent(session.caller_id):
            facts = self.svc.memory.get_facts(session.caller_id)
        if not facts:
            answer = "I don't have any information saved about you."
        else:
            parts = []
            for fact in facts:
                desc = self._describe_fact(fact["key"], fact["value"])
                if fact["status"] == "provisional":
                    desc += " (not yet confirmed)"
                parts.append(desc)
            answer = "Here's what I have saved about you: " + "; ".join(parts) + "."
        utterances = [answer]
        prompt = self._resume_prompt(session)
        if prompt and session.state != "done":
            utterances.append(prompt)
        return utterances

    def _do_delete(self, session: DialogueSession) -> list[str]:
        mem = self.svc.memory
        name = mem.get_confirmed(session.caller_id, "name") or session.slots.get("name", "")
        phone = mem.get_confirmed(session.caller_id, "phone") or session.slots.get("phone", "")
        receipt = delete_caller_data(
            memory=mem,
            transcripts=self.svc.transcripts,
            booking=self.svc.booking,
            caller_id=session.caller_id,
            name=name,
            phone=phone,
            deleted_at=session.now.isoformat(),
        )
        session.deletion_receipt = receipt.to_dict()
        session.pending_facts.clear()
        session.provisional_keys.clear()
        session.confirm_queue.clear()
        session.outcome = "deleted"
        session.state = "done"
        session.events.append(
            {"type": "deletion", "receipt_id": receipt.receipt_id, "verified": receipt.verified}
        )
        verified_text = (
            "A verification scan of all of our stores found no remaining copies."
            if receipt.verified
            else "A verification scan did not come back clean; our team has been notified."
        )
        return [
            "I've deleted your saved information and call transcripts, and "
            "anonymized your appointment records. "
            f"Your deletion receipt is {receipt.receipt_id}. {verified_text} "
            "Thank you for calling."
        ]

    # ------------------------------------------------------------------
    # price questions (tool-grounded)
    # ------------------------------------------------------------------

    def _maybe_price_answer(self, session: DialogueSession, text: str) -> list[str] | None:
        lower = text.lower()
        if not any(cue in lower for cue in _PRICE_CUES):
            return None
        service = extract_service(text) or session.slots.get("service")
        if service:
            price = self.svc.booking.get_price_cents(service)
            session.tool_log.append(
                {"tool": "get_price", "service": service, "result_cents": price}
            )
            if price is None:
                return self._escalate_with_notice(
                    session,
                    "unanswerable_question",
                    f"I don't have pricing information for {service}, so I'm "
                    "connecting you to our front desk team who can help.",
                )
            utterances = [f"A {service} costs {format_price(price)}."]
            prompt = self._resume_prompt(session)
            if prompt and session.state != "done":
                utterances.append(prompt)
            return utterances
        m = _PRICE_PHRASE_RE.search(lower)
        if m:
            phrase = m.group(1).strip()
            if phrase and "my" not in phrase and "appointment" not in phrase:
                return self._escalate_with_notice(
                    session,
                    "unanswerable_question",
                    f"I don't have information about {phrase} in our system, "
                    "so I'm connecting you to our front desk team who can help.",
                )
        return ["Which service would you like the price for?"]

    # ------------------------------------------------------------------
    # memory assertion capture (poisoning-aware)
    # ------------------------------------------------------------------

    def _other_person_assertion(self, session: DialogueSession, text: str) -> bool:
        m = _OTHER_ASSERT_RE.search(text)
        if not m:
            return False
        known = (self._known_name(session) or "").lower()
        return m.group(1).strip().lower() != known

    def _register_pending(self, session: DialogueSession, key: str, value: str, route: str) -> None:
        for fact in session.pending_facts:
            if fact["key"] == key:
                fact["value"] = value
                fact["route"] = route
                return
        session.pending_facts.append({"key": key, "value": value, "route": route})

    def _capture_assertions(self, session: DialogueSession, text: str) -> None:
        """Pick up caller-asserted facts (preference / email / phone) and
        route them through the governed store (or the pending list when
        there is no consent yet)."""
        found: list[tuple[str, str]] = []
        m = _PREF_RE.search(text.lower())
        if m:
            found.append(("preferred_time", m.group(1)))
        m = _EMAIL_RE.search(text.lower())
        if m:
            found.append(("email", m.group(1)))
        lower = text.lower()
        if "my number is" in lower or "my phone is" in lower or "my phone number is" in lower:
            phone = extract_phone(text)
            if phone:
                found.append(("phone", phone))
        for key, value in found:
            if self.svc.memory.get_consent(session.caller_id):
                result = self.svc.memory.assert_fact(
                    session.caller_id, key, value, ts=session.now.isoformat()
                )
                if result.kind == "stored_provisional":
                    session.provisional_keys.append(key)
                elif result.kind == "conflict_flagged":
                    session.events.append(
                        {"type": "memory_conflict", "key": key}
                    )
            else:
                self._register_pending(session, key, value, "confirm_step")

    # ------------------------------------------------------------------
    # booking flow
    # ------------------------------------------------------------------

    def _resume_prompt(self, session: DialogueSession) -> str | None:
        state = session.state
        if state == "collect_name":
            return "May I have your full name, please?"
        if state == "collect_phone":
            return "What's the best phone number for this appointment?"
        if state == "collect_service":
            return self._service_prompt(session)
        if state == "collect_when":
            return "What day and time would you like?"
        if state == "pick_alternative":
            return self._alternatives_utterance(session)
        if state == "readback":
            return self._readback_utterance(session)
        if state == "greeting":
            return "How can I help you today?"
        if state == "consent":
            return (
                "Would you like me to remember your details for next time? "
                "I won't save anything without your permission."
            )
        if state == "confirm_items" and session.confirm_queue:
            return self._confirm_question(session.confirm_queue[0])
        return None

    def _service_prompt(self, session: DialogueSession) -> str:
        names = self.svc.booking.service_names()
        session.tool_log.append({"tool": "service_names", "result": names})
        return "What type of appointment would you like? We offer " + ", ".join(names) + "."

    def _begin_booking(self, session: DialogueSession, text: str) -> list[str]:
        utterances: list[str] = []
        mem = self.svc.memory
        if mem.get_consent(session.caller_id):
            name = mem.get_confirmed(session.caller_id, "name")
            phone = mem.get_confirmed(session.caller_id, "phone")
            if name:
                session.slots["name"] = name
            if phone:
                session.slots["phone"] = phone
            if name and phone:
                utterances.append(
                    "I'll use the name and phone number I have on file; "
                    "I'll read everything back for you to confirm before booking."
                )
        service = extract_service(text)
        if service:
            session.slots["service"] = service
        prompt = self._next_slot_prompt(session)
        if prompt:
            utterances.append(prompt)
        return utterances

    def _next_slot_prompt(self, session: DialogueSession) -> str | None:
        if not session.slots.get("name"):
            session.state = "collect_name"
            return "May I have your full name, please?"
        if not session.slots.get("phone"):
            session.state = "collect_phone"
            return "What's the best phone number for this appointment?"
        if not session.slots.get("service"):
            session.state = "collect_service"
            return self._service_prompt(session)
        session.state = "collect_when"
        return "What day and time would you like?"

    def _state_greeting(self, session: DialogueSession, text: str) -> list[str]:
        lower = text.lower()
        if "my appointment" in lower and ("when" in lower or "what time" in lower):
            appts = self.svc.booking.appointments_for_caller(session.caller_id)
            session.tool_log.append(
                {"tool": "appointments_for_caller", "result_count": len(appts)}
            )
            if appts:
                appt = appts[0]
                start = datetime.fromisoformat(appt["start_ts"])
                session.events.append({"type": "info", "kind": "own_appointment"})
                return [
                    f"Your next appointment is a {appt['service']} on "
                    f"{format_dt(start)}. Your confirmation code is "
                    f"{appt['confirmation_code']}."
                ]
            return ["I don't see an upcoming appointment booked for you."]
        if any(cue in lower for cue in _BOOKING_CUES) or extract_service(text):
            return self._begin_booking(session, text)
        return [
            "I can help you book an appointment, check prices, or tell you "
            "what information I have saved about you. What would you like to do?"
        ]

    def _state_collect_name(self, session: DialogueSession, text: str) -> list[str]:
        name = extract_name(text)
        if not name:
            return ["I'm sorry, I didn't catch that. May I have your full name, please?"]
        session.slots["name"] = name
        self._register_pending(session, "name", name, "readback")
        return [self._next_slot_prompt(session)]

    def _state_collect_phone(self, session: DialogueSession, text: str) -> list[str]:
        phone = extract_phone(text)
        if not phone:
            return [
                "I need a 10-digit phone number, like 555-010-1234. "
                "What's the best number for this appointment?"
            ]
        session.slots["phone"] = phone
        self._register_pending(session, "phone", phone, "readback")
        return [self._next_slot_prompt(session)]

    def _state_collect_service(self, session: DialogueSession, text: str) -> list[str]:
        service = extract_service(text)
        if service:
            session.slots["service"] = service
            session.state = "collect_when"
            return ["What day and time would you like?"]
        session.service_attempts += 1
        if session.service_attempts >= 2:
            return self._escalate_with_notice(
                session,
                "unknown_service",
                "I'm sorry, that's not a service I can book. I'm connecting "
                "you to our front desk team who can help.",
            )
        return [
            "I'm sorry, I don't see that service. " + self._service_prompt(session)
        ]

    def _state_collect_when(self, session: DialogueSession, text: str) -> list[str]:
        parts = parse_when(
            text,
            session.now,
            default_from=session.slots.get("start"),
        )
        if parts.day is not None:
            session.when_day = parts.day
        if parts.tod is not None:
            session.when_tod = parts.tod
        if session.when_day is not None and session.when_tod is not None:
            start = datetime.combine(session.when_day, session.when_tod)
            return self._check_and_continue(session, start)
        if session.when_day is not None:
            return [f"What time on {session.when_day.strftime('%A, %B')} {session.when_day.day}?"]
        if session.when_tod is not None:
            return ["And which day would you like?"]
        return [
            "I didn't catch the day and time. You can say something like "
            "'October 20 at 10 AM'."
        ]

    def _check_and_continue(self, session: DialogueSession, start: datetime) -> list[str]:
        service = session.slots["service"]
        result = self.svc.booking.check_availability(start, service)
        session.tool_log.append(
            {
                "tool": "check_availability",
                "service": service,
                "start": start.isoformat(),
                "available": result.available,
                "alternatives": [a.isoformat() for a in result.alternatives],
            }
        )
        if result.available:
            session.slots["start"] = start
            session.state = "readback"
            return [self._readback_utterance(session)]
        if not result.alternatives:
            return self._escalate_with_notice(
                session,
                "no_availability",
                f"I'm sorry, {format_dt(start)} isn't available, and I don't "
                "see other open times in the next few days. I'm connecting "
                "you to our front desk team who can help.",
            )
        session.offered_alternatives = list(result.alternatives)
        session.state = "pick_alternative"
        return [
            f"I'm sorry, {format_dt(start)} isn't available for a {service}.",
            self._alternatives_utterance(session),
        ]

    def _alternatives_utterance(self, session: DialogueSession) -> str:
        alts = session.offered_alternatives
        listing = "; ".join(
            f"{i + 1}) {format_dt(a)}" for i, a in enumerate(alts)
        )
        return f"The next open times are: {listing}. Which works for you?"

    def _state_pick_alternative(self, session: DialogueSession, text: str) -> list[str]:
        lower = text.lower()
        if "none" in lower or "neither" in lower or "no thank" in lower:
            return self._escalate_with_notice(
                session,
                "no_availability",
                "I'm sorry those times don't work. I'm connecting you to "
                "our front desk team who can look for other options.",
            )
        m = re.search(r"\b(first|second|third)\b", lower)
        if m:
            idx = {"first": 0, "second": 1, "third": 2}[m.group(1)]
            if idx < len(session.offered_alternatives):
                start = session.offered_alternatives[idx]
                session.slots["start"] = start
                session.state = "readback"
                return [self._readback_utterance(session)]
        parts = parse_when(text, session.now, default_from=session.slots.get("start"))
        if parts.day is not None and parts.tod is not None:
            start = datetime.combine(parts.day, parts.tod)
            if start in session.offered_alternatives:
                session.slots["start"] = start
                session.state = "readback"
                return [self._readback_utterance(session)]
            return self._check_and_continue(session, start)
        if parts.tod is not None:
            for alt in session.offered_alternatives:
                if alt.time() == parts.tod:
                    session.slots["start"] = alt
                    session.state = "readback"
                    return [self._readback_utterance(session)]
        return ["I didn't catch that. " + self._alternatives_utterance(session)]

    # -- read-back & booking completion --------------------------------

    def _readback_utterance(self, session: DialogueSession) -> str:
        s = session.slots
        return (
            "Let me read that back to make sure I have everything right. "
            f"An appointment for {s['name']}, phone number {s['phone']}, "
            f"for a {s['service']} on {format_dt(s['start'])}. Is that correct?"
        )

    def _extract_slot_updates(self, session: DialogueSession, text: str) -> dict:
        updates: dict = {}
        slots = session.slots
        phone = extract_phone(text)
        if phone and phone != slots.get("phone"):
            updates["phone"] = phone
        else:
            last4 = extract_phone_last4(text)
            current = slots.get("phone", "")
            if last4 and current and not current.endswith(last4):
                updates["phone"] = current.rsplit("-", 1)[0] + "-" + last4
        service = extract_service(text)
        if service and service != slots.get("service"):
            updates["service"] = service
        parts = parse_when(text, session.now, default_from=slots.get("start"))
        if slots.get("start") is not None:
            new_start = slots["start"]
            changed = False
            if parts.day is not None and parts.day != slots["start"].date():
                new_start = new_start.replace(
                    year=parts.day.year, month=parts.day.month, day=parts.day.day
                )
                changed = True
            if parts.tod is not None and parts.tod != slots["start"].time():
                new_start = new_start.replace(
                    hour=parts.tod.hour, minute=parts.tod.minute, second=0
                )
                changed = True
            if changed:
                updates["start"] = new_start
        cleaned = re.sub(
            r"^(no|nope|actually|well|so)[,.\s]+", "", text.strip(), flags=re.IGNORECASE
        )
        name = extract_name(cleaned)
        if name and name != slots.get("name") and (
            re.search(r"(my name is|this is|name is)", text, re.IGNORECASE)
            or re.fullmatch(r"[A-Za-z]+(\s+[A-Za-z]+){0,2}", cleaned.strip())
        ):
            updates["name"] = name
        return updates

    def _state_readback(self, session: DialogueSession, text: str) -> list[str]:
        lower = text.lower()
        updates = self._extract_slot_updates(session, text)
        correction_cues = any(
            cue in lower
            for cue in (
                "actually", "i said", "i meant", "should be", "change",
                "wrong", "isn't", "not right", "incorrect", "ends in",
                "ends with",
            )
        )
        if updates and (correction_cues or _is_negation(lower) or not _is_affirmation(lower)):
            for key, value in updates.items():
                session.slots[key] = value
                if key in ("name", "phone"):
                    self._register_pending(session, key, value, "readback")
            session.events.append({"type": "slot_correction", "slots": sorted(updates)})
            start = session.slots["start"]
            if "start" in updates or "service" in updates:
                result = self.svc.booking.check_availability(
                    start, session.slots["service"]
                )
                session.tool_log.append(
                    {
                        "tool": "check_availability",
                        "service": session.slots["service"],
                        "start": start.isoformat(),
                        "available": result.available,
                        "alternatives": [a.isoformat() for a in result.alternatives],
                    }
                )
                if not result.available:
                    session.offered_alternatives = list(result.alternatives)
                    session.state = "pick_alternative"
                    return [
                        "Thank you, I've updated that. However, the new time "
                        "isn't available.",
                        self._alternatives_utterance(session),
                    ]
            return ["Thank you, I've updated that.", self._readback_utterance(session)]
        if _is_affirmation(lower) and not _is_negation(lower):
            return self._complete_booking(session)
        if updates:
            return ["Thank you, I've updated that.", self._readback_utterance(session)]
        return [
            "I'm sorry, I didn't catch that. " + self._readback_utterance(session)
        ]

    def _complete_booking(self, session: DialogueSession) -> list[str]:
        s = session.slots
        try:
            appt = self.svc.booking.book(
                caller_id=session.caller_id,
                patient_name=s["name"],
                phone=s["phone"],
                service=s["service"],
                start=s["start"],
            )
        except SlotUnavailable:
            result = self.svc.booking.check_availability(s["start"], s["service"])
            session.tool_log.append(
                {
                    "tool": "check_availability",
                    "service": s["service"],
                    "start": s["start"].isoformat(),
                    "available": result.available,
                    "alternatives": [a.isoformat() for a in result.alternatives],
                }
            )
            if result.alternatives:
                session.offered_alternatives = list(result.alternatives)
                session.state = "pick_alternative"
                return [
                    "I'm sorry, that time was just taken.",
                    self._alternatives_utterance(session),
                ]
            return self._escalate_with_notice(
                session, "no_availability",
                "I'm sorry, that time is no longer available. I'm connecting "
                "you to our front desk team who can help.",
            )
        session.tool_log.append({"tool": "book", "result": appt})
        session.booked_appointment = appt
        session.outcome = "booked"
        session.events.append({"type": "booking", "confirmation_code": appt["confirmation_code"]})
        start = datetime.fromisoformat(appt["start_ts"])
        utterances = [
            f"You're booked. Your confirmation code is {appt['confirmation_code']}. "
            f"That's a {appt['service']} on {format_dt(start)} for {appt['patient_name']}. "
            f"We'll send reminders to {appt['phone']}."
        ]
        self._store_readback_facts(session)
        self._build_confirm_queue(session)
        if not self.svc.memory.get_consent(session.caller_id) and not session.consent_asked:
            session.consent_asked = True
            session.state = "consent"
            utterances.append(
                "Before you go: would you like me to remember your name, "
                "phone number, and preferences for next time? I won't save "
                "anything without your permission."
            )
        elif session.confirm_queue:
            session.state = "confirm_items"
            utterances.append(self._confirm_question(session.confirm_queue[0]))
        else:
            session.state = "done"
            utterances.append(f"Thanks for calling {CLINIC_NAME}. Goodbye.")
        return utterances

    def _store_readback_facts(self, session: DialogueSession) -> None:
        """Persist read-back-confirmed facts under the store's rules.
        With no consent yet, they wait in pending_facts for the consent
        step; with consent, they go through assert_fact (which flags
        conflicts instead of overwriting)."""
        if not self.svc.memory.get_consent(session.caller_id):
            return
        remaining = []
        for fact in session.pending_facts:
            if fact["route"] != "readback":
                remaining.append(fact)
                continue
            result = self.svc.memory.assert_fact(
                session.caller_id,
                fact["key"],
                fact["value"],
                via_readback=True,
                source="booking_readback",
                ts=session.now.isoformat(),
            )
            if result.kind == "conflict_flagged":
                session.events.append({"type": "memory_conflict", "key": fact["key"]})
        session.pending_facts = remaining

    def _build_confirm_queue(self, session: DialogueSession) -> None:
        queue: list[ConfirmItem] = []
        seen: set[str] = set()
        mem = self.svc.memory
        for conflict in mem.pending_conflicts(session.caller_id):
            queue.append(
                ConfirmItem(
                    kind="conflict",
                    key=conflict["key"],
                    value=conflict["asserted_value"],
                    existing_value=conflict["existing_value"],
                )
            )
            seen.add(conflict["key"])
        for key in session.provisional_keys:
            if key in seen:
                continue
            for fact in mem.get_facts(session.caller_id):
                if fact["key"] == key and fact["status"] == "provisional":
                    queue.append(ConfirmItem(kind="fact", key=key, value=fact["value"]))
                    seen.add(key)
        session.confirm_queue = queue

    def _confirm_question(self, item: ConfirmItem) -> str:
        if item.kind == "conflict":
            hint = ""
            if item.key == "phone" and item.existing_value:
                hint = f" ending in {item.existing_value[-4:]}"
            return (
                f"I have a different {item.key} on file{hint}. You gave "
                f"{item.value} today. Should I update your saved {item.key} "
                f"to {item.value}?"
            )
        return (
            f"You mentioned {self._describe_fact(item.key, item.value)}. "
            "Should I remember that for next time?"
        )

    def _state_consent(self, session: DialogueSession, text: str) -> list[str]:
        lower = text.lower()
        if _is_affirmation(lower) and not _is_negation(lower):
            self.svc.memory.set_consent(session.caller_id, True, session.now.isoformat())
            session.events.append({"type": "consent", "granted": True})
            # Flush everything observed this call through the store rules.
            for fact in list(session.pending_facts):
                if fact["route"] == "readback":
                    result = self.svc.memory.assert_fact(
                        session.caller_id, fact["key"], fact["value"],
                        via_readback=True, source="booking_readback",
                        ts=session.now.isoformat(),
                    )
                    if result.kind == "conflict_flagged":
                        session.events.append(
                            {"type": "memory_conflict", "key": fact["key"]}
                        )
                else:
                    result = self.svc.memory.assert_fact(
                        session.caller_id, fact["key"], fact["value"],
                        ts=session.now.isoformat(),
                    )
                    if result.kind == "stored_provisional":
                        session.provisional_keys.append(fact["key"])
            session.pending_facts.clear()
            self._build_confirm_queue(session)
            utterances = ["Thank you, I'll remember those details."]
            if session.confirm_queue:
                session.state = "confirm_items"
                utterances.append(self._confirm_question(session.confirm_queue[0]))
            else:
                session.state = "done"
                utterances.append(f"Thanks for calling {CLINIC_NAME}. Goodbye.")
            return utterances
        if _is_negation(lower):
            self.svc.memory.set_consent(session.caller_id, False, session.now.isoformat())
            session.events.append({"type": "consent", "granted": False})
            session.pending_facts.clear()
            session.provisional_keys.clear()
            session.state = "done"
            return [
                "No problem, I won't save your details. "
                f"Thanks for calling {CLINIC_NAME}. Goodbye."
            ]
        return [
            "Sorry, I didn't catch that. Would you like me to remember your "
            "details for next time?"
        ]

    def _state_confirm_items(self, session: DialogueSession, text: str) -> list[str]:
        if not session.confirm_queue:
            session.state = "done"
            return [f"Thanks for calling {CLINIC_NAME}. Goodbye."]
        item = session.confirm_queue[0]
        lower = text.lower()
        mem = self.svc.memory
        if _is_affirmation(lower) and not _is_negation(lower):
            if item.kind == "fact":
                mem.confirm_fact(session.caller_id, item.key, ts=session.now.isoformat())
            else:
                mem.resolve_conflict(
                    session.caller_id, item.key, accept_new=True,
                    ts=session.now.isoformat(),
                )
            session.events.append({"type": "memory_confirmed", "key": item.key})
            session.confirm_queue.pop(0)
            lead = "Done, I'll remember that."
        elif _is_negation(lower):
            if item.kind == "fact":
                mem.discard_provisional(session.caller_id, item.key)
            else:
                mem.resolve_conflict(
                    session.caller_id, item.key, accept_new=False,
                    ts=session.now.isoformat(),
                )
            session.events.append({"type": "memory_declined", "key": item.key})
            session.confirm_queue.pop(0)
            lead = "Okay, I won't change that."
        else:
            return ["Sorry, I didn't catch that. " + self._confirm_question(item)]
        utterances = [lead]
        if session.confirm_queue:
            utterances.append(self._confirm_question(session.confirm_queue[0]))
        else:
            session.state = "done"
            utterances.append(f"Thanks for calling {CLINIC_NAME}. Goodbye.")
        return utterances

    def _state_done(self, session: DialogueSession, text: str) -> list[str]:
        return [f"Thanks for calling {CLINIC_NAME}. Goodbye."]
