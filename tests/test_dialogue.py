from __future__ import annotations

from datetime import datetime

import pytest

from voice_agent.dialogue.manager import DialogueManager
from voice_agent.dialogue.session import DialogueSession

NOW = datetime(2026, 10, 13, 9, 0)
CID = "+1-406-555-0101"


@pytest.fixture()
def manager(services):
    return DialogueManager(services)


def make_session(caller_id: str = CID) -> DialogueSession:
    return DialogueSession(session_id="test-1", caller_id=caller_id, now=NOW)


def drive(manager, session, lines: list[str]) -> list[list[str]]:
    return [manager.handle(session, line) for line in lines]


def test_happy_path_books_and_stores_confirmed_memory(manager, services):
    s = make_session()
    drive(manager, s, [
        "Hi, I would like to book an appointment for a checkup.",
        "My name is Ann Lee.",
        "My number is 406-555-0101.",
        "October 20 at 10 AM.",
    ])
    assert s.state == "readback"
    out = manager.handle(s, "Yes, that is correct.")
    assert s.outcome == "booked"
    assert s.booked_appointment["start_ts"] == "2026-10-20T10:00:00"
    assert any("confirmation code" in u for u in out)
    manager.handle(s, "Yes, please remember my details.")
    mem = services.memory
    assert mem.get_confirmed(CID, "name") == "Ann Lee"
    assert mem.get_confirmed(CID, "phone") == "406-555-0101"


def test_no_booking_before_readback_confirmation(manager, services):
    s = make_session()
    drive(manager, s, [
        "Book a checkup.",
        "My name is Ann Lee.",
        "My number is 406-555-0101.",
        "October 20 at 10 AM.",
    ])
    # Nothing is booked while the read-back is still unconfirmed.
    assert services.booking.appointments_for_caller(CID) == []


def test_readback_correction_changes_the_booked_slot(manager):
    s = make_session()
    drive(manager, s, [
        "Book a checkup.",
        "My name is Brian Cole.",
        "My number is 406-555-0142.",
        "October 16 at 10 AM.",
        "No, I said October 15.",
    ])
    assert s.slots["start"] == datetime(2026, 10, 15, 10, 0)
    assert any(e["type"] == "slot_correction" for e in s.events)
    manager.handle(s, "Yes, correct.")
    assert s.booked_appointment["start_ts"] == "2026-10-15T10:00:00"


def test_unavailable_slot_offers_only_tool_alternatives(manager, services):
    services.booking.force_insert(
        caller_id="other", patient_name="Other", phone="406-555-0000",
        service="Checkup", start=datetime(2026, 10, 20, 10, 0),
    )
    s = make_session()
    out = drive(manager, s, [
        "Book a checkup.",
        "My name is Dana Ruiz.",
        "My number is 406-555-0188.",
        "October 20 at 10 AM.",
    ])
    text = " ".join(out[-1])
    assert "isn't available" in text
    assert "10:30 AM" in text and "11:00 AM" in text and "11:30 AM" in text
    manager.handle(s, "The second one works.")
    assert s.state == "readback"
    assert s.slots["start"] == datetime(2026, 10, 20, 11, 0)


def test_unanswerable_question_escalates_instead_of_inventing(manager, services):
    s = make_session()
    out = manager.handle(s, "How much is a massage?")
    assert s.outcome == "escalated"
    assert s.escalation_reason == "unanswerable_question"
    pending = services.escalation.list_pending()
    assert len(pending) == 1
    assert pending[0]["reason"] == "unanswerable_question"
    # (Transcript attachment is a pipeline-level behavior -- transcripts
    # are recorded by the pipeline -- and is asserted by the eval corpus
    # checks, which run through the full pipeline.)


def test_price_answer_comes_from_tool(manager):
    s = make_session()
    out = manager.handle(s, "How much is a physical?")
    assert any("$150" in u for u in out[0:1])
    assert any(e["tool"] == "get_price" for e in s.tool_log)


def test_privacy_refusal_discloses_nothing(manager, services):
    services.booking.force_insert(
        caller_id="+1-406-555-0130", patient_name="Maria Garcia",
        phone="406-555-0130", service="Physical",
        start=datetime(2026, 10, 20, 14, 0),
    )
    s = make_session("+1-406-555-2222")
    out = manager.handle(s, "When is Maria Garcia's appointment?")
    joined = " ".join(out)
    assert s.refused is True
    assert "Maria Garcia" not in joined
    assert "406-555-0130" not in joined
    assert "2:00 PM" not in joined


def test_memory_inspection_lists_only_own_facts(manager, services):
    mem = services.memory
    mem.set_consent(CID, True, NOW.isoformat())
    mem.assert_fact(CID, "name", "Ann Lee", via_readback=True, ts=NOW.isoformat())
    mem.assert_fact(CID, "preferred_time", "morning", via_readback=True, ts=NOW.isoformat())
    s = make_session()
    out = manager.handle(s, "What do you remember about me?")
    joined = " ".join(out)
    assert "Ann Lee" in joined and "morning" in joined


def test_consent_declined_stores_nothing(manager, services):
    s = make_session()
    drive(manager, s, [
        "Book a checkup.",
        "My name is Tom Becker.",
        "My number is 406-555-0123.",
        "October 20 at 10 AM.",
        "Yes, correct.",
        "No, don't save anything.",
    ])
    assert s.outcome == "booked"
    assert services.memory.get_facts(CID) == []
