from __future__ import annotations

from datetime import datetime

from voice_agent.memory.deletion import delete_caller_data, scan_for_residuals

CID = "+1-406-555-0101"
NAME = "Ann Lee"
PHONE = "406-555-0101"
TS = "2026-10-13T09:00:00"


def _seed_everything(services) -> None:
    mem, tx, bk = services.memory, services.transcripts, services.booking
    mem.set_consent(CID, True, TS)
    mem.assert_fact(CID, "name", NAME, via_readback=True, ts=TS)
    mem.assert_fact(CID, "phone", PHONE, via_readback=True, ts=TS)
    mem.assert_fact(CID, "phone", "406-555-9999", ts=TS)  # conflict + provisional
    mem.resolve_conflict(CID, "phone", accept_new=False, ts=TS)
    mem.assert_fact(CID, "preferred_time", "morning", ts=TS)
    tx.append(session_id="s1", caller_id=CID, role="caller",
              text="My number is 406-555-0101", ts=TS)
    tx.append(session_id="s1", caller_id=CID, role="agent",
              text="Thanks, Ann Lee", ts=TS)
    bk.book(caller_id=CID, patient_name=NAME, phone=PHONE,
            service="Checkup", start=datetime(2026, 10, 20, 11, 0))
    # Another caller whose data must survive untouched.
    mem.set_consent("+1-406-555-0202", True, TS)
    mem.assert_fact("+1-406-555-0202", "name", "Sara Kim", via_readback=True, ts=TS)


def test_deletion_is_provably_complete(services):
    _seed_everything(services)
    before = scan_for_residuals(
        memory=services.memory, transcripts=services.transcripts,
        booking=services.booking, caller_id=CID, name=NAME, phone=PHONE,
    )
    assert all(v > 0 for v in before.values())

    receipt = delete_caller_data(
        memory=services.memory, transcripts=services.transcripts,
        booking=services.booking, caller_id=CID, name=NAME, phone=PHONE,
        deleted_at="2026-10-13T10:00:00",
    )
    assert receipt.verified is True
    assert receipt.total_residuals == 0
    assert receipt.residuals == {"memory": 0, "transcripts": 0, "appointments": 0}
    assert receipt.receipt_id.startswith("DEL-")
    assert receipt.deleted_counts["memory_facts"] >= 3
    assert receipt.deleted_counts["transcripts"] == 2
    assert receipt.deleted_counts["transcript_index"] > 0
    assert receipt.anonymized_appointments == 1

    # Independent verification: raw store reads, not the receipt's word.
    assert services.memory.get_facts(CID) == []
    assert services.memory.get_consent(CID) is False
    assert services.transcripts.for_caller(CID) == []
    assert services.booking.appointments_for_caller(CID) == []
    after = scan_for_residuals(
        memory=services.memory, transcripts=services.transcripts,
        booking=services.booking, caller_id=CID, name=NAME, phone=PHONE,
    )
    assert all(v == 0 for v in after.values())

    # The other caller's data is untouched.
    assert services.memory.get_confirmed("+1-406-555-0202", "name") == "Sara Kim"
