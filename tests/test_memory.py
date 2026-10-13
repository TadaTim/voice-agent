from __future__ import annotations

import pytest

from voice_agent.memory.store import MemoryStore

CID = "+1-406-555-0101"
TS = "2026-10-13T09:00:00"


@pytest.fixture()
def memory(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    yield store
    store.close()


def test_no_consent_no_storage(memory):
    result = memory.assert_fact(CID, "name", "Ann Lee", ts=TS)
    assert result.kind == "not_stored_no_consent"
    assert memory.get_facts(CID) == []


def test_provisional_until_readback_confirms(memory):
    memory.set_consent(CID, True, TS)
    result = memory.assert_fact(CID, "preferred_time", "morning", ts=TS)
    assert result.kind == "stored_provisional"
    assert memory.get_confirmed(CID, "preferred_time") is None
    assert memory.confirm_fact(CID, "preferred_time", ts=TS) is True
    assert memory.get_confirmed(CID, "preferred_time") == "morning"


def test_readback_assertion_stores_confirmed(memory):
    memory.set_consent(CID, True, TS)
    result = memory.assert_fact(CID, "name", "Ann Lee", via_readback=True, ts=TS)
    assert result.kind == "stored_confirmed"
    assert memory.get_confirmed(CID, "name") == "Ann Lee"


def test_conflict_is_flagged_and_confirmed_value_survives(memory):
    memory.set_consent(CID, True, TS)
    memory.assert_fact(CID, "phone", "406-555-0101", via_readback=True, ts=TS)
    result = memory.assert_fact(CID, "phone", "406-555-9999", ts=TS)
    assert result.kind == "conflict_flagged"
    assert result.existing_value == "406-555-0101"
    # The poisoning defense: stored value is NOT silently overwritten.
    assert memory.get_confirmed(CID, "phone") == "406-555-0101"
    assert len(memory.pending_conflicts(CID)) == 1


def test_conflict_resolution_reject_keeps_original(memory):
    memory.set_consent(CID, True, TS)
    memory.assert_fact(CID, "phone", "406-555-0101", via_readback=True, ts=TS)
    memory.assert_fact(CID, "phone", "406-555-9999", ts=TS)
    assert memory.resolve_conflict(CID, "phone", accept_new=False, ts=TS) is True
    assert memory.get_confirmed(CID, "phone") == "406-555-0101"
    assert memory.pending_conflicts(CID) == []
    facts = {f["key"]: f for f in memory.get_facts(CID)}
    assert facts["phone"]["status"] == "confirmed"


def test_conflict_resolution_accept_archives_history(memory):
    memory.set_consent(CID, True, TS)
    memory.assert_fact(CID, "phone", "406-555-0101", via_readback=True, ts=TS)
    memory.assert_fact(CID, "phone", "406-555-9999", ts=TS)
    assert memory.resolve_conflict(CID, "phone", accept_new=True, ts=TS) is True
    assert memory.get_confirmed(CID, "phone") == "406-555-9999"
    rows = memory._conn.execute(
        "SELECT * FROM memory_history WHERE caller_id = ?", (CID,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["old_value"] == "406-555-0101"


def test_discard_provisional(memory):
    memory.set_consent(CID, True, TS)
    memory.assert_fact(CID, "email", "ann@example.com", ts=TS)
    assert memory.discard_provisional(CID, "email") is True
    assert memory.get_facts(CID) == []
