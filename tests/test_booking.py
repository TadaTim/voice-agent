from __future__ import annotations

from datetime import datetime

import pytest

from voice_agent.tools.booking import BookingService, SlotUnavailable


@pytest.fixture()
def booking(tmp_path):
    svc = BookingService(str(tmp_path / "b.db"))
    yield svc
    svc.close()


def test_free_slot_is_available(booking):
    result = booking.check_availability(datetime(2026, 10, 20, 10, 0), "Checkup")
    assert result.available is True
    assert result.alternatives == []


def test_taken_slot_reports_deterministic_alternatives(booking):
    booking.force_insert(
        caller_id="x", patient_name="Other", phone="406-555-0000",
        service="Checkup", start=datetime(2026, 10, 20, 10, 0),
    )
    result = booking.check_availability(datetime(2026, 10, 20, 10, 0), "Checkup")
    assert result.available is False
    assert result.reason == "slot_taken"
    assert [a.isoformat() for a in result.alternatives] == [
        "2026-10-20T10:30:00",
        "2026-10-20T11:00:00",
        "2026-10-20T11:30:00",
    ]


def test_weekend_and_after_hours_are_unavailable(booking):
    saturday = booking.check_availability(datetime(2026, 10, 17, 10, 0), "Checkup")
    assert saturday.available is False
    assert saturday.reason == "clinic_closed_weekend"
    evening = booking.check_availability(datetime(2026, 10, 20, 16, 45), "Checkup")
    assert evening.available is False
    assert evening.reason == "outside_clinic_hours"


def test_double_booking_is_rejected_at_write_time(booking):
    start = datetime(2026, 10, 20, 11, 0)
    booking.book(
        caller_id="a", patient_name="A", phone="406-555-0001",
        service="Checkup", start=start,
    )
    with pytest.raises(SlotUnavailable):
        booking.book(
            caller_id="b", patient_name="B", phone="406-555-0002",
            service="Physical", start=start,
        )


def test_prices_come_from_the_service_table(booking):
    assert booking.get_price_cents("Physical") == 15000
    assert booking.get_price_cents("flu shot") == 4000
    assert booking.get_price_cents("Massage") is None
    assert booking.service_info("Massage") is None


def test_anonymize_removes_identifiers(booking):
    booking.book(
        caller_id="caller-1", patient_name="Ann Lee", phone="406-555-0101",
        service="Checkup", start=datetime(2026, 10, 20, 11, 0),
    )
    assert booking.count_identifier_matches(
        caller_id="caller-1", name="Ann Lee", phone_digits="4065550101"
    ) == 1
    assert booking.anonymize_caller("caller-1") == 1
    assert booking.count_identifier_matches(
        caller_id="caller-1", name="Ann Lee", phone_digits="4065550101"
    ) == 0
    rows = booking.appointments_for_caller("caller-1")
    assert rows == []
