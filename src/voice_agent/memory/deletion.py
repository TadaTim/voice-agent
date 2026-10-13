"""Provable deletion.

Deleting a caller removes their memory facts, conflicts, history, consent
record, transcripts and transcript index entries, and anonymizes their
appointment records. Deletion is not trusted -- it is *verified*: after
the deletes, an independent scan re-reads every store looking for any
residual reference to the caller (by caller id, by name, or by phone
digits). The scan result is emitted as a deletion receipt; ``verified``
is true only when every store reports zero residuals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..tools.booking import BookingService
from .store import MemoryStore
from .transcripts import TranscriptStore


@dataclass
class DeletionReceipt:
    caller_id: str
    deleted_at: str
    deleted_counts: dict[str, int]
    anonymized_appointments: int
    residuals: dict[str, int]
    total_residuals: int
    verified: bool
    receipt_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def scan_for_residuals(
    *,
    memory: MemoryStore,
    transcripts: TranscriptStore,
    booking: BookingService,
    caller_id: str,
    name: str = "",
    phone: str = "",
) -> dict[str, int]:
    """Count anything in any store still traceable to the caller."""
    phone_digits = "".join(c for c in phone if c.isdigit())
    return {
        "memory": memory.count_identifier_matches(
            caller_id=caller_id, name=name, phone_digits=phone_digits
        ),
        "transcripts": transcripts.count_identifier_matches(
            caller_id=caller_id, name=name, phone_digits=phone_digits
        ),
        "appointments": booking.count_identifier_matches(
            caller_id=caller_id, name=name, phone_digits=phone_digits
        ),
    }


def delete_caller_data(
    *,
    memory: MemoryStore,
    transcripts: TranscriptStore,
    booking: BookingService,
    caller_id: str,
    name: str = "",
    phone: str = "",
    deleted_at: str,
) -> DeletionReceipt:
    """Delete everywhere, then prove it with a zero-residual scan."""
    deleted_counts: dict[str, int] = {}
    deleted_counts.update(memory.delete_caller(caller_id))
    deleted_counts.update(transcripts.delete_caller(caller_id))
    anonymized = booking.anonymize_caller(caller_id)

    residuals = scan_for_residuals(
        memory=memory,
        transcripts=transcripts,
        booking=booking,
        caller_id=caller_id,
        name=name,
        phone=phone,
    )
    total = sum(residuals.values())
    payload = {
        "caller_id": caller_id,
        "deleted_at": deleted_at,
        "deleted_counts": deleted_counts,
        "anonymized_appointments": anonymized,
        "residuals": residuals,
    }
    receipt_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return DeletionReceipt(
        caller_id=caller_id,
        deleted_at=deleted_at,
        deleted_counts=deleted_counts,
        anonymized_appointments=anonymized,
        residuals=residuals,
        total_residuals=total,
        verified=(total == 0),
        receipt_id=f"DEL-{receipt_id}",
    )
