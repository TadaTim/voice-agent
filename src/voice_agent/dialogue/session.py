"""Dialogue session state and the service bundle the manager runs on."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..escalation import EscalationQueue
from ..memory.store import MemoryStore
from ..memory.transcripts import TranscriptStore
from ..tools.booking import BookingService

CLINIC_NAME = "BrightPath Clinic"


@dataclass
class Services:
    """Everything the dialogue layer is allowed to touch."""

    booking: BookingService
    memory: MemoryStore
    transcripts: TranscriptStore
    escalation: EscalationQueue


@dataclass
class ConfirmItem:
    """A memory item awaiting the caller's explicit confirmation."""

    kind: str  # 'fact' (provisional fact) | 'conflict' (change vs. on-file value)
    key: str
    value: str
    existing_value: str | None = None


@dataclass
class DialogueSession:
    session_id: str
    caller_id: str
    now: datetime
    state: str = "greeting"
    slots: dict = field(default_factory=dict)  # name, phone, service, start
    when_day: object = None  # partially-collected date (datetime.date)
    when_tod: object = None  # partially-collected time (datetime.time)
    offered_alternatives: list = field(default_factory=list)
    # Facts observed this call that are not yet in the memory store:
    # {key, value, route} with route 'readback' (confirmed by the booking
    # read-back) or 'confirm_step' (need their own confirmation question).
    pending_facts: list[dict] = field(default_factory=list)
    provisional_keys: list[str] = field(default_factory=list)
    confirm_queue: list[ConfirmItem] = field(default_factory=list)
    service_attempts: int = 0
    consent_asked: bool = False
    refused: bool = False
    outcome: str | None = None  # booked | escalated | deleted | info
    escalation_reason: str | None = None
    escalation_priority: str | None = None
    booked_appointment: dict | None = None
    deletion_receipt: dict | None = None
    events: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    turns: int = 0
