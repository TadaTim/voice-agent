from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from voice_agent.dialogue.session import Services
from voice_agent.escalation import EscalationQueue
from voice_agent.memory.store import MemoryStore
from voice_agent.memory.transcripts import TranscriptStore
from voice_agent.tools.booking import BookingService


@pytest.fixture()
def services():
    base = Path(tempfile.mkdtemp(prefix="va-test-"))
    svc = Services(
        booking=BookingService(str(base / "booking.db")),
        memory=MemoryStore(str(base / "memory.db")),
        transcripts=TranscriptStore(str(base / "transcripts.db")),
        escalation=EscalationQueue(str(base / "escalation.db")),
    )
    yield svc
    svc.booking.close()
    svc.memory.close()
    svc.transcripts.close()
    svc.escalation.close()
