#!/usr/bin/env python3
"""Scripted demo call through the full pipeline (simulated audio):
a booking with a barge-in correction during the read-back, followed by a
memory deletion with its verification receipt.

    python scripts/demo_call.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_agent.audio.factory import build_audio_stack  # noqa: E402
from voice_agent.audio.frames import text_to_frames  # noqa: E402
from voice_agent.dialogue.session import Services  # noqa: E402
from voice_agent.escalation import EscalationQueue  # noqa: E402
from voice_agent.latency import LatencyRecorder  # noqa: E402
from voice_agent.memory.store import MemoryStore  # noqa: E402
from voice_agent.memory.transcripts import TranscriptStore  # noqa: E402
from voice_agent.pipeline import VoicePipeline  # noqa: E402
from voice_agent.tools.booking import BookingService  # noqa: E402


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="va-demo-"))
    services = Services(
        booking=BookingService(str(base / "booking.db")),
        memory=MemoryStore(str(base / "memory.db")),
        transcripts=TranscriptStore(str(base / "transcripts.db")),
        escalation=EscalationQueue(str(base / "escalation.db")),
    )
    recorder = LatencyRecorder()
    pipeline = VoicePipeline(
        audio=build_audio_stack("simulated"), services=services, recorder=recorder
    )
    session, greeting = pipeline.start_call(
        "+1-406-555-0101", datetime(2026, 10, 13, 9, 0), session_id="demo-1"
    )
    for utterance in greeting.utterances:
        print(f"AGENT : {utterance}")

    script = [
        "Hi, I would like to book an appointment for a checkup.",
        "My name is Ann Lee.",
        "My number is 406-555-0101.",
        "October 20 at 10 AM.",
    ]
    last = None
    for line in script:
        last = pipeline.caller_turn(session, text_to_frames(line))
        print(f"CALLER: {line}")
        for utterance in last.utterances:
            print(f"AGENT : {utterance}")

    print("\n-- caller barges in during the read-back, 10 frames in --")
    result = pipeline.barge_in(
        session,
        last.playbacks[-1],
        after_frames=10,
        caller_frames=text_to_frames("No, the twenty-first, not the twentieth."),
    )
    print("CALLER: No, the twenty-first, not the twentieth.")
    for utterance in result.utterances:
        print(f"AGENT : {utterance}")

    for line in ["Yes, that is correct.", "Yes, please remember my details."]:
        out = pipeline.caller_turn(session, text_to_frames(line))
        print(f"CALLER: {line}")
        for utterance in out.utterances:
            print(f"AGENT : {utterance}")

    print("\n-- a later call: the caller asks to be forgotten --")
    session2, _ = pipeline.start_call(
        "+1-406-555-0101", datetime(2026, 10, 14, 9, 0), session_id="demo-2"
    )
    out = pipeline.caller_turn(session2, text_to_frames("Please delete my information."))
    for utterance in out.utterances:
        print(f"AGENT : {utterance}")
    print("\ndeletion receipt:")
    print(json.dumps(session2.deletion_receipt, indent=2))

    print("\nlatency summary (this demo's runs):")
    for stage, stats in recorder.summary().items():
        print(
            f"  {stage:<10} p50 {stats['p50_ms']:>8.3f} ms  "
            f"p95 {stats['p95_ms']:>8.3f} ms  ({stats['basis']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
