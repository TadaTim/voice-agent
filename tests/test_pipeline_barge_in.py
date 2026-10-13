from __future__ import annotations

from datetime import datetime

import pytest

from voice_agent.audio.factory import build_audio_stack
from voice_agent.audio.frames import text_to_frames
from voice_agent.latency import LatencyRecorder
from voice_agent.pipeline import VoicePipeline

NOW = datetime(2026, 10, 13, 9, 0)
CID = "+1-406-555-0145"


@pytest.fixture()
def pipeline(services):
    return VoicePipeline(
        audio=build_audio_stack("simulated"),
        services=services,
        recorder=LatencyRecorder(),
    )


def _drive_to_readback(pipeline):
    session, _ = pipeline.start_call(CID, NOW, session_id="barge-test")
    for line in [
        "I'd like to book a vaccination.",
        "My name is Pete Wong.",
        "My number is 406-555-0145.",
    ]:
        pipeline.caller_turn(session, text_to_frames(line))
    outcome = pipeline.caller_turn(session, text_to_frames("October 19 at 9:30 AM."))
    return session, outcome


def test_playback_frame_accounting(pipeline):
    session, _ = pipeline.start_call(CID, NOW, session_id="frames-test")
    playback = pipeline.speak(session, "This is a long agent utterance for testing.")
    total = playback.total_frames
    played = sum(1 for _ in playback.play(max_frames=5))
    assert played == 5
    assert playback.played_frames == 5
    playback.interrupt()
    assert playback.interrupted is True
    assert playback.discarded_frames == total - 5


def test_barge_in_discards_partial_utterance_and_processes_correction(pipeline):
    session, outcome = _drive_to_readback(pipeline)
    readback = outcome.playbacks[-1]
    total = readback.total_frames
    assert total > 8

    result = pipeline.barge_in(
        session,
        readback,
        after_frames=8,
        caller_frames=text_to_frames("No, I need 10:30 instead."),
    )
    # Playback halted where interrupted; the remainder was discarded.
    assert readback.interrupted is True
    assert readback.played_frames == 8
    assert readback.discarded_frames == total - 8
    # The interruption was processed as the caller's turn.
    assert result.caller_text == "No, I need 10:30 instead."
    assert session.slots["start"] == datetime(2026, 10, 19, 10, 30)
    event = [e for e in session.events if e["type"] == "barge_in"]
    assert len(event) == 1
    assert event[0]["discarded_frames"] == total - 8

    # And the corrected booking completes normally.
    pipeline.caller_turn(session, text_to_frames("Yes, that's right."))
    assert session.outcome == "booked"
    assert session.booked_appointment["start_ts"] == "2026-10-19T10:30:00"


def test_pipeline_records_latency_for_every_stage(pipeline):
    session, _ = pipeline.start_call(CID, NOW, session_id="latency-test")
    pipeline.caller_turn(session, text_to_frames("How much is a physical?"))
    summary = pipeline.recorder.summary()
    for stage in ("vad", "stt", "dialogue", "tts", "turn_total"):
        assert stage in summary, stage
        assert summary[stage]["count"] >= 1
    assert summary["stt"]["basis"] == "modeled"
    assert summary["dialogue"]["basis"] == "measured"


def test_nothing_is_recorded_after_deletion(pipeline, services):
    session, _ = pipeline.start_call(CID, NOW, session_id="delete-test")
    services.memory.set_consent(CID, True, NOW.isoformat())
    services.memory.assert_fact(
        CID, "name", "Pete Wong", via_readback=True, ts=NOW.isoformat()
    )
    pipeline.caller_turn(session, text_to_frames("Please delete my information."))
    assert session.deletion_receipt is not None
    assert session.deletion_receipt["verified"] is True
    # The deletion-confirmation utterance itself, and anything after,
    # must not re-create transcript rows for this caller.
    assert services.transcripts.for_caller(CID) == []
    pipeline.caller_turn(session, text_to_frames("Thank you, goodbye."))
    assert services.transcripts.for_caller(CID) == []
