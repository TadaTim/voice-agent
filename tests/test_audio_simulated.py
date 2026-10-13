from __future__ import annotations

import pytest

from voice_agent.audio.factory import build_audio_stack
from voice_agent.audio.frames import text_to_frames
from voice_agent.audio.stt import SimulatedStt
from voice_agent.audio.tts import SimulatedTts
from voice_agent.audio.vad import EnergyVad


def test_text_frame_roundtrip_through_vad_and_stt():
    stack = build_audio_stack("simulated")
    frames = text_to_frames("I would like to book an appointment please")
    segments = stack.vad.segment(frames)
    assert len(segments) == 1
    result = stack.stt.transcribe(segments)
    assert result.text == "I would like to book an appointment please"
    assert result.latency_basis == "modeled"
    assert stack.stt.simulated is True


def test_vad_trims_silence_and_bridges_short_gaps():
    vad = EnergyVad()
    frames = text_to_frames("hello world", leading_silence_frames=10, trailing_silence_frames=10)
    segments = vad.segment(frames)
    assert len(segments) == 1
    seg = segments[0]
    assert all(f.is_speech or f.payload for f in seg.frames)
    assert seg.frames[0].payload != ""


def test_simulated_stt_applies_mishearing_profile():
    stt = SimulatedStt(mishearings={"fifteenth": "sixteenth"})
    frames = text_to_frames("October fifteenth at ten")
    segments = EnergyVad().segment(frames)
    result = stt.transcribe(segments)
    assert result.text == "October sixteenth at ten"


def test_simulated_tts_is_deterministic_and_labeled():
    tts = SimulatedTts()
    a = tts.synthesize("Your appointment is booked.")
    b = tts.synthesize("Your appointment is booked.")
    assert [f.payload for f in a.frames] == [f.payload for f in b.frames]
    assert a.audio_ms == b.audio_ms > 0
    assert a.latency_basis == "modeled"
    assert tts.simulated is True


def test_factory_report_is_honest_about_simulation():
    stack = build_audio_stack("simulated")
    report = stack.report()
    assert report["stt"] == {"engine": "simulated-stt", "simulated": True}
    assert report["tts"] == {"engine": "simulated-tts", "simulated": True}


def test_factory_real_mode_raises_without_voice_model():
    with pytest.raises(Exception):
        build_audio_stack("real", piper_voice_path=None)
