"""Real-engine integration tests.

Skipped unless the optional real-audio dependencies are installed AND
the models are present locally -- the simulated-path test suite is the
one that must always pass. When these run, they prove the real Piper ->
faster-whisper path works end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VOICE = ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"
WHISPER = ROOT / "models" / "whisper-tiny"


def _real_stack_available() -> bool:
    if not (VOICE.exists() and WHISPER.exists()):
        return False
    try:
        import faster_whisper  # noqa: F401
        import piper  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _real_stack_available(),
    reason="real audio engines/models not installed (optional path)",
)


def test_piper_synthesizes_real_audio():
    from voice_agent.audio.tts import PiperTts

    tts = PiperTts(str(VOICE))
    result = tts.synthesize("Hello, this is a test.")
    assert result.engine == "piper"
    assert tts.simulated is False
    assert result.latency_basis == "measured"
    assert result.wav_bytes and len(result.wav_bytes) > 1000
    assert result.audio_ms > 500
    assert all(f.pcm for f in result.frames)


def test_whisper_transcribes_piper_speech():
    from voice_agent.audio.frames import Segment
    from voice_agent.audio.stt import FasterWhisperStt
    from voice_agent.audio.tts import PiperTts

    tts = PiperTts(str(VOICE))
    stt = FasterWhisperStt(str(WHISPER), None, sample_rate=22050)
    synth = tts.synthesize("I would like to book an appointment.")
    segment = Segment(start_frame=0, end_frame=len(synth.frames), frames=synth.frames)
    heard = stt.transcribe([segment])
    assert heard.engine == "faster-whisper"
    assert heard.latency_basis == "measured"
    lowered = heard.text.lower()
    assert "appointment" in lowered
    assert "book" in lowered
