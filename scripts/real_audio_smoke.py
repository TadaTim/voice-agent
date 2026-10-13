#!/usr/bin/env python3
"""Real-audio smoke test: Piper (real TTS) speaks scripted sentences and
faster-whisper (real STT) transcribes them. This is the evidence behind
the README's 'real engines ran' claim; it requires the optional
real-audio dependencies and locally downloaded models:

    pip install -e '.[real-audio]'
    python -m piper.download_voices --download-dir models/piper en_US-lessac-medium
    # faster-whisper model: 'tiny' downloads on first use, or pass a local dir

Usage: python scripts/real_audio_smoke.py [whisper_model_or_path]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_agent.audio.frames import Segment  # noqa: E402
from voice_agent.audio.stt import FasterWhisperStt  # noqa: E402
from voice_agent.audio.tts import PiperTts  # noqa: E402

SENTENCES = [
    "Hello, I would like to book a checkup appointment on Tuesday morning.",
    "My phone number is 406-555-0101.",
]


def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny"
    voice = "models/piper/en_US-lessac-medium.onnx"
    tts = PiperTts(voice)
    stt = FasterWhisperStt(model, None, sample_rate=22050)
    for text in SENTENCES:
        synth = tts.synthesize(text)
        segment = Segment(
            start_frame=0, end_frame=len(synth.frames), frames=synth.frames
        )
        heard = stt.transcribe([segment])
        said, got = _norm(text), _norm(heard.text)
        matched = sum(1 for x, y in zip(said, got) if x == y)
        print(f"SAID : {text}")
        print(
            f"HEARD: {heard.text!r}  (tts {synth.latency_ms:.0f} ms synth / "
            f"{synth.audio_ms} ms audio; stt {heard.latency_ms:.0f} ms; measured)"
        )
        print(f"words matched in order: {matched}/{len(said)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
