"""Text-to-speech engines.

``SimulatedTts`` encodes agent text into pseudo-audio frames. It produces
no sound and claims none; what it provides is a frame stream with a
deterministic duration, which is what the pipeline's playback and barge-in
machinery operate on.

``PiperTts`` is a real local TTS adapter (optional dependency) producing
actual WAV audio via a locally-downloaded Piper voice.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Protocol

from .frames import FRAME_MS, AudioFrame, text_to_frames

# Modeled real-time factor for simulated synthesis (see stt.py).
SIMULATED_TTS_RTF = 0.25


@dataclass
class TtsResult:
    frames: list[AudioFrame]
    audio_ms: int
    latency_ms: float
    latency_basis: str  # "measured" | "modeled"
    engine: str
    wav_bytes: bytes | None = None  # real path only


class Tts(Protocol):
    name: str
    simulated: bool

    def synthesize(self, text: str) -> TtsResult:
        ...


class SimulatedTts:
    name = "simulated-tts"
    simulated = True

    def synthesize(self, text: str) -> TtsResult:
        frames = text_to_frames(text, leading_silence_frames=0, trailing_silence_frames=0)
        audio_ms = sum(f.duration_ms for f in frames)
        return TtsResult(
            frames=frames,
            audio_ms=audio_ms,
            latency_ms=round(audio_ms * SIMULATED_TTS_RTF, 3),
            latency_basis="modeled",
            engine=self.name,
        )


class PiperTts:
    """Real local TTS via Piper (optional dependency).

    ``voice_path`` is a local .onnx voice model (with its .json config
    alongside). Produces real 16-bit PCM WAV audio, split into 20 ms
    frames with energy computed from sample RMS.
    """

    name = "piper"
    simulated = False

    def __init__(self, voice_path: str):
        from piper import PiperVoice  # optional dependency

        self._voice = PiperVoice.load(voice_path)

    def synthesize(self, text: str) -> TtsResult:
        import io
        import time
        import wave

        buf = io.BytesIO()
        t0 = time.perf_counter()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)  # sets the WAV format itself
        latency_ms = (time.perf_counter() - t0) * 1000.0
        wav_bytes = buf.getvalue()

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = struct.unpack(f"<{len(raw) // 2}h", raw)
        frame_samples = max(1, int(sample_rate * FRAME_MS / 1000))
        frames: list[AudioFrame] = []
        for i in range(0, len(samples), frame_samples):
            chunk = samples[i : i + frame_samples]
            if not chunk:
                break
            rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
            energy = min(1.0, rms / 8000.0)
            pcm = struct.pack(f"<{len(chunk)}h", *chunk)
            frames.append(
                AudioFrame(index=len(frames), energy=round(energy, 3), pcm=pcm)
            )
        audio_ms = int(len(samples) / sample_rate * 1000)
        return TtsResult(
            frames=frames,
            audio_ms=audio_ms,
            latency_ms=round(latency_ms, 3),
            latency_basis="measured",
            engine=self.name,
            wav_bytes=wav_bytes,
        )
