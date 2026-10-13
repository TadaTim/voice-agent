"""Speech-to-text engines.

``SimulatedStt`` decodes pseudo-audio frames back to text. It is NOT speech
recognition: no acoustic model is involved. A *mishearing profile* (a set of
literal substitutions) may be attached to model recognition errors, so the
dialogue layer's read-back protocol can be exercised and evaluated against
realistic STT mistakes. Profiles are declared in the eval corpus and labeled
as simulated recognition errors.

``FasterWhisperStt`` is a real local engine adapter (optional dependency);
it transcribes real PCM audio with a locally-downloaded Whisper model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .frames import AudioFrame, Segment, frames_duration_ms

# Modeled real-time factor for the simulated decoder: decoding is claimed to
# take 20% of the audio duration. This is a *model*, reported as such.
SIMULATED_STT_RTF = 0.20


@dataclass
class SttResult:
    text: str
    audio_ms: int
    latency_ms: float
    latency_basis: str  # "measured" | "modeled"
    engine: str


class Stt(Protocol):
    name: str
    simulated: bool

    def transcribe(self, segments: list[Segment]) -> SttResult:
        ...


@dataclass
class SimulatedStt:
    """Deterministic frame decoder with optional simulated mishearings.

    ``mishearings`` is an ordered mapping of literal substitutions applied
    to the decoded text, e.g. {"fifteenth": "sixteenth"} or
    {"0101": "0107"}. It models *that* an STT error occurred; the dialogue
    layer must catch it via read-back, exactly as with a real engine.
    """

    mishearings: dict[str, str] = field(default_factory=dict)
    name: str = "simulated-stt"
    simulated: bool = True

    def transcribe(self, segments: list[Segment]) -> SttResult:
        frames: list[AudioFrame] = [f for seg in segments for f in seg.frames]
        text = "".join(f.payload for f in frames)
        for source, replacement in self.mishearings.items():
            text = text.replace(source, replacement)
        audio_ms = frames_duration_ms(frames)
        return SttResult(
            text=text,
            audio_ms=audio_ms,
            latency_ms=round(audio_ms * SIMULATED_STT_RTF, 3),
            latency_basis="modeled",
            engine=self.name,
        )


def _resample_linear(samples, src_rate: int, dst_rate: int):
    """Minimal linear-interpolation resampler (numpy). A pragmatic
    stand-in for a DSP resampler; adequate for speech intelligibility."""
    import numpy as np

    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    target_len = int(duration * dst_rate)
    positions = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(positions, np.arange(len(samples)), samples).astype("float32")


class FasterWhisperStt:
    """Real local STT via faster-whisper (optional dependency).

    Consumes real PCM frames (mono int16 at ``sample_rate``; Piper voices
    are typically 22050 Hz, telephony 16000/8000 Hz) and feeds the model a
    float32 array directly, resampled to Whisper's 16 kHz. Raises if
    constructed without the dependency or a model; the factory falls back
    to the simulated engine instead of failing the call.
    """

    name = "faster-whisper"
    simulated = False

    def __init__(
        self,
        model_size_or_path: str = "tiny",
        model_dir: str | None = None,
        *,
        sample_rate: int = 16000,
    ):
        from faster_whisper import WhisperModel  # optional dependency

        self._model = WhisperModel(
            model_size_or_path,
            device="cpu",
            compute_type="int8",
            download_root=model_dir,
        )
        self._sample_rate = sample_rate

    def transcribe(self, segments: list[Segment]) -> SttResult:
        import time

        import numpy as np

        pcm = b"".join(f.pcm for seg in segments for f in seg.frames)
        samples = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
        audio = _resample_linear(samples, self._sample_rate, 16000)
        t0 = time.perf_counter()
        parts, _info = self._model.transcribe(audio, beam_size=1)
        text = "".join(p.text for p in parts).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        audio_ms = int(len(samples) / self._sample_rate * 1000)
        return SttResult(
            text=text,
            audio_ms=audio_ms,
            latency_ms=round(latency_ms, 3),
            latency_basis="measured",
            engine=self.name,
        )
