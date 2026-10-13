"""Audio frames: the unit that flows through the pipeline.

A frame is a fixed 20 ms slice. On the simulated path a frame carries a
small slice of the source text as its payload (this is what makes the
simulated STT a *decoder*, not a bypass: text is tokenized into frames,
optionally corrupted by a mishearing profile, and re-assembled by the STT
stage). On the real path a frame carries raw 16-bit PCM samples instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FRAME_MS = 20
# Simulated speaking rate: ~16.7 chars/sec ~= 1000 chars/min of speech.
CHARS_PER_FRAME = 3
SILENCE_ENERGY = 0.0
SPEECH_ENERGY = 0.8


@dataclass
class AudioFrame:
    index: int
    duration_ms: int = FRAME_MS
    energy: float = SILENCE_ENERGY
    # Simulated path only: slice of source text carried by this frame.
    payload: str = ""
    # Real path only: little-endian int16 mono PCM for this frame.
    pcm: bytes = b""

    @property
    def is_speech(self) -> bool:
        return self.energy > 0.1


def text_to_frames(
    text: str,
    *,
    leading_silence_frames: int = 3,
    trailing_silence_frames: int = 3,
) -> list[AudioFrame]:
    """Encode text as pseudo-audio frames (simulated capture/synthesis).

    Deterministic: the same text always produces the same frames.
    """
    frames: list[AudioFrame] = []
    idx = 0
    for _ in range(leading_silence_frames):
        frames.append(AudioFrame(index=idx))
        idx += 1
    chunks = [text[i : i + CHARS_PER_FRAME] for i in range(0, len(text), CHARS_PER_FRAME)]
    for n, chunk in enumerate(chunks):
        # Deterministic pseudo-prosody: energy wobbles slightly by position.
        energy = 0.7 + (n % 5) * 0.05
        frames.append(AudioFrame(index=idx, energy=energy, payload=chunk))
        idx += 1
    for _ in range(trailing_silence_frames):
        frames.append(AudioFrame(index=idx))
        idx += 1
    return frames


def frames_duration_ms(frames: list[AudioFrame]) -> int:
    return sum(f.duration_ms for f in frames)


@dataclass
class Segment:
    """A contiguous run of speech frames produced by VAD."""

    start_frame: int
    end_frame: int  # exclusive
    frames: list[AudioFrame] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return frames_duration_ms(self.frames)
