"""Voice activity detection / segmentation.

The energy VAD works on both paths: simulated frames carry an energy value
directly; real PCM frames get their energy from sample RMS at capture time.
It drops leading/trailing silence and bridges short intra-utterance gaps.
"""

from __future__ import annotations

from typing import Protocol

from .frames import AudioFrame, Segment

ENERGY_THRESHOLD = 0.1
MAX_GAP_FRAMES = 5  # bridge pauses shorter than 100 ms inside an utterance


class Vad(Protocol):
    name: str
    simulated: bool

    def segment(self, frames: list[AudioFrame]) -> list[Segment]:
        ...


class EnergyVad:
    """Deterministic energy-threshold VAD."""

    name = "energy-vad"
    simulated = False  # the algorithm itself is real; it is path-agnostic

    def segment(self, frames: list[AudioFrame]) -> list[Segment]:
        segments: list[Segment] = []
        current: list[AudioFrame] = []
        gap_frames: list[AudioFrame] = []
        gap = 0
        for frame in frames:
            if frame.energy > ENERGY_THRESHOLD:
                if gap and current:
                    current.extend(gap_frames)
                gap_frames = []
                gap = 0
                current.append(frame)
            elif current:
                gap += 1
                if gap > MAX_GAP_FRAMES:
                    segments.append(self._make_segment(current))
                    current = []
                    gap = 0
                    gap_frames = []
                else:
                    gap_frames.append(frame)
        if current:
            segments.append(self._make_segment(current))
        return segments

    @staticmethod
    def _make_segment(frames: list[AudioFrame]) -> Segment:
        return Segment(
            start_frame=frames[0].index,
            end_frame=frames[-1].index + 1,
            frames=list(frames),
        )
