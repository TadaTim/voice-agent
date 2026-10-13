"""Audio stack construction with honest capability reporting.

``mode="simulated"`` always works (stdlib only). ``mode="auto"`` tries the
real local engines and falls back to simulated ones, reporting exactly
what was selected. ``mode="real"`` raises if a real engine is unavailable
rather than silently substituting a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stt import FasterWhisperStt, SimulatedStt, Stt
from .tts import PiperTts, SimulatedTts, Tts
from .vad import EnergyVad, Vad


@dataclass
class AudioStack:
    vad: Vad
    stt: Stt
    tts: Tts

    def report(self) -> dict[str, dict]:
        return {
            "vad": {"engine": self.vad.name, "simulated": self.vad.simulated},
            "stt": {"engine": self.stt.name, "simulated": self.stt.simulated},
            "tts": {"engine": self.tts.name, "simulated": self.tts.simulated},
        }


def build_audio_stack(
    mode: str = "simulated",
    *,
    mishearings: dict[str, str] | None = None,
    whisper_model: str = "tiny",
    whisper_model_dir: str | None = None,
    piper_voice_path: str | None = None,
) -> AudioStack:
    vad = EnergyVad()
    if mode == "simulated":
        return AudioStack(
            vad=vad,
            stt=SimulatedStt(mishearings=mishearings or {}),
            tts=SimulatedTts(),
        )
    if mode not in ("auto", "real"):
        raise ValueError(f"unknown audio mode: {mode}")

    stt: Stt
    try:
        stt = FasterWhisperStt(whisper_model, whisper_model_dir)
    except Exception:
        if mode == "real":
            raise
        stt = SimulatedStt(mishearings=mishearings or {})

    tts: Tts
    try:
        if not piper_voice_path:
            raise RuntimeError("no Piper voice model configured")
        tts = PiperTts(piper_voice_path)
    except Exception:
        if mode == "real":
            raise
        tts = SimulatedTts()

    # A mishearing profile is a simulation device; a real STT engine makes
    # its own mistakes, so attaching one would be incoherent.
    if mishearings and not stt.simulated:
        raise ValueError("mishearing profiles apply only to the simulated STT")
    return AudioStack(vad=vad, stt=stt, tts=tts)
