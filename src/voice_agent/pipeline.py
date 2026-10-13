"""Voice pipeline: audio-in -> VAD -> STT -> dialogue -> tools -> TTS ->
audio-out, with per-stage latency instrumentation and barge-in.

Playback model: an agent utterance is synthesized into a ``Playback``
(a frame sequence). Playing consumes frames one at a time. A caller may
barge in mid-playback: playback halts, the unplayed remainder is recorded
as discarded, and the caller's new audio is processed as the next turn --
the interrupted utterance never advances the dialogue state, because the
dialogue only moves when a caller turn is processed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

from .audio.factory import AudioStack
from .audio.frames import AudioFrame
from .dialogue.manager import DialogueManager
from .dialogue.session import DialogueSession, Services
from .latency import LatencyRecorder


class _TimedTool:
    """Proxy that times every tool call into the latency recorder."""

    def __init__(self, target, recorder: LatencyRecorder, stage: str = "tool"):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_stage", stage)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_target"), name)
        if not callable(attr):
            return attr
        recorder = object.__getattribute__(self, "_recorder")
        stage = object.__getattribute__(self, "_stage")

        def wrapper(*args, **kwargs):
            with recorder.measure(stage):
                return attr(*args, **kwargs)

        return wrapper


@dataclass
class Playback:
    """One synthesized agent utterance and its playback state."""

    utterance_id: int
    text: str
    frames: list[AudioFrame]
    played_frames: int = 0
    interrupted: bool = False

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def discarded_frames(self) -> int:
        return self.total_frames - self.played_frames if self.interrupted else 0

    def play(self, max_frames: int | None = None):
        """Yield frames as they are 'played', tracking progress."""
        remaining = self.total_frames - self.played_frames
        count = remaining if max_frames is None else min(max_frames, remaining)
        for _ in range(count):
            frame = self.frames[self.played_frames]
            self.played_frames += 1
            yield frame

    def play_all(self) -> None:
        for _ in self.play():
            pass

    def interrupt(self) -> None:
        """Barge-in: halt now; the unplayed remainder is discarded."""
        self.interrupted = True


@dataclass
class TurnOutcome:
    caller_text: str
    utterances: list[str]
    playbacks: list[Playback] = field(default_factory=list)


class VoicePipeline:
    def __init__(
        self,
        *,
        audio: AudioStack,
        services: Services,
        recorder: LatencyRecorder | None = None,
    ):
        self.audio = audio
        self.recorder = recorder or LatencyRecorder()
        timed_booking = _TimedTool(services.booking, self.recorder)
        wrapped = Services(
            booking=timed_booking,  # type: ignore[arg-type]
            memory=services.memory,
            transcripts=services.transcripts,
            escalation=services.escalation,
        )
        self.services = services
        self.dialogue = DialogueManager(wrapped)
        self._session_counter = itertools.count(1)
        self._utterance_counter = itertools.count(1)

    # -- call lifecycle --------------------------------------------------

    def start_call(
        self, caller_id: str, now: datetime, *, session_id: str | None = None
    ) -> tuple[DialogueSession, TurnOutcome]:
        session_id = session_id or f"call-{next(self._session_counter)}"
        session = DialogueSession(session_id=session_id, caller_id=caller_id, now=now)
        greetings = self.dialogue.open_call(session)
        outcome = TurnOutcome(caller_text="", utterances=greetings)
        for text in greetings:
            playback = self._synthesize(session, text)
            playback.play_all()
            outcome.playbacks.append(playback)
        return session, outcome

    # -- turns -------------------------------------------------------------

    def caller_turn(self, session: DialogueSession, frames: list[AudioFrame]) -> TurnOutcome:
        import time

        t0 = time.perf_counter()
        with self.recorder.measure("vad"):
            segments = self.audio.vad.segment(frames)
        stt_result = self.audio.stt.transcribe(segments)
        self.recorder.record("stt", stt_result.latency_ms, basis=stt_result.latency_basis)
        caller_text = stt_result.text
        # Once a caller has exercised deletion, nothing further from this
        # call is recorded -- otherwise the deletion receipt for this very
        # call would be undone by its own confirmation utterance.
        if session.deletion_receipt is None:
            self.services.transcripts.append(
                session_id=session.session_id,
                caller_id=session.caller_id,
                role="caller",
                text=caller_text,
                ts=session.now.isoformat(),
            )
        with self.recorder.measure("dialogue"):
            utterances = self.dialogue.handle(session, caller_text)
        outcome = TurnOutcome(caller_text=caller_text, utterances=utterances)
        for text in utterances:
            playback = self._synthesize(session, text)
            playback.play_all()
            outcome.playbacks.append(playback)
        self.recorder.record("turn_total", (time.perf_counter() - t0) * 1000.0)
        return outcome

    def speak(self, session: DialogueSession, text: str) -> Playback:
        """Synthesize an utterance *without* playing it (playback control
        is left to the caller, e.g. the barge-in path or a test)."""
        return self._synthesize(session, text)

    def barge_in(
        self,
        session: DialogueSession,
        playback: Playback,
        *,
        after_frames: int,
        caller_frames: list[AudioFrame],
    ) -> TurnOutcome:
        """Interrupt ``playback`` after ``after_frames`` frames have played,
        discard the rest, and process the caller's interrupting audio.

        In the scripted-call model the previous utterance is treated as
        still playing when a barge-in turn arrives, so playback position is
        (re)advanced deterministically to the interruption point first."""
        playback.played_frames = 0
        for _ in playback.play(max_frames=after_frames):
            pass
        playback.interrupt()
        session.events.append(
            {
                "type": "barge_in",
                "utterance_id": playback.utterance_id,
                "played_frames": playback.played_frames,
                "discarded_frames": playback.discarded_frames,
            }
        )
        return self.caller_turn(session, caller_frames)

    # -- internals ---------------------------------------------------------

    def _synthesize(self, session: DialogueSession, text: str) -> Playback:
        result = self.audio.tts.synthesize(text)
        self.recorder.record("tts", result.latency_ms, basis=result.latency_basis)
        if session.deletion_receipt is None:  # see caller_turn: no recording after deletion
            self.services.transcripts.append(
                session_id=session.session_id,
                caller_id=session.caller_id,
                role="agent",
                text=text,
                ts=session.now.isoformat(),
            )
        return Playback(
            utterance_id=next(self._utterance_counter),
            text=text,
            frames=result.frames,
        )
