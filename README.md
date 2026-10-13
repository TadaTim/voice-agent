# Local-First Voice Agent with Governed Memory

A clinic appointment voice agent designed to run entirely on local /
private infrastructure: no cloud services, no API keys, no external
network calls at runtime. Callers book appointments by voice; the agent
remembers them only with consent, lets them inspect what is remembered,
and can delete it with a *verifiable proof* that nothing remains.

Everything below the "Honesty table" line is careful to distinguish what
actually executed from what is simulated — this project treats that
distinction as a feature, not a footnote.

## Architecture

```
 caller audio
     │
     ▼
 ┌────────┐   ┌────────┐   ┌───────────────┐   ┌─────────┐   ┌────────┐
 │  VAD / │──▶│  STT   │──▶│   Dialogue    │──▶│  Tools  │──▶│  TTS   │──▶ audio out
 │segment │   │        │   │   manager     │   │(booking)│   │        │
 └────────┘   └────────┘   └──────┬────────┘   └─────────┘   └────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
 ┌─────────────┐          ┌──────────────┐          ┌────────────────┐
 │ Governed    │          │ Guardrails + │          │ Transcript     │
 │ memory store│          │ escalation   │          │ store + index  │
 └─────────────┘          └──────────────┘          └────────────────┘
```

- **Pipeline stages** (`src/voice_agent/pipeline.py`): audio-in → VAD →
  STT → dialogue → tool calls → TTS → audio-out. Every stage sits behind
  an interface, and every stage's latency is recorded (p50/p95 reported
  from real eval runs; see below).
- **Dialogue** (`dialogue/`): a deterministic state machine — there is
  **no LLM anywhere in this system**. Language coverage is deliberately
  narrower than an LLM's, in exchange for behavior that is exactly
  reproducible and auditable.
- **Tools** (`tools/booking.py`): a local SQLite booking service. It is
  the *only* source of availability, prices and confirmation details.

### Grounding discipline (hard gate)

The agent may state availability, prices, service lists and confirmation
details **only from tool results** (logged per call). A question the
tools cannot answer — e.g. the price of a service the clinic doesn't
offer — escalates to a human instead of being invented. The eval harness
independently audits every agent utterance: any stated date/time or
price that traces neither to a tool result nor to the caller's own words
is a **grounding violation**, and the eval fails if the count is not
exactly zero.

Critical slots (name, phone, date/time) go through a mandatory
**read-back confirmation** before anything is booked. If recognition
garbles a slot, the read-back is where the caller catches and corrects
it — the eval corpus contains mishearing scenarios for exactly this.

### Barge-in

Agent speech is a frame sequence (20 ms frames). Playback consumes
frames one at a time; a caller interruption halts playback, the unplayed
remainder is recorded as *discarded*, and the interrupting audio is
processed as the next turn. The interrupted utterance never advances the
dialogue state — the correction the caller barged in with is what the
dialogue acts on. Demonstrated in `tests/test_pipeline_barge_in.py` and
eval call `15-barge-in-correction`.

### Governed memory

Per-caller memory (`memory/store.py`) enforces its rules in the store
itself, not by convention:

- **Consent gate** — nothing is persisted without a consent record.
  Facts observed before consent live only as session state.
- **Provisional until read-back** — caller-asserted facts are stored as
  `provisional`; only an explicit read-back confirmation promotes them
  to `confirmed`.
- **No silent overwrite (poisoning defense)** — asserting a different
  value for a confirmed fact flags a *conflict*. The confirmed value
  stays in force; the caller is asked explicitly whether to update it,
  and an accepted change is written to an audit history. Facts asserted
  about *other people* are refused outright.
- **Inspection** — "what do you remember about me?" lists the caller's
  own facts (confirmed, plus provisional marked as unconfirmed).
- **Provable deletion** — deletion removes memory facts, conflicts,
  history, the consent record, transcripts and transcript-index entries,
  and anonymizes appointment records. It then *verifies itself*: an
  independent scan re-reads every store for any residual trace of the
  caller (by caller id, name, or phone digits) and emits a **deletion
  receipt** that is `verified` only when every count is zero. After a
  caller deletes, the pipeline also stops recording that call entirely,
  so the confirmation utterance itself can't resurrect their data.

### Safety & escalation

Deterministic guardrails run on every caller turn before dialogue logic
(`dialogue/guardrails.py`):

- Requests for **another person's** information → refused. (Hard gate:
  privacy leakage = 0 across the eval corpus, audited against every
  other caller's stored name/phone/appointment/confirmation code.)
- **Medical-advice** requests → escalated to the human queue.
- **Distress / emergency** language → escalated, high priority.
- **Social engineering** (authority claims + data demands) → refused
  *and* escalated as suspected fraud.

Escalations carry the call's full transcript (`escalation.py`).

## Honesty table — what actually ran

| Component | Engine | Status |
|---|---|---|
| STT (eval/test path) | `SimulatedStt` — deterministic frame decoder | **Simulated.** Not speech recognition; exists so the full pipeline executes. Mishearing profiles are *injected* substitutions declared in the corpus. |
| STT (real path) | faster-whisper `tiny`, CPU/int8 | **Ran for real.** Verified round-trip on Piper-synthesized speech (see below). |
| TTS (eval/test path) | `SimulatedTts` — text→pseudo-audio frames | **Simulated.** Produces no sound; provides the frame stream playback/barge-in operate on. |
| TTS (real path) | Piper `en_US-lessac-medium` | **Ran for real.** Real WAV audio synthesized locally. |
| VAD | Energy-threshold segmentation | **Real algorithm**, path-agnostic (energy on simulated frames, RMS on PCM frames). |
| Dialogue / tools / memory / guardrails | This repo | **Real**, deterministic; no LLM. |
| Latency figures for simulated stages | RTF model (STT 0.20×, TTS 0.25× audio duration) | **Modeled**, always labeled `basis: modeled`. Measured stages say `measured`. |

Real-audio smoke test (`scripts/real_audio_smoke.py`, this machine):

```
SAID : Hello, I would like to book a checkup appointment on Tuesday morning.
HEARD: 'Hello! I would like to book a check up appointment on Tuesday morning.'
       (tts 796 ms synth / 3750 ms audio; stt 1871 ms; both measured)

SAID : My phone number is 406-555-0101.
HEARD: 'My phone number is 46555-0101.'
       (tts 701 ms synth / 4864 ms audio; stt 2059 ms; both measured)
```

Note the second line: the **real** recognizer genuinely misheard the
phone number — exactly the failure class the read-back protocol and the
corpus's simulated mishearings exist to defend against.

## Eval scorecard

16 labeled scripted calls (`eval/corpus/`): happy-path booking, two
mishearing corrections, unavailable-slot grounding, price known/unknown,
returning-caller memory, inspection + deletion, two poisoning attempts,
privacy extraction, social engineering, medical advice, distress,
barge-in, consent declined.

Latest run (`python eval/run_eval.py`):

```
calls: 16/16 passed          task success rate: 1.0
slot accuracy: 32/32 = 1.0   escalation correctness: 4/4 = 1.0
HARD GATES: grounding violations = 0, privacy leakage = 0

stage       p50       p95      basis     n
vad         0.007 ms  0.011 ms measured  59
stt         36.0  ms  84.0  ms modeled   59
dialogue    0.198 ms  0.851 ms measured  59
tool        0.132 ms  0.430 ms measured  22
tts         130.0 ms  310.0 ms modeled   93
turn_total  0.632 ms  3.168 ms measured  59
```

Latency reading guide: `turn_total` is the wall-clock cost of the
pipeline's own computation per caller turn (excluding modeled audio
stage time) — i.e. the dialogue system adds well under 1 ms median on
top of whatever real engines cost. (Measured rows vary slightly with
machine load; the modeled rows are deterministic.) The real engines' measured costs are in the
smoke test above (Whisper-tiny ≈ 1–2 s per utterance on this CPU; that,
not the dialogue layer, is where a production deployment would optimize
first, e.g. a larger/faster model or streaming decode).

## Run it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"

./.venv/bin/python -m pytest -q          # 45 tests (2 real-audio tests skip without models)
./.venv/bin/python eval/run_eval.py      # eval corpus; non-zero exit on hard-gate failure
./.venv/bin/python scripts/demo_call.py  # scripted demo: booking, barge-in, deletion receipt

# Optional real audio:
./.venv/bin/pip install -e ".[real-audio]"
./.venv/bin/python -m piper.download_voices --download-dir models/piper en_US-lessac-medium
# Whisper model: 'tiny' auto-downloads on first use, or place a local
# faster-whisper model dir at models/whisper-tiny
./.venv/bin/python scripts/real_audio_smoke.py models/whisper-tiny
```

CI (`.github/workflows/ci.yml`) runs the test suite and the eval corpus
on every push/PR — stdlib-only core means no model downloads in CI.

## Layout

```
src/voice_agent/
  pipeline.py          stage orchestration, playback, barge-in, latency hooks
  latency.py           per-stage p50/p95 recorder (measured vs modeled basis)
  audio/               frames, VAD, STT/TTS interfaces + simulated & real engines, factory
  dialogue/            state machine, slot parsing, guardrails, session state
  tools/booking.py     SQLite booking service (the only source of facts)
  memory/              governed store, transcripts + token index, provable deletion
  escalation.py        human queue with transcripts attached
  eval_harness.py      corpus runner + grounding/privacy auditors
eval/corpus/*.json     16 labeled scripted calls
scripts/               demo_call.py, real_audio_smoke.py
tests/                 unit + integration tests (45)
```

## Known limitations

- Dialogue language coverage is rule-based and intentionally narrow;
  phrasings far outside the corpus patterns may fall back to
  clarification prompts or escalation rather than being understood.
  Escalation, not guessing, is the designed failure mode.
- Simulated-stage latencies are a deterministic model, not measurements
  of real engines; the real engines were verified functionally (above)
  but the full eval corpus runs on the simulated path for determinism.
- The real STT adapter resamples with linear interpolation (a stand-in
  for a DSP resampler) and decodes per-utterance, not streaming.
- Appointment deletion anonymizes rather than erasing the slot: the
  clinic still needs to know a patient is coming, but the record keeps
  no name, phone, or caller link — the zero-residual scan enforces it.
