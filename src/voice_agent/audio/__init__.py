"""Audio layer: frames, VAD, STT and TTS behind small interfaces.

Two engine families implement the same interfaces:

- Simulated engines (always available, stdlib-only): deterministic
  text <-> pseudo-audio conversion. They execute the full pipeline --
  segmentation, transcription, synthesis, barge-in, latency accounting --
  but they do NOT perform real speech recognition or synthesis, and are
  labeled as simulated everywhere.
- Real local engines (optional deps): faster-whisper for STT and Piper
  for TTS. Used only when installed with models present.
"""
