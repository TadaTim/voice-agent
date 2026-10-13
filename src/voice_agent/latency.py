"""Per-stage latency instrumentation.

Every pipeline stage records samples here during real runs. Each sample is
tagged with its *basis*:

- ``measured``  -- wall-clock time of code that actually executed.
- ``modeled``   -- a deterministic model used by the *simulated* audio
  engines (they have no real signal processing to time). Modeled values are
  derived from the pseudo-audio duration so latency accounting, budgets and
  p50/p95 reporting still execute end-to-end, and they are always labeled
  as modeled -- never presented as measurements of real speech engines.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not sorted_values:
        return 0.0
    k = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return sorted_values[k - 1]


@dataclass
class LatencyRecorder:
    """Collects per-stage latency samples and reports p50/p95."""

    _samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _basis: dict[str, str] = field(default_factory=dict)

    def record(self, stage: str, ms: float, *, basis: str = "measured") -> None:
        if basis not in ("measured", "modeled"):
            raise ValueError(f"unknown latency basis: {basis}")
        self._samples[stage].append(float(ms))
        # A stage's basis is whatever its engine declares; simulated stages
        # must never silently upgrade themselves to "measured".
        self._basis[stage] = basis

    def measure(self, stage: str):
        """Context manager timing a block as a *measured* sample."""
        recorder = self

        class _Timer:
            def __enter__(self):
                self._t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                recorder.record(stage, (time.perf_counter() - self._t0) * 1000.0)
                return False

        return _Timer()

    def summary(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for stage, values in sorted(self._samples.items()):
            s = sorted(values)
            out[stage] = {
                "count": len(s),
                "p50_ms": round(_percentile(s, 50), 3),
                "p95_ms": round(_percentile(s, 95), 3),
                "max_ms": round(s[-1], 3),
                "basis": self._basis.get(stage, "measured"),
            }
        return out

    def reset(self) -> None:
        self._samples.clear()
        self._basis.clear()
