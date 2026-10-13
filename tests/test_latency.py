from __future__ import annotations

from voice_agent.latency import LatencyRecorder


def test_percentiles_and_basis():
    rec = LatencyRecorder()
    for i in range(1, 101):
        rec.record("stt", float(i), basis="modeled")
    summary = rec.summary()
    assert summary["stt"]["count"] == 100
    assert summary["stt"]["p50_ms"] == 50.0
    assert summary["stt"]["p95_ms"] == 95.0
    assert summary["stt"]["basis"] == "modeled"


def test_measure_context_manager_records_wall_clock():
    rec = LatencyRecorder()
    with rec.measure("dialogue"):
        sum(range(10000))
    summary = rec.summary()
    assert summary["dialogue"]["count"] == 1
    assert summary["dialogue"]["basis"] == "measured"
    assert summary["dialogue"]["p50_ms"] >= 0
