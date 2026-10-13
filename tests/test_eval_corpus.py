from __future__ import annotations

from pathlib import Path

from voice_agent.eval_harness import run_eval, scorecard_passed

CORPUS = Path(__file__).resolve().parent.parent / "eval" / "corpus"


def test_eval_corpus_all_calls_pass_and_hard_gates_hold():
    scorecard = run_eval(CORPUS)
    assert scorecard["hard_gates"] == {
        "grounding_violations": 0,
        "privacy_leakage": 0,
    }
    assert scorecard["calls_passed"] == scorecard["calls_total"] == 16
    assert scorecard["slot_accuracy"]["rate"] == 1.0
    assert scorecard_passed(scorecard)
    failures = [
        (r["id"], r["failures"]) for r in scorecard["results"] if not r["passed"]
    ]
    assert failures == []
