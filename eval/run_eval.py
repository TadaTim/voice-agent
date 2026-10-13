#!/usr/bin/env python3
"""Run the labeled eval corpus end to end.

Usage:
    python eval/run_eval.py [--corpus DIR] [--out SCORECARD.json]

Exit code is non-zero if any hard gate fails (grounding violations > 0
or privacy leakage > 0) or if any scripted call fails its expectations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_agent.eval_harness import run_eval, scorecard_passed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parent / "corpus"),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    scorecard = run_eval(args.corpus)
    for result in scorecard["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']} ({result['category']}) -> {result['outcome']}")
        for failure in result["failures"]:
            print(f"       - {failure}")
    print()
    print(f"calls: {scorecard['calls_passed']}/{scorecard['calls_total']} passed")
    print(f"task success rate: {scorecard['task_success_rate']}")
    slots = scorecard["slot_accuracy"]
    print(f"slot accuracy: {slots['correct']}/{slots['total']} = {slots['rate']}")
    esc = scorecard["escalations"]
    print(f"escalation correctness: {esc['correct']}/{esc['expected']} = {esc['rate']}")
    gates = scorecard["hard_gates"]
    print(
        f"HARD GATES: grounding violations = {gates['grounding_violations']} "
        f"(must be 0), privacy leakage = {gates['privacy_leakage']} (must be 0)"
    )
    print("latency (per stage, from these runs):")
    for stage, stats in scorecard["latency"].items():
        print(
            f"  {stage:<10} p50 {stats['p50_ms']:>9.3f} ms  "
            f"p95 {stats['p95_ms']:>9.3f} ms  ({stats['basis']}, n={stats['count']})"
        )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scorecard, indent=2, default=str))
        print(f"\nscorecard written to {args.out}")
    ok = scorecard_passed(scorecard)
    print("\nEVAL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
