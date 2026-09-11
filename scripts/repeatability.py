"""
scripts/repeatability.py

Repeatability harness. Replaces the old scratch_3runs.py.

This is the single most important test in the project. A benchmark whose score
moves 12% between back-to-back runs cannot tell you whether machine A is faster
than machine B by 8%. Before trusting any score, or before declaring a
benchmark change complete, run this.

Cooldown between runs is deliberate and defaults to 30 seconds: without it, run
2 starts on a hot chip and the "repeatability" you measure is really thermal
carry-over.

Usage:
    python -m scripts.repeatability --runs 5 --mode standard
    python run.py repeat --runs 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict, List


def run_repeatability(runs: int = 3, mode: str = "standard",
                      cooldown: float = 30.0) -> Dict[str, Any]:
    from ai.stability_engine import repeatability_from_scores
    from api.main import BenchmarkRequest, _execute_benchmark
    from monitoring.validity import check_run_validity

    print(f"Repeatability check: {runs} runs in '{mode}' mode, "
          f"{cooldown:.0f}s cooldown between them.\n")

    gate = check_run_validity(sample_seconds=3.0)
    if gate.verdict != "valid":
        print(f"! Conditions are '{gate.verdict}' before the first run:")
        for issue in gate.issues:
            print(f"    {issue.message}")
        print("  Results below will be noisier than they should be.\n")

    indices: List[float] = []
    singles: List[float] = []
    multis: List[float] = []
    peaks: List[float] = []
    rows: List[Dict[str, Any]] = []

    req = BenchmarkRequest(mode=mode, include_gpu=False, include_scaling=False,
                           save_to_history=False)

    for i in range(1, runs + 1):
        if i > 1 and cooldown > 0:
            print(f"  cooling down {cooldown:.0f}s ...", end="", flush=True)
            time.sleep(cooldown)
            print(" done")

        print(f"  run {i}/{runs} ...", end="", flush=True)
        started = time.time()
        result = _execute_benchmark(req)
        cpu = result["cpu_benchmark"]
        telemetry = cpu.get("telemetry_summary", {}) or {}

        indices.append(cpu["cpu_index"])
        singles.append(cpu["single_core_score"])
        multis.append(cpu["multi_core_score"])
        if telemetry.get("max_cpu_temp"):
            peaks.append(telemetry["max_cpu_temp"])

        rows.append({
            "run": i,
            "cpu_index": cpu["cpu_index"],
            "ci_pct": cpu["cpu_index_ci_pct"],
            "single": cpu["single_core_score"],
            "multi": cpu["multi_core_score"],
            "peak_temp": telemetry.get("max_cpu_temp"),
            "throttled": (result.get("throttling") or {}).get("throttling_detected"),
            "seconds": round(time.time() - started, 1),
        })
        print(f" index {cpu['cpu_index']:,.0f} "
              f"(+/- {cpu['cpu_index_ci_pct']:.1f}% within-run)")

    print("\n" + "-" * 66)
    print(f"{'run':>4} {'index':>9} {'single':>9} {'multi':>9} {'peak C':>8} {'secs':>7}")
    for r in rows:
        peak = f"{r['peak_temp']:.0f}" if r["peak_temp"] else "-"
        print(f"{r['run']:>4} {r['cpu_index']:>9,.0f} {r['single']:>9,.0f} "
              f"{r['multi']:>9,.0f} {peak:>8} {r['seconds']:>7.1f}")

    report = repeatability_from_scores(indices)
    print("-" * 66)
    print(f"  mean index      {report['mean_score']:,.2f}")
    print(f"  std deviation   {report['std_dev']:,.2f}")
    print(f"  spread          {report['spread_pct']:.2f}%")
    print(f"  range           {report['min_score']:,.0f} to {report['max_score']:,.0f}")

    spread = report["spread_pct"]
    if spread < 2:
        verdict, note = "excellent", "Differences under 2% between machines are still noise."
    elif spread < 5:
        verdict, note = "good", "Usable for coarse comparisons; treat sub-5% gaps as a tie."
    elif spread < 10:
        verdict, note = "fair", "Something is interfering. Check background load and cooling."
    else:
        verdict, note = "poor", "Not trustworthy. Do not compare these scores to anything."

    print(f"\n  Verdict: {verdict.upper()} - {note}")
    if peaks:
        print(f"  Peak temperatures: {min(peaks):.0f} to {max(peaks):.0f} C")
    if any(r["throttled"] for r in rows):
        print("  Throttling was detected in at least one run; the spread partly "
              "reflects thermal state, not measurement error.")

    return {
        "runs": rows,
        "verdict": verdict,
        "spread_pct": spread,
        "mean_index": report["mean_score"],
        "single_core_spread_pct": repeatability_from_scores(singles).get("spread_pct"),
        "multi_core_spread_pct": repeatability_from_scores(multis).get("spread_pct"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BenchMind repeatability check.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--cooldown", type=float, default=30.0)
    args = parser.parse_args()
    result = run_repeatability(args.runs, args.mode, args.cooldown)
    return 0 if result["verdict"] in ("excellent", "good") else 1


if __name__ == "__main__":
    raise SystemExit(main())
