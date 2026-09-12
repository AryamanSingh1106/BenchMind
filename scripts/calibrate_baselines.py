"""
scripts/calibrate_baselines.py

Recalibrate the scoring baselines against a reference machine.

A score of 1000 in a category means "matched the reference machine on that
workload". That is only true while CATEGORY_BASELINES holds the reference
machine's actual measured metrics. If you change a workload's parameters, the
old baselines become meaningless and every stored score becomes incomparable.

Procedure:
  1. Pick one machine and keep it. Document it.
  2. Get it into a clean state: mains power, nothing else running, cool.
  3. Run this script with several repetitions.
  4. Paste the emitted block into benchmarks/cpu/common.py.
  5. Bump BASELINE_VERSION, update docs/BENCHMARK_SPEC.md and CHANGELOG.md in
     the SAME commit.

Do not tweak individual baselines to make a score look better. That is the
exact failure mode the project's own philosophy section warns about.

Spread gate (added in 2.0.2)
----------------------------
The script now REFUSES to emit a baseline block when any category's spread
across passes exceeds --max-spread (default 5%).

This is not pedantry. A baseline captured at 37% spread is one sample from a
very wide distribution, and every future score in that category is then
measured against a number that could just as easily have been 30% different.
The first real calibration run on an i5-13450HX produced exactly that:

    integer          2,019.90 Mops/sec   spread 36.92%
    branch_heavy        29.20 Mops/sec   spread 10.74%
    matrix              45.86 GFLOPS     spread  1.18%

Adopting those numbers would have permanently miscalibrated the integer
category. The cause was pinning to logical core 0, fixed in 2.0.2 (see
monitoring/topology.py) -- but the gate stays, because the next cause of noisy
calibration will be something else.

Usage:
    python -m scripts.calibrate_baselines --reps 5
    python -m scripts.calibrate_baselines --reps 9 --max-spread 3
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, List


def evaluate_spreads(spreads: Dict[str, float], missing: List[str],
                     max_spread: float = 5.0) -> Dict[str, object]:
    """
    Decide whether a set of per-category spreads is fit to become baselines.

    Pure function so the policy can be tested without running a benchmark.
    Returns `acceptable` plus the offending categories, worst first.
    """
    noisy = {k: v for k, v in spreads.items() if v > max_spread}
    return {
        "acceptable": not noisy and not missing,
        "noisy": dict(sorted(noisy.items(), key=lambda kv: -kv[1])),
        "missing": list(missing),
        "worst_spread_pct": round(max(spreads.values()), 2) if spreads else None,
        "max_spread": max_spread,
    }


def calibrate(reps: int = 5, mode_target: float = 1.5,
              max_spread: float = 5.0, force: bool = False) -> Dict[str, float]:
    from benchmarks.cpu import registry
    from benchmarks.cpu.common import run_timed_subtest
    from benchmarks.cpu.single_core import elevated_priority, pinned_to_core
    from monitoring.environment import describe_fingerprint, get_environment_fingerprint
    from monitoring.topology import get_topology, select_benchmark_core
    from monitoring.validity import check_run_validity

    import threadpoolctl

    gate = check_run_validity(sample_seconds=3.0)
    print(f"Conditions: {gate.verdict}")
    for issue in gate.issues:
        print(f"  ! {issue.message}")
    if gate.verdict != "valid":
        print("\nRefusing to calibrate under these conditions. Baselines captured on a "
              "busy or battery-powered machine will misprice every future score.")
        raise SystemExit(1)

    fingerprint = get_environment_fingerprint()
    print(f"\nReference environment: {describe_fingerprint(fingerprint)}")
    print(f"Fingerprint: {fingerprint['fingerprint_hash']}")

    topology = get_topology()
    selection = select_benchmark_core(topology)
    print(f"Topology: {topology.describe()}")
    print(f"Pinning to logical core {selection['logical_id']} ({selection['reason']})\n")

    measurements: Dict[str, List[float]] = {w.category: [] for w in registry.WORKLOADS}
    short_reps: set = set()

    with pinned_to_core():
        with elevated_priority():
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                for rep in range(1, reps + 1):
                    print(f"  pass {rep}/{reps}")
                    for spec in registry.WORKLOADS:
                        result = run_timed_subtest(spec, scale=1.0,
                                                   target_duration=mode_target)
                        if result.validation_passed:
                            measurements[spec.category].append(result.raw_metric_value)
                            if result.short_rep_warning:
                                short_reps.add(spec.category)
                        else:
                            print(f"    ! {spec.category} failed validation: "
                                  f"{result.error_message}")

    baselines: Dict[str, float] = {}
    spreads: Dict[str, float] = {}
    missing: List[str] = []

    print("\n" + "-" * 70)
    for spec in registry.WORKLOADS:
        values = measurements[spec.category]
        if not values:
            print(f"  {spec.category:<16} NO VALID MEASUREMENTS")
            missing.append(spec.category)
            continue
        median = statistics.median(values)
        spread = (statistics.stdev(values) / median * 100) if len(values) > 1 else 0.0
        baselines[spec.category] = round(median, 2)
        spreads[spec.category] = spread
        flag = "  <-- TOO NOISY" if spread > max_spread else ""
        if spec.category in short_reps:
            flag += "  [repetitions too short to be stable]"
        print(f"  {spec.category:<16} {median:>12,.2f} {spec.raw_metric_name:<10} "
              f"spread {spread:>6.2f}%{flag}")
    print("-" * 70)

    gate = evaluate_spreads(spreads, missing, max_spread)
    noisy = gate["noisy"]
    if not gate["acceptable"] and not force:
        print(f"\nREFUSING to emit baselines: {len(noisy)} categor"
              f"{'y' if len(noisy) == 1 else 'ies'} exceeded the {max_spread:.0f}% "
              "spread limit.")
        for category, spread in noisy.items():
            print(f"    {category}: {spread:.2f}%")
        if missing:
            print(f"    no valid measurements: {', '.join(missing)}")

        print("\nA baseline captured from noisy data permanently miscalibrates that")
        print("category: every future score is measured against a number that could")
        print("just as easily have been far different. Fix the noise first.")
        print("\nWhat to check, roughly in order of likelihood:")
        print("  - background processes: close browsers, chat apps, sync clients")
        print("  - mains power, and a performance power plan rather than balanced")
        print("  - let the machine idle 10 minutes so it starts cold")
        print("  - raise --reps (9 averages more of whatever interference remains)")
        print("  - check `python run.py bench` reports a sensible pinned core; "
              "pinning to core 0 on Windows is a known source of large spread")
        print("\nRe-run when those are addressed. Override with --force only if you")
        print("understand exactly why the spread is irreducible on this machine.")
        return {}

    if noisy and force:
        print(f"\nWARNING: --force used with {len(noisy)} noisy categor"
              f"{'y' if len(noisy) == 1 else 'ies'}. These baselines are not "
              "trustworthy and should not be published.")

    print("\nPaste into benchmarks/cpu/common.py:\n")
    print("CATEGORY_BASELINES: Dict[str, float] = {")
    for spec in registry.WORKLOADS:
        if spec.category in baselines:
            print(f'    "{spec.category}": {baselines[spec.category]},'
                  f'{"":<4}# {spec.raw_metric_name}  '
                  f'(spread {spreads[spec.category]:.2f}% over {reps} passes)')
    print("}")

    print("\nAlso record in docs/BENCHMARK_SPEC.md:")
    print(json.dumps({
        "fingerprint_hash": fingerprint["fingerprint_hash"],
        "cpu": fingerprint["os"]["processor"] or fingerprint["os"]["machine"],
        "python": fingerprint["python"]["version"],
        "numpy": fingerprint["numpy"].get("version"),
        "blas": [f"{b.get('internal_api')} {b.get('version')}"
                 for b in fingerprint.get("blas_backends", [])],
        "topology": topology.describe(),
        "pinned_logical_core": selection["logical_id"],
        "passes": reps,
        "worst_spread_pct": round(max(spreads.values()), 2) if spreads else None,
    }, indent=2))

    print("\nThen bump BASELINE_VERSION in benchmarks/cpu/common.py and add a")
    print("CHANGELOG entry, in the SAME commit. Scores across baseline versions")
    print("are not comparable.")

    return baselines


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalibrate BenchMind baselines.")
    parser.add_argument("--reps", type=int, default=5,
                        help="Passes over every workload. More is better.")
    parser.add_argument("--max-spread", type=float, default=5.0,
                        help="Refuse to emit baselines above this spread percentage.")
    parser.add_argument("--force", action="store_true",
                        help="Emit baselines even when too noisy. Not recommended.")
    args = parser.parse_args()
    result = calibrate(reps=args.reps, max_spread=args.max_spread, force=args.force)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
