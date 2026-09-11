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

Usage:
    python -m scripts.calibrate_baselines --reps 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, List


def calibrate(reps: int = 5, mode_target: float = 0.8) -> Dict[str, float]:
    from benchmarks.cpu import registry
    from benchmarks.cpu.common import run_timed_subtest
    from benchmarks.cpu.single_core import elevated_priority, pinned_to_core
    from monitoring.environment import describe_fingerprint, get_environment_fingerprint
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
    print(f"Fingerprint: {fingerprint['fingerprint_hash']}\n")

    measurements: Dict[str, List[float]] = {w.category: [] for w in registry.WORKLOADS}

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
                        else:
                            print(f"    ! {spec.category} failed validation: "
                                  f"{result.error_message}")

    baselines: Dict[str, float] = {}
    print("\n" + "-" * 66)
    for spec in registry.WORKLOADS:
        values = measurements[spec.category]
        if not values:
            print(f"  {spec.category:<16} NO VALID MEASUREMENTS")
            continue
        median = statistics.median(values)
        spread = (statistics.stdev(values) / median * 100) if len(values) > 1 else 0.0
        baselines[spec.category] = round(median, 2)
        print(f"  {spec.category:<16} {median:>12,.2f} {spec.raw_metric_name:<10} "
              f"spread {spread:>5.2f}%")

    print("-" * 66)
    print("\nPaste into benchmarks/cpu/common.py:\n")
    print("CATEGORY_BASELINES: Dict[str, float] = {")
    for spec in registry.WORKLOADS:
        if spec.category in baselines:
            print(f'    "{spec.category}": {baselines[spec.category]},'
                  f'{"":<4}# {spec.raw_metric_name}')
    print("}")

    print("\nAlso record in docs/BENCHMARK_SPEC.md:")
    print(json.dumps({
        "fingerprint_hash": fingerprint["fingerprint_hash"],
        "cpu": fingerprint["os"]["processor"] or fingerprint["os"]["machine"],
        "python": fingerprint["python"]["version"],
        "numpy": fingerprint["numpy"].get("version"),
        "blas": [f"{b.get('internal_api')} {b.get('version')}"
                 for b in fingerprint.get("blas_backends", [])],
    }, indent=2))

    return baselines


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalibrate BenchMind baselines.")
    parser.add_argument("--reps", type=int, default=5,
                        help="Passes over every workload. More is better.")
    args = parser.parse_args()
    calibrate(reps=args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
