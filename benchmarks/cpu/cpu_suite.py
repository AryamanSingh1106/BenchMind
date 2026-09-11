"""
benchmarks/cpu/cpu_suite.py

Master CPU benchmark orchestrator.

Clock separation rule (unchanged from 1.x, still correct):
  time.perf_counter()  -> ALL benchmark performance timing
  time.monotonic()     -> telemetry timestamps and telemetry window filtering

New in 2.0:
  * every result carries an environment fingerprint, so two runs are only
    compared when the software stack matches
  * every score carries a 95% confidence interval
  * the composite index excludes runtime-bound categories
  * telemetry windows feed thermal throttle analysis
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

from benchmarks.cpu import registry
from benchmarks.cpu.common import (
    BASELINE_VERSION,
    REFERENCE_MACHINE,
    SubtestResult,
    calculate_cpu_index,
    combine_ci_geometric,
    combined_index_ci,
    geometric_mean,
)
from benchmarks.cpu.multi_core import run_multi_core_suite
from benchmarks.cpu.single_core import run_single_core_suite
from monitoring.environment import get_environment_fingerprint
from monitoring.telemetry_service import TelemetryService

logger = logging.getLogger("BenchMind.CPU.Suite")

MODE_TARGETS = {
    "quick": 0.30,
    "standard": 0.80,
    "full": 2.00,
}
MODE_SCALE = {
    "quick": 0.5,
    "standard": 1.0,
    "full": 1.0,
}
MODE_REPS = {
    "quick": 3,
    "standard": 5,
    "full": 9,
}

ProgressFn = Optional[Callable[[str, float], None]]


def _emit(progress: ProgressFn, stage: str, fraction: float) -> None:
    if progress is not None:
        try:
            progress(stage, fraction)
        except Exception:  # noqa: BLE001
            logger.debug("Progress callback raised; ignoring.", exc_info=True)


def run_full_cpu_suite(
    mode: str = "standard",
    single_weight: float = 0.4,
    multi_weight: float = 0.6,
    include_runtime_bound: bool = True,
    pin_core: bool = True,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    """
    Run the complete CPU benchmark.

    Modes:
      quick    - reduced scale, 3 reps. Roughly 20-40 s. Use for smoke tests.
      standard - full scale, 5 reps. The default and the only mode whose scores
                 should be stored or compared.
      full     - full scale, 9 reps. Tightest confidence intervals.

    Scores from different modes are NOT comparable; `mode` is recorded in the
    result and the history layer refuses to compare across modes.
    """
    if mode not in MODE_TARGETS:
        raise ValueError(f"Unknown mode {mode!r}. Choose from {sorted(MODE_TARGETS)}.")

    target_duration = MODE_TARGETS[mode]
    scale = MODE_SCALE[mode]
    reps = MODE_REPS[mode]

    telemetry_service = TelemetryService.get_instance()
    telemetry_service.start()

    start_perf = time.perf_counter()
    start_mono = time.monotonic()
    logger.info("Starting BenchMind CPU Suite, mode=%s scale=%.2f reps=%d", mode, scale, reps)

    _emit(progress, "single_core", 0.05)
    single_results, single_score, single_time, single_meta = run_single_core_suite(
        target_duration_per_subtest=target_duration,
        scale=scale,
        pin_core=pin_core,
        include_runtime_bound=include_runtime_bound,
    )

    _emit(progress, "multi_core", 0.50)
    multi_results, multi_score, multi_time, cores_used, multi_meta = run_multi_core_suite(
        scale=scale,
        reps=reps,
        include_runtime_bound=include_runtime_bound,
    )

    end_perf = time.perf_counter()
    end_mono = time.monotonic()
    _emit(progress, "analysis", 0.90)

    telemetry_logs = telemetry_service.get_logs_format(
        start_time=start_mono, end_time=end_mono, use_monotonic=True
    )

    single_ci = single_meta.get("composite_ci_pct", 0.0)
    multi_ci = multi_meta.get("composite_ci_pct", 0.0)

    cpu_index = calculate_cpu_index(single_score, multi_score, single_weight, multi_weight)
    index_ci = combined_index_ci(single_ci, multi_ci, single_score, multi_score,
                                 single_weight, multi_weight)

    all_subtests: List[SubtestResult] = single_results + multi_results
    category_scores = _category_breakdown(single_results, multi_results)

    return {
        "schema_version": 2,
        "mode": mode,
        "scale": scale,
        "baseline_version": BASELINE_VERSION,
        "reference_machine": REFERENCE_MACHINE,

        "cpu_index": cpu_index,
        "cpu_index_ci_pct": index_ci,
        "single_core_score": single_score,
        "single_core_ci_pct": single_ci,
        "multi_core_score": multi_score,
        "multi_core_ci_pct": multi_ci,

        "single_core_time": round(single_time, 4),
        "multi_core_time": round(multi_time, 4),
        "total_suite_time": round(end_perf - start_perf, 4),
        "cores_used": cores_used,
        "multi_core_scaling": (
            round(multi_score / single_score, 3) if single_score > 0 else 0.0
        ),

        "scoring_weights": {
            "single_core_weight": single_weight,
            "multi_core_weight": multi_weight,
        },
        "isolation": single_meta,
        "multi_core_meta": multi_meta,

        "category_scores": category_scores,
        "subtests": [s.to_dict() for s in all_subtests],

        "environment": get_environment_fingerprint(),
        "telemetry": telemetry_logs,
        "telemetry_summary": summarize_telemetry(telemetry_logs),
        "telemetry_window": {"start_monotonic": start_mono, "end_monotonic": end_mono},
    }


def _category_breakdown(single_results: List[SubtestResult],
                        multi_results: List[SubtestResult]) -> Dict[str, Any]:
    """
    Per-category scores, reported separately for single and multi core.

    1.x computed `category_scores` from single-core results only, while still
    calling it the category breakdown of the whole suite. Multi-core subtests
    silently never contributed. Both are now reported explicitly.
    """
    breakdown: Dict[str, Any] = {}
    for spec in registry.WORKLOADS:
        cat = spec.category
        s = [r for r in single_results if r.category == cat and r.validation_passed]
        m = [r for r in multi_results if r.category == cat and r.validation_passed]
        breakdown[cat] = {
            "display_name": spec.name,
            "profile": spec.workload_profile,
            "counted_in_index": spec.counted_in_index,
            "single_core": geometric_mean([r.score for r in s]),
            "single_core_ci_pct": combine_ci_geometric([r.score_ci_pct for r in s]),
            "multi_core": geometric_mean([r.score for r in m]),
            "multi_core_ci_pct": combine_ci_geometric([r.score_ci_pct for r in m]),
            "raw_metric_name": spec.raw_metric_name,
            "single_core_raw": s[0].raw_metric_value if s else 0.0,
            "multi_core_raw": m[0].raw_metric_value if m else 0.0,
        }
    return breakdown


def summarize_telemetry(logs: Dict[str, List]) -> Dict[str, Any]:
    cpu = [v for v in logs.get("cpu", []) if v is not None]
    ram = [v for v in logs.get("ram", []) if v is not None]
    cpu_temps = [t for t in logs.get("cpu_temp", []) if t is not None]
    gpu_temps = [t for t in logs.get("gpu_temp", []) if t is not None]
    freqs = [f for f in logs.get("cpu_freq", []) if f is not None]

    def stat(vals, fn, nd=2):
        return round(fn(vals), nd) if vals else None

    return {
        "sample_count": len(logs.get("cpu", [])),
        "avg_cpu_utilization": stat(cpu, statistics.fmean) or 0.0,
        "max_cpu_utilization": stat(cpu, max) or 0.0,
        "avg_ram_utilization": stat(ram, statistics.fmean) or 0.0,
        "max_cpu_temp": stat(cpu_temps, max),
        "min_cpu_temp": stat(cpu_temps, min),
        "avg_cpu_temp": stat(cpu_temps, statistics.fmean),
        "max_gpu_temp": stat(gpu_temps, max),
        "avg_cpu_freq_mhz": stat(freqs, statistics.fmean),
        "min_cpu_freq_mhz": stat(freqs, min),
        "max_cpu_freq_mhz": stat(freqs, max),
        "temperature_source_available": bool(cpu_temps),
    }
