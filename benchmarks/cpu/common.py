"""
benchmarks/cpu/common.py

BenchMind measurement core.

Design rules enforced here (see docs/BENCHMARK_SPEC.md):

1.  SETUP IS NEVER TIMED.
    Every workload is split into `setup(scale) -> ctx` and `run(ctx) -> (output, work_units)`.
    Allocation, RNG generation and payload construction happen in `setup`.
    Only `run` sits inside the perf_counter window.

2.  `time.perf_counter()` is the ONLY clock used for performance measurement.
    `time.monotonic()` is reserved for telemetry windowing.

3.  Every measurement reports a confidence interval.
    A score without a spread is not a measurement, it is a number.

4.  Correctness validation happens OUTSIDE the timed region, after the reps.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("BenchMind.CPU.Common")

# --------------------------------------------------------------------------
# Reference baselines
# --------------------------------------------------------------------------
# A score of 1000 points in a category means "matched the BenchMind reference
# machine". The reference machine and its raw measured metrics are documented
# in docs/BENCHMARK_SPEC.md. These are NOT arbitrary round numbers chosen to
# make results look nice -- if you recalibrate them, update the spec and the
# CHANGELOG in the same commit.
REFERENCE_MACHINE = "BenchMind Reference R1 (see docs/BENCHMARK_SPEC.md)"
BASELINE_VERSION = "2.0.0"

# Every number below is the median single-thread raw metric actually measured
# on reference machine R1, whose full specification and environment
# fingerprint are recorded in docs/BENCHMARK_SPEC.md. They are not round
# numbers picked to make scores look tidy.
#
# To recalibrate against your own reference, run:
#     python scripts/calibrate_baselines.py
# and update this block together with the spec and CHANGELOG in one commit.
CATEGORY_BASELINES: Dict[str, float] = {
    "integer": 3300.0,          # Mops/sec  - int64 SIMD ALU throughput
    "floating_point": 3750.0,   # MFLOPS    - L2-resident FP32/FP64 FMA
    "matrix": 65.0,             # GFLOPS    - single-thread BLAS dgemm
    "vector_simd": 1.0,         # GFLOPS    - DRAM-bound streaming FMA
    "compression": 22.5,        # MB/s      - zlib + bz2, mixed-entropy corpus
    "hashing": 800.0,           # MB/s      - SHA-256 + BLAKE2b
    "branch_heavy": 85.0,       # Mops/sec  - sort + binary search + gather
    "interpreter": 45.0,        # Mops/sec  - pure CPython loop (excluded from index)
}

# Categories that measure the CPython interpreter rather than the hardware.
# They are reported, but excluded from the composite index by default, because
# their score changes when you upgrade Python without touching the hardware.
RUNTIME_BOUND_CATEGORIES = {"interpreter"}


# --------------------------------------------------------------------------
# Result schema
# --------------------------------------------------------------------------
@dataclass
class SubtestResult:
    name: str
    category: str
    workload_profile: str   # compute_bound, memory_bound, branch_bound, vectorized,
                            # compression, crypto, mixed, runtime_bound
    status: str             # passed, failed, skipped
    execution_time: float   # total wall time of the subtest including setup
    repetitions: int
    best_time: float
    median_time: float
    worst_time: float
    std_dev: float
    stability_pct: float
    raw_metric_name: str
    raw_metric_value: float
    score: float

    # --- added in 2.0 ---
    score_ci_pct: float = 0.0        # +/- half-width of the 95% CI, percent
    raw_metric_ci_pct: float = 0.0
    setup_time: float = 0.0          # untimed setup cost, reported for transparency
    arithmetic_intensity: float = 0.0  # work units per byte moved (roofline input)
    bytes_moved: float = 0.0
    working_set_bytes: float = 0.0
    threads: int = 1
    measures_runtime: bool = False   # True => reflects CPython speed, not hardware
    counted_in_index: bool = True

    validation_passed: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadSpec:
    """Single source of truth for one benchmark workload."""
    key: str
    name: str
    category: str
    workload_profile: str
    raw_metric_name: str
    setup_fn: Callable[[float], Any]
    run_fn: Callable[[Any], Tuple[Any, float]]
    validate_fn: Callable[[Any], bool]
    # Roofline inputs, declared per workload rather than guessed later.
    arithmetic_intensity: float = 0.0   # work units per byte of DRAM traffic
    bytes_per_run: float = 0.0
    working_set_bytes: float = 0.0
    multi_core_capable: bool = True
    measures_runtime: bool = False
    default_scale: float = 1.0
    fields: Dict[str, Any] = field(default_factory=dict)

    @property
    def counted_in_index(self) -> bool:
        return self.category not in RUNTIME_BOUND_CATEGORIES


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
# Two-sided 95% t-distribution critical values, indexed by degrees of freedom.
# Hardcoded so BenchMind does not need SciPy for a table lookup.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}


def t_critical_95(dof: int) -> float:
    if dof <= 0:
        return float("inf")
    if dof in _T95:
        return _T95[dof]
    for k in sorted(_T95):
        if dof <= k:
            return _T95[k]
    return 1.96


def confidence_interval_pct(samples: List[float]) -> float:
    """
    Half-width of the 95% confidence interval of the mean, as a percentage of
    the mean. Returns 0.0 when there are not enough samples to say anything.
    """
    clean = [s for s in samples if s > 0 and math.isfinite(s)]
    if len(clean) < 2:
        return 0.0
    mean = statistics.fmean(clean)
    if mean <= 0:
        return 0.0
    sd = statistics.stdev(clean)
    half_width = t_critical_95(len(clean) - 1) * sd / math.sqrt(len(clean))
    return round((half_width / mean) * 100.0, 2)


def geometric_mean(scores: List[float]) -> float:
    """Geometric mean over normalized scores. Zero and non-finite values are dropped."""
    valid = [s for s in scores if s > 0 and math.isfinite(s)]
    if not valid:
        return 0.0
    return round(math.exp(sum(math.log(s) for s in valid) / len(valid)), 2)


def combine_ci_geometric(ci_pcts: List[float]) -> float:
    """
    Propagate per-subtest relative uncertainty through a geometric mean.

    For G = (x1*x2*...*xn)^(1/n), the relative uncertainty of G is
    (1/n) * sqrt(sum(rel_i^2)) assuming independent subtests.
    """
    clean = [c for c in ci_pcts if c and math.isfinite(c)]
    if not clean:
        return 0.0
    n = len(clean)
    return round(math.sqrt(sum(c * c for c in clean)) / n, 2)


def calculate_subtest_score(category: str, raw_metric_value: float) -> float:
    baseline = CATEGORY_BASELINES.get(category)
    if baseline is None:
        logger.warning("No baseline registered for category '%s'; score suppressed.", category)
        return 0.0
    if baseline <= 0 or raw_metric_value <= 0:
        return 0.0
    return round((raw_metric_value / baseline) * 1000.0, 2)


def calculate_cpu_index(
    single_core_score: float,
    multi_core_score: float,
    single_weight: float = 0.4,
    multi_weight: float = 0.6,
) -> float:
    """
    BenchMind CPU Index. Defaults to 0.4 * single + 0.6 * multi.

    Single-core and multi-core scores are only meaningfully combinable because
    both suites run the *identical* workloads at the *identical* per-unit scale
    (see benchmarks/cpu/registry.py). Do not change one side's scale without
    changing the other.
    """
    index = (single_weight * single_core_score) + (multi_weight * multi_core_score)
    return round(index, 2)


def combined_index_ci(single_ci: float, multi_ci: float,
                      single_score: float, multi_score: float,
                      single_weight: float = 0.4, multi_weight: float = 0.6) -> float:
    """Propagate uncertainty through the weighted sum."""
    a = single_weight * single_score * (single_ci / 100.0)
    b = multi_weight * multi_score * (multi_ci / 100.0)
    total = (single_weight * single_score) + (multi_weight * multi_score)
    if total <= 0:
        return 0.0
    return round((math.sqrt(a * a + b * b) / total) * 100.0, 2)


def scores_are_distinguishable(score_a: float, ci_a_pct: float,
                               score_b: float, ci_b_pct: float) -> bool:
    """
    True only when the two 95% intervals do not overlap.

    BenchMind refuses to claim machine A is faster than machine B when the
    measurement cannot support it.
    """
    a_lo = score_a * (1 - ci_a_pct / 100.0)
    a_hi = score_a * (1 + ci_a_pct / 100.0)
    b_lo = score_b * (1 - ci_b_pct / 100.0)
    b_hi = score_b * (1 + ci_b_pct / 100.0)
    return a_hi < b_lo or b_hi < a_lo


# --------------------------------------------------------------------------
# The timed runner
# --------------------------------------------------------------------------
def _failed_result(spec: "WorkloadSpec", elapsed: float, message: str,
                   setup_time: float = 0.0, threads: int = 1) -> SubtestResult:
    return SubtestResult(
        name=spec.name,
        category=spec.category,
        workload_profile=spec.workload_profile,
        status="failed",
        execution_time=round(elapsed, 6),
        repetitions=0,
        best_time=0.0,
        median_time=0.0,
        worst_time=0.0,
        std_dev=0.0,
        stability_pct=0.0,
        raw_metric_name=spec.raw_metric_name,
        raw_metric_value=0.0,
        score=0.0,
        setup_time=round(setup_time, 6),
        arithmetic_intensity=spec.arithmetic_intensity,
        bytes_moved=spec.bytes_per_run,
        working_set_bytes=spec.working_set_bytes,
        threads=threads,
        measures_runtime=spec.measures_runtime,
        counted_in_index=spec.counted_in_index,
        validation_passed=False,
        error_message=message,
    )


def run_timed_subtest(
    spec: WorkloadSpec,
    scale: float = 1.0,
    target_duration: float = 0.5,
    min_reps: int = 5,
    max_reps: int = 15,
) -> SubtestResult:
    """
    Execute one workload and return a fully statistically described result.

    Sequence:
      1. setup(scale)                       -- UNTIMED
      2. one warmup run                     -- UNTIMED (page faults, caches, JIT-ish effects)
      3. between min_reps and max_reps runs -- each individually timed with perf_counter
      4. validation of the last output      -- UNTIMED
      5. robust aggregation + 95% CI

    `min_reps` defaults to 5 rather than 3 because a 95% confidence interval
    from 3 samples has a t-critical value of 4.303 and is close to useless.
    """
    subtest_start = time.perf_counter()

    # 1. Untimed setup
    try:
        setup_t0 = time.perf_counter()
        ctx = spec.setup_fn(scale)
        setup_time = time.perf_counter() - setup_t0
    except Exception as e:  # noqa: BLE001
        logger.error("Subtest '%s' failed during setup: %s", spec.name, e, exc_info=True)
        return _failed_result(spec, time.perf_counter() - subtest_start, f"Setup error: {e}")

    # 2. Untimed warmup
    try:
        last_output, _ = spec.run_fn(ctx)
    except Exception as e:  # noqa: BLE001
        logger.error("Subtest '%s' failed during warmup: %s", spec.name, e, exc_info=True)
        return _failed_result(spec, time.perf_counter() - subtest_start,
                              f"Warmup error: {e}", setup_time)

    # 3. Timed repetitions
    run_times: List[float] = []
    work_per_run: List[float] = []
    reps_start = time.perf_counter()

    try:
        for _ in range(max_reps):
            t0 = time.perf_counter()
            output, work_units = spec.run_fn(ctx)
            t1 = time.perf_counter()

            run_times.append(max(t1 - t0, 1e-9))
            work_per_run.append(work_units)
            last_output = output

            if len(run_times) >= min_reps and (time.perf_counter() - reps_start) >= target_duration:
                break
    except Exception as e:  # noqa: BLE001
        logger.error("Subtest '%s' failed during execution: %s", spec.name, e, exc_info=True)
        return _failed_result(spec, time.perf_counter() - subtest_start,
                              f"Execution error: {e}", setup_time)

    # 4. Untimed validation
    try:
        validation_passed = bool(spec.validate_fn(last_output))
    except Exception as e:  # noqa: BLE001
        logger.error("Subtest '%s' validation raised: %s", spec.name, e, exc_info=True)
        validation_passed = False

    if not validation_passed:
        logger.warning("Subtest '%s' FAILED correctness validation. Score suppressed.", spec.name)

    # 5. Aggregation
    best_time = min(run_times)
    worst_time = max(run_times)
    median_time = statistics.median(run_times)
    std_dev = statistics.stdev(run_times) if len(run_times) > 1 else 0.0
    safe_median = max(median_time, 1e-9)

    median_work = statistics.median(work_per_run) if work_per_run else 0.0
    raw_metric_value = round(median_work / safe_median, 3)

    # Per-rep throughput samples -> CI on the metric itself, not on the timings.
    throughputs = [w / t for w, t in zip(work_per_run, run_times) if t > 0]
    ci_pct = confidence_interval_pct(throughputs)

    cv = std_dev / safe_median
    stability_pct = max(0.0, min(100.0, (1.0 - cv) * 100.0))

    score = calculate_subtest_score(spec.category, raw_metric_value) if validation_passed else 0.0

    return SubtestResult(
        name=spec.name,
        category=spec.category,
        workload_profile=spec.workload_profile,
        status="passed" if validation_passed else "failed",
        execution_time=round(time.perf_counter() - subtest_start, 6),
        repetitions=len(run_times),
        best_time=round(best_time, 6),
        median_time=round(median_time, 6),
        worst_time=round(worst_time, 6),
        std_dev=round(std_dev, 6),
        stability_pct=round(stability_pct, 2),
        raw_metric_name=spec.raw_metric_name,
        raw_metric_value=raw_metric_value,
        score=score,
        score_ci_pct=ci_pct,
        raw_metric_ci_pct=ci_pct,
        setup_time=round(setup_time, 6),
        arithmetic_intensity=spec.arithmetic_intensity,
        bytes_moved=spec.bytes_per_run * scale,
        working_set_bytes=spec.working_set_bytes * scale,
        threads=1,
        measures_runtime=spec.measures_runtime,
        counted_in_index=spec.counted_in_index,
        validation_passed=validation_passed,
    )
