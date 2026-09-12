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

2b. WARMUP IS A DURATION, NOT A REPETITION COUNT (2.1.0).
    A single warmup repetition is enough for a 500 ms workload and useless for
    a 10 ms one. A modern laptop CPU bursts to its maximum turbo and then
    settles toward a sustained clock over hundreds of milliseconds, so a short
    workload with one warmup rep samples a different point on that ramp every
    time. Warmup now runs until at least WARMUP_MIN_SECONDS has elapsed.

2c. REPETITIONS THAT ARE TOO SHORT ARE FLAGGED (2.1.0).
    Below MIN_USEFUL_REP_SECONDS a repetition cannot average out OS scheduling
    quanta, interrupts or clock transitions, and no amount of repetition fixes
    a structurally noisy measurement. The result says so rather than reporting
    a confident-looking number.

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
REFERENCE_MACHINE = "BenchMind Reference R2 (see docs/BENCHMARK_SPEC.md)"

# Bumped in 2.3.0: the raw metric is now estimated from the fastest half of
# the repetitions rather than the median (see robust_throughput below), which
# changes every number slightly upward. Not comparable to 2.1.x or 2.2.x.
#
# NOTE: `branch_heavy` below is a 2.3.1 value and is now STALE. Its random
# gather no longer pays NumPy's index bounds check (see branch_heavy.py,
# GATHER_MODE), which cut that component's cost by about 36%. Recalibrate:
#     python -m scripts.calibrate_baselines --reps 5
# Every other category is unaffected by the 2.4.0 change.
BASELINE_VERSION = "2.4.0"

# Every number below is the median single-thread raw metric measured on
# reference machine R2 -- a physical Lenovo laptop with an i5-13450HX -- over
# five calibration passes, with the validity gate reporting 'valid'. The worst
# spread across all eight categories was 1.68%; five were under 0.5%.
#
# R2 replaces R1, which was a shared cloud instance and therefore a poor
# reference: noisy neighbours and unknown turbo behaviour. Full specification
# and per-category spreads are in docs/BENCHMARK_SPEC.md.
#
# To recalibrate against your own reference, run:
#     python -m scripts.calibrate_baselines --reps 5
# The script refuses to emit a block whose spread exceeds 5%. Update this
# table, BASELINE_VERSION, the spec and the CHANGELOG in one commit.
CATEGORY_BASELINES: Dict[str, float] = {
    "integer": 2258.54,         # Mops/sec  spread 0.45%
    "floating_point": 5251.46,  # MFLOPS    spread 3.23%
    "matrix": 47.07,            # GFLOPS    spread 0.20%
    "vector_simd": 1.75,        # GFLOPS    spread 0.42%
    "compression": 24.89,       # MB/s      spread 0.21%
    "hashing": 870.27,          # MB/s      spread 1.11%
    "branch_heavy": 34.72,      # Mops/sec  spread 1.12%
    "interpreter": 45.45,       # Mops/sec  spread 0.40%
}

# Warmup runs until this much time has elapsed, not for a fixed rep count.
# One repetition is enough for a 500 ms workload and useless for a 10 ms one:
# a laptop CPU settles from peak turbo toward its sustained clock over
# hundreds of milliseconds.
WARMUP_MIN_SECONDS = 0.25
WARMUP_MAX_REPS = 500

# A repetition shorter than this cannot average out scheduling quanta,
# interrupts or clock transitions. Flagged, not silently accepted.
MIN_USEFUL_REP_SECONDS = 0.020

# Contention handling (2.3.0).
#
# Interference in a pinned single-thread measurement is ONE-DIRECTIONAL: a
# competing thread can only ever make the measurement slower, never faster.
# The median is therefore the wrong estimator, because it treats a slow
# repetition as equally likely to be signal as a fast one.
#
# This was measured on reference machine R2, where five categories landed
# within 2% of baseline while `integer` reported 1,185 Mops/sec against its
# 2,106 baseline -- a 44% shortfall. The shape is exactly SMT contention: the
# process is pinned to logical core 10, and its sibling logical 11 cannot be
# reserved, so anything Windows schedules there shares one physical core's
# execution units.
#
# Two changes follow. `robust_throughput` estimates from the fastest half of
# the repetitions, which is unbiased under one-directional interference. And
# when contention is still evident, the whole measurement is retried, because
# no estimator can recover a subtest that was contended end to end.
CONTENTION_RETRY_PCT = 8.0
MAX_MEASUREMENT_ATTEMPTS = 3

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
    contention_pct: float = 0.0      # how far the median sat below the estimate
    attempts: int = 1                # measurement attempts used
    warmup_time: float = 0.0         # untimed warmup cost
    warmup_reps: int = 0
    short_rep_warning: bool = False   # repetitions too brief to be stable
    chunk_waves: int = 0              # multi-core: chunks dispatched per worker
    imbalance_bound_pct: float = 0.0  # multi-core: worst-case straggler effect
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


def robust_throughput(throughputs: List[float]) -> Dict[str, float]:
    """
    Estimate uncontended throughput from a set of per-repetition samples.

    Uses the mean of the fastest half. Under one-directional interference --
    which is what CPU contention is -- the slow tail is contamination, not
    signal, so a central estimator is biased downward by construction.

    `contention_pct` is how far the overall median sits below this estimate.
    It is a direct measure of how much of the run was disturbed: 0% means
    every repetition agreed, 40% means most of them were slowed.

    Returns estimate, its 95% CI, and the contention figure.
    """
    clean = sorted((t for t in throughputs if t > 0 and math.isfinite(t)),
                   reverse=True)
    if not clean:
        return {"estimate": 0.0, "ci_pct": 0.0, "contention_pct": 0.0, "samples": 0}
    if len(clean) < 4:
        estimate = statistics.median(clean)
        return {"estimate": estimate,
                "ci_pct": confidence_interval_pct(clean),
                "contention_pct": 0.0,
                "samples": len(clean)}

    keep = max(3, (len(clean) + 1) // 2)
    fastest = clean[:keep]
    estimate = statistics.fmean(fastest)
    overall_median = statistics.median(clean)
    contention = ((estimate - overall_median) / estimate * 100.0) if estimate > 0 else 0.0

    return {
        "estimate": estimate,
        "ci_pct": confidence_interval_pct(fastest),
        "contention_pct": round(max(0.0, contention), 2),
        "samples": len(clean),
    }


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

    # 2. Untimed warmup, run for a minimum DURATION rather than a fixed count.
    #    This is what gets a short workload past the CPU's turbo ramp before
    #    any of its repetitions are timed.
    warmup_reps = 0
    warmup_start = time.perf_counter()
    try:
        while True:
            last_output, _ = spec.run_fn(ctx)
            warmup_reps += 1
            elapsed = time.perf_counter() - warmup_start
            if elapsed >= WARMUP_MIN_SECONDS or warmup_reps >= WARMUP_MAX_REPS:
                break
    except Exception as e:  # noqa: BLE001
        logger.error("Subtest '%s' failed during warmup: %s", spec.name, e, exc_info=True)
        return _failed_result(spec, time.perf_counter() - subtest_start,
                              f"Warmup error: {e}", setup_time)
    warmup_time = time.perf_counter() - warmup_start

    # 3. Timed repetitions, retried if contention is evident.
    #
    #    Retrying is not double-dipping: interference only ever slows a pinned
    #    measurement, so a later attempt that runs faster is strictly closer to
    #    the machine's real capability. Attempts stop as soon as one comes back
    #    clean, so an idle machine pays nothing.
    best: Optional[Dict[str, Any]] = None
    attempts = 0

    try:
        for attempt in range(1, MAX_MEASUREMENT_ATTEMPTS + 1):
            attempts = attempt
            run_times: List[float] = []
            work_per_run: List[float] = []
            reps_start = time.perf_counter()

            for _ in range(max_reps):
                t0 = time.perf_counter()
                output, work_units = spec.run_fn(ctx)
                t1 = time.perf_counter()

                run_times.append(max(t1 - t0, 1e-9))
                work_per_run.append(work_units)
                last_output = output

                if (len(run_times) >= min_reps
                        and (time.perf_counter() - reps_start) >= target_duration):
                    break

            throughputs = [w / t for w, t in zip(work_per_run, run_times) if t > 0]
            robust = robust_throughput(throughputs)

            if best is None or robust["estimate"] > best["robust"]["estimate"]:
                best = {"robust": robust, "run_times": run_times,
                        "work_per_run": work_per_run}

            if robust["contention_pct"] <= CONTENTION_RETRY_PCT:
                break

            if attempt < MAX_MEASUREMENT_ATTEMPTS:
                logger.info(
                    "Subtest '%s' showed %.1f%% contention; re-measuring "
                    "(attempt %d of %d).",
                    spec.name, robust["contention_pct"], attempt + 1,
                    MAX_MEASUREMENT_ATTEMPTS)
    except Exception as e:  # noqa: BLE001
        logger.error("Subtest '%s' failed during execution: %s", spec.name, e, exc_info=True)
        return _failed_result(spec, time.perf_counter() - subtest_start,
                              f"Execution error: {e}", setup_time)

    assert best is not None
    run_times = best["run_times"]
    work_per_run = best["work_per_run"]
    robust = best["robust"]

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

    raw_metric_value = round(robust["estimate"], 3)
    ci_pct = robust["ci_pct"]
    contention_pct = robust["contention_pct"]

    if contention_pct > CONTENTION_RETRY_PCT:
        logger.warning(
            "Subtest '%s' remained %.1f%% contended after %d attempts. Another "
            "process is competing for this core, most likely on its SMT sibling, "
            "which BenchMind cannot reserve.",
            spec.name, contention_pct, attempts)

    cv = std_dev / safe_median
    stability_pct = max(0.0, min(100.0, (1.0 - cv) * 100.0))

    score = calculate_subtest_score(spec.category, raw_metric_value) if validation_passed else 0.0

    # Only meaningful at representative scale. A deliberately scaled-down run
    # (quick mode, or a test using scale=0.02) has short repetitions by
    # construction, and warning about it is noise rather than information.
    short_rep = median_time < MIN_USEFUL_REP_SECONDS and scale >= 1.0
    if short_rep:
        logger.warning(
            "Subtest '%s' has a median repetition of %.1f ms, below the %.0f ms "
            "floor. Its spread will be dominated by scheduling noise rather than "
            "by the hardware; raise its loop count.",
            spec.name, median_time * 1000, MIN_USEFUL_REP_SECONDS * 1000)

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
        contention_pct=contention_pct,
        attempts=attempts,
        warmup_time=round(warmup_time, 6),
        warmup_reps=warmup_reps,
        short_rep_warning=short_rep,
        arithmetic_intensity=spec.arithmetic_intensity,
        bytes_moved=spec.bytes_per_run * scale,
        working_set_bytes=spec.working_set_bytes * scale,
        threads=1,
        measures_runtime=spec.measures_runtime,
        counted_in_index=spec.counted_in_index,
        validation_passed=validation_passed,
    )
