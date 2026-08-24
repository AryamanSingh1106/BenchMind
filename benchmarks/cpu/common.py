import time
import math
import statistics
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger("BenchMind.CPU.Common")

# Internal BenchMind Reference Baselines for Score Normalization
# (Note: These are arbitrary internal BenchMind references, NOT Cinebench/Geekbench equivalents)
CATEGORY_BASELINES: Dict[str, float] = {
    "integer": 100.0,        # 100.0 Mops/sec -> 1000 pts
    "floating_point": 200.0, # 200.0 MFLOPS -> 1000 pts
    "matrix": 35.0,          # Recalibrated single-thread BLAS: 35.0 GFLOPS -> 1000 pts
    "vector_simd": 5.0,      # 5.0 GFLOPS -> 1000 pts
    "compression": 50.0,     # 50.0 MB/s -> 1000 pts
    "hashing": 150.0,        # 150.0 MB/s -> 1000 pts
    "branch_heavy": 50.0,    # 50.0 Mops/sec -> 1000 pts
}



@dataclass
class SubtestResult:
    name: str
    category: str
    workload_profile: str  # compute_bound, memory_bound, branch_bound, vectorized, compression, crypto, mixed
    status: str            # passed, failed, skipped
    execution_time: float
    repetitions: int
    best_time: float
    median_time: float
    worst_time: float
    std_dev: float
    stability_pct: float
    raw_metric_name: str
    raw_metric_value: float
    score: float
    validation_passed: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_subtest_score(category: str, raw_metric_value: float) -> float:
    baseline = CATEGORY_BASELINES.get(category, 100.0)
    if baseline <= 0 or raw_metric_value <= 0:
        return 0.0
    return round((raw_metric_value / baseline) * 1000.0, 2)


def geometric_mean(scores: List[float]) -> float:
    """
    Calculates the geometric mean over a list of normalized scores.
    Handles zero and invalid scores safely.
    """
    valid_scores = [s for s in scores if s > 0 and not math.isnan(s) and not math.isinf(s)]
    if not valid_scores:
        return 0.0
    log_sum = sum(math.log(s) for s in valid_scores)
    return round(math.exp(log_sum / len(valid_scores)), 2)


def calculate_cpu_index(
    single_core_score: float,
    multi_core_score: float,
    single_weight: float = 0.4,
    multi_weight: float = 0.6
) -> float:
    """
    Configurable BenchMind CPU Index calculation.
    Defaults to 0.4 * single + 0.6 * multi.
    """
    index = (single_weight * single_core_score) + (multi_weight * multi_core_score)
    return round(index, 2)


def run_timed_subtest(
    name: str,
    category: str,
    workload_profile: str,
    raw_metric_name: str,
    workload_fn: Callable[[], Tuple[Any, float]],
    validate_fn: Callable[[Any], bool],
    target_duration: float = 0.5,
    min_reps: int = 3,
    max_reps: int = 5
) -> SubtestResult:
    """
    Executes a benchmark subtest:
    1. Warmup call
    2. Calibrated repetitions with high-resolution time.perf_counter()
    3. Result validation outside the timed block
    4. Robust statistical aggregation
    """
    start_total = time.perf_counter()

    # 1. Warmup run
    try:
        warmup_out, _ = workload_fn()
    except Exception as e:
        logger.error("Subtest '%s' failed during warmup: %s", name, e, exc_info=True)
        return SubtestResult(
            name=name,
            category=category,
            workload_profile=workload_profile,
            status="failed",
            execution_time=round(time.perf_counter() - start_total, 6),
            repetitions=0,
            best_time=0.0,
            median_time=0.0,
            worst_time=0.0,
            std_dev=0.0,
            stability_pct=0.0,
            raw_metric_name=raw_metric_name,
            raw_metric_value=0.0,
            score=0.0,
            validation_passed=False,
            error_message=f"Warmup error: {e}"
        )

    # 2. Timed Repetitions using high-precision time.perf_counter()
    run_times: List[float] = []
    ops_per_run: List[float] = []
    last_output = warmup_out

    for _ in range(max_reps):
        t0 = time.perf_counter()
        output, total_ops = workload_fn()
        t1 = time.perf_counter()

        elapsed = max(t1 - t0, 1e-9)
        run_times.append(elapsed)
        ops_per_run.append(total_ops)
        last_output = output

        # Stop if target duration accumulated and min reps reached
        if len(run_times) >= min_reps and (time.perf_counter() - start_total) >= target_duration:
            break

    total_time = time.perf_counter() - start_total

    # 3. Validation outside the timed region
    try:
        validation_passed = validate_fn(last_output)
    except Exception as e:
        logger.error("Subtest '%s' validation raised exception: %s", name, e, exc_info=True)
        validation_passed = False

    if not validation_passed:
        logger.warning("Subtest '%s' failed correctness validation!", name)

    # 4. Statistical Metrics
    best_time = min(run_times)
    worst_time = max(run_times)
    median_time = statistics.median(run_times)
    std_dev = statistics.stdev(run_times) if len(run_times) > 1 else 0.0

    # Guard against median_time rounding down to 0
    safe_median_time = max(median_time, 1e-9)

    # Calculate raw metric value from median run
    median_ops = statistics.median(ops_per_run) if ops_per_run else 0.0
    raw_metric_value = round(median_ops / safe_median_time, 2)

    # Calculate stability percentage
    cv = std_dev / safe_median_time
    stability_pct = max(0.0, min(100.0, (1.0 - cv) * 100.0))

    score = calculate_subtest_score(category, raw_metric_value) if validation_passed else 0.0
    status = "passed" if validation_passed else "failed"

    return SubtestResult(
        name=name,
        category=category,
        workload_profile=workload_profile,
        status=status,
        execution_time=round(total_time, 6),
        repetitions=len(run_times),
        best_time=round(best_time, 6),
        median_time=round(median_time, 6),
        worst_time=round(worst_time, 6),
        std_dev=round(std_dev, 6),
        stability_pct=round(stability_pct, 2),
        raw_metric_name=raw_metric_name,
        raw_metric_value=raw_metric_value,
        score=score,
        validation_passed=validation_passed
    )
