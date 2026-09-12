"""
Multi-core CPU suite.

Fixes carried over from the review of 1.x:

* The process pool is created and warmed BEFORE any timing starts. Worker
  startup and NumPy import used to sit inside the measured window.
* Each worker builds its workload context once (see _worker.prepare) instead of
  regenerating arrays on every dispatched chunk.
* The suite now performs real repetitions and reports median, standard
  deviation and a 95% confidence interval. 1.x hardcoded `std_dev=0.0` and
  `stability_pct=100.0` from a single measurement.
* Workloads and scales come from the shared registry, so single-core and
  multi-core measure the same unit of work and the blended CPU Index means
  something.

Also included: `run_scaling_curve`, which sweeps thread count and reports
scaling efficiency. On a hybrid CPU this is where P-core versus E-core
behaviour, SMT gain, and the memory-bandwidth saturation point become visible.
"""

from __future__ import annotations

import concurrent.futures
import logging
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from benchmarks.cpu import registry
from benchmarks.cpu._worker import execute as worker_execute
from benchmarks.cpu._worker import init_worker, prepare as worker_prepare, warmup as worker_warmup
from benchmarks.cpu.common import (
    SubtestResult,
    calculate_subtest_score,
    combine_ci_geometric,
    confidence_interval_pct,
    geometric_mean,
)

logger = logging.getLogger("BenchMind.CPU.MultiCore")

DEFAULT_REPS = 5

# Chunking is chosen adaptively per workload; see `_choose_waves`.
#
# A "wave" is one chunk per worker. With too few waves, a repetition's wall
# time is set by whichever worker finishes last, and on a hybrid CPU an E-core
# chunk takes substantially longer than a P-core one -- so every repetition is
# dominated by a straggler and which core drew which chunk varies run to run.
#
# Measured on an i5-13450HX (6P + 4E) at the fixed 2 waves used through 2.1.1:
#     single-core confidence interval  +/- 0.7%
#     multi-core confidence interval   +/- 5.6%
#
# More waves let the pool self-balance: residual imbalance is roughly one
# chunk out of `waves`, so 8 waves cuts it to an eighth. But a workload with a
# 1.3 s chunk (compression) cannot afford 8 waves, hence the time budget.
MIN_WAVES = 2
MAX_WAVES = 8
TARGET_REP_SECONDS = 1.0


class WarmPool:
    """
    A ProcessPoolExecutor that is fully started before it is used.

    Creating the pool is not enough: on the spawn start method, workers are
    created lazily as tasks arrive. We submit one ping per worker slot and wait
    for distinct PIDs, which forces every process into existence and pays the
    interpreter and NumPy import cost up front.
    """

    def __init__(self, max_workers: int):
        self.max_workers = max(1, max_workers)
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.pids: set = set()
        self.startup_seconds = 0.0

    def __enter__(self) -> "WarmPool":
        t0 = time.perf_counter()
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_worker,
        )
        # Oversubscribe the ping so every slot gets work and every process starts.
        futures = [self.executor.submit(worker_warmup)
                   for _ in range(self.max_workers * 4)]
        for f in concurrent.futures.as_completed(futures):
            try:
                self.pids.add(f.result())
            except Exception as e:  # noqa: BLE001
                logger.warning("Worker warmup ping failed: %s", e)
        self.startup_seconds = time.perf_counter() - t0
        logger.info("Warm pool ready: %d workers, %d distinct PIDs, %.3fs startup",
                    self.max_workers, len(self.pids), self.startup_seconds)
        return self

    def __exit__(self, *exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        return False

    def prepare_workload(self, workload_key: str, scale: float) -> None:
        """Force every worker to build and cache its context before timing."""
        assert self.executor is not None
        futures = [self.executor.submit(worker_prepare, workload_key, scale)
                   for _ in range(self.max_workers * 4)]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                logger.error("Worker prepare failed for %s: %s", workload_key, e)


def _choose_waves(chunk_seconds: float) -> int:
    """
    Pick how many chunks per worker to dispatch, from one timed chunk.

    Enough waves that the pool can hide the P-core/E-core speed difference,
    few enough that a slow workload does not make each repetition take
    forever. Clamped to [MIN_WAVES, MAX_WAVES].
    """
    if chunk_seconds <= 0:
        return MAX_WAVES
    waves = round(TARGET_REP_SECONDS / chunk_seconds)
    return int(max(MIN_WAVES, min(MAX_WAVES, waves)))


def run_multi_core_subtest(
    pool: WarmPool,
    spec: registry.WorkloadSpec,
    num_workers: int,
    scale: float = 1.0,
    chunks_per_worker: Optional[int] = None,
    reps: int = DEFAULT_REPS,
) -> SubtestResult:
    """
    Run one workload across `num_workers` processes, `reps` times.

    Timing starts only after the pool is warm, every worker has its context
    cached, AND one calibration wave has been timed so the chunk count can be
    chosen to suit this workload's duration. None of that is inside the
    measured window.
    """
    subtest_start = time.perf_counter()

    prep_t0 = time.perf_counter()
    pool.prepare_workload(spec.key, scale)

    # Untimed calibration: one wave, to learn how long a chunk takes here.
    if chunks_per_worker is None:
        assert pool.executor is not None
        cal_t0 = time.perf_counter()
        cal_futures = [pool.executor.submit(worker_execute, spec.key, scale, 1)
                       for _ in range(num_workers)]
        for f in concurrent.futures.as_completed(cal_futures):
            f.result()
        chunk_seconds = time.perf_counter() - cal_t0
        waves = _choose_waves(chunk_seconds)
        logger.info("%s: chunk ~%.3fs -> %d waves (%d chunks over %d workers)",
                    spec.key, chunk_seconds, waves, waves * num_workers, num_workers)
    else:
        waves = max(1, chunks_per_worker)
        chunk_seconds = 0.0

    setup_time = time.perf_counter() - prep_t0

    total_tasks = max(1, num_workers * waves)
    run_times: List[float] = []
    work_per_rep: List[float] = []
    validation_passed = True
    worker_pids: set = set()

    try:
        assert pool.executor is not None
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            futures = [
                pool.executor.submit(worker_execute, spec.key, scale, 1)
                for _ in range(total_tasks)
            ]
            total_work = 0.0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                total_work += res["work_units"]
                worker_pids.add(res["pid"])
                if not res["valid"]:
                    validation_passed = False
            elapsed = max(time.perf_counter() - t0, 1e-9)

            run_times.append(elapsed)
            work_per_rep.append(total_work)

    except Exception as e:  # noqa: BLE001
        logger.error("Multi-core subtest '%s' failed: %s", spec.name, e, exc_info=True)
        return SubtestResult(
            name=f"Multi-Core {spec.name}",
            category=spec.category,
            workload_profile=spec.workload_profile,
            status="failed",
            execution_time=round(time.perf_counter() - subtest_start, 6),
            repetitions=0,
            best_time=0.0, median_time=0.0, worst_time=0.0,
            std_dev=0.0, stability_pct=0.0,
            raw_metric_name=spec.raw_metric_name,
            raw_metric_value=0.0, score=0.0,
            setup_time=round(setup_time, 6),
            threads=num_workers,
            measures_runtime=spec.measures_runtime,
            counted_in_index=spec.counted_in_index,
            validation_passed=False,
            error_message=str(e),
        )

    best_time = min(run_times)
    worst_time = max(run_times)
    median_time = statistics.median(run_times)
    std_dev = statistics.stdev(run_times) if len(run_times) > 1 else 0.0
    safe_median = max(median_time, 1e-9)

    median_work = statistics.median(work_per_rep)
    raw_metric_value = round(median_work / safe_median, 3)

    throughputs = [w / t for w, t in zip(work_per_rep, run_times) if t > 0]
    ci_pct = confidence_interval_pct(throughputs)

    cv = std_dev / safe_median
    stability_pct = max(0.0, min(100.0, (1.0 - cv) * 100.0))

    score = calculate_subtest_score(spec.category, raw_metric_value) if validation_passed else 0.0

    return SubtestResult(
        name=f"Multi-Core {spec.name}",
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
        bytes_moved=spec.bytes_per_run * scale * total_tasks,
        working_set_bytes=spec.working_set_bytes * scale,
        threads=len(worker_pids) or num_workers,
        chunk_waves=waves,
        # Residual load imbalance is bounded by roughly one chunk out of
        # `waves`. Reported because it is the dominant remaining source of
        # multi-core spread on the two slowest workloads, where the time
        # budget forces waves down to MIN_WAVES.
        imbalance_bound_pct=round(100.0 / max(waves, 1), 1),
        measures_runtime=spec.measures_runtime,
        counted_in_index=spec.counted_in_index,
        validation_passed=validation_passed,
    )


def run_multi_core_suite(
    scale: float = 1.0,
    reps: int = DEFAULT_REPS,
    include_runtime_bound: bool = True,
    workers: Optional[int] = None,
) -> Tuple[List[SubtestResult], float, float, int, Dict[str, Any]]:
    """
    Execute every registered workload across all logical cores.

    Returns (results, composite_score, total_time, cores_used, meta).
    """
    logical_cores = workers or psutil.cpu_count(logical=True) or 1
    specs = registry.WORKLOADS if include_runtime_bound else registry.INDEX_WORKLOADS

    start_time = time.perf_counter()
    results: List[SubtestResult] = []

    with WarmPool(logical_cores) as pool:
        pool_startup = pool.startup_seconds
        for spec in specs:
            logger.info("Multi-core: %s", spec.name)
            results.append(
                run_multi_core_subtest(pool, spec, logical_cores, scale=scale, reps=reps)
            )

    total_time = time.perf_counter() - start_time

    scored = [r for r in results if r.validation_passed and r.counted_in_index]
    composite = geometric_mean([r.score for r in scored])
    composite_ci = combine_ci_geometric([r.score_ci_pct for r in scored])

    meta = {
        "pool_startup_seconds": round(pool_startup, 4),
        "pool_warmed_before_timing": True,
        "adaptive_chunking": True,
        "chunk_waves_range": [MIN_WAVES, MAX_WAVES],
        "chunk_waves_by_workload": {
            r.category: r.chunk_waves for r in results if r.chunk_waves
        },
        "load_balance_limited": sorted(
            r.category for r in results if r.chunk_waves <= MIN_WAVES
        ),
        "composite_ci_pct": composite_ci,
        "reps_per_subtest": reps,
        "subtests_counted": len(scored),
    }

    return results, composite, round(total_time, 6), logical_cores, meta


# --------------------------------------------------------------------------
# Thread scaling curve
# --------------------------------------------------------------------------
def _thread_counts(max_threads: int) -> List[int]:
    counts = [1]
    n = 2
    while n < max_threads:
        counts.append(n)
        n *= 2
    if max_threads not in counts:
        counts.append(max_threads)
    return counts


def run_scaling_curve(
    workload_key: str = "integer",
    scale: float = 1.0,
    reps: int = 3,
    max_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sweep thread count over 1, 2, 4, ... N and report scaling efficiency.

    Efficiency at N threads is throughput(N) / (N * throughput(1)). Perfect
    scaling is 1.0. Reading the curve:

      * a knee that appears exactly at the physical core count -> SMT gain is
        small for this workload
      * a knee well before the core count on `vector_simd` -> memory bandwidth
        saturated, adding threads will not help
      * a gentle taper rather than a knee on a hybrid CPU -> E-cores are
        contributing less per thread than P-cores, which is expected
    """
    max_threads = max_threads or psutil.cpu_count(logical=True) or 1
    spec = registry.get(workload_key)
    points: List[Dict[str, Any]] = []
    baseline_throughput = None

    for n in _thread_counts(max_threads):
        with WarmPool(n) as pool:
            result = run_multi_core_subtest(
                pool, spec, n, scale=scale, reps=reps,
            )
        throughput = result.raw_metric_value
        if baseline_throughput is None:
            baseline_throughput = throughput or 1e-9

        speedup = throughput / baseline_throughput if baseline_throughput else 0.0
        points.append({
            "threads": n,
            "raw_metric_value": throughput,
            "raw_metric_name": spec.raw_metric_name,
            "ci_pct": result.score_ci_pct,
            "speedup": round(speedup, 3),
            "efficiency": round(speedup / n, 3) if n else 0.0,
            "validation_passed": result.validation_passed,
        })

    # Saturation point: last thread count that still added at least 15% throughput.
    saturation = points[0]["threads"] if points else 1
    for prev, cur in zip(points, points[1:]):
        if cur["speedup"] >= prev["speedup"] * 1.15:
            saturation = cur["threads"]

    return {
        "workload": workload_key,
        "workload_name": spec.name,
        "workload_profile": spec.workload_profile,
        "max_threads": max_threads,
        "points": points,
        "saturation_threads": saturation,
        "peak_speedup": max((p["speedup"] for p in points), default=0.0),
        "efficiency_at_max": points[-1]["efficiency"] if points else 0.0,
    }
