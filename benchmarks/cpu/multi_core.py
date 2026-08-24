import time
import psutil
import concurrent.futures
import logging
from typing import List, Tuple, Dict, Any

from benchmarks.cpu.common import SubtestResult, calculate_subtest_score, geometric_mean
from benchmarks.cpu.integer import validate_integer_workload
from benchmarks.cpu.floating_point import validate_floating_point_workload
from benchmarks.cpu.matrix import validate_matrix_workload
from benchmarks.cpu.compression import validate_compression_workload
from benchmarks.cpu.hashing import validate_hashing_workload

# Import the worker from a dedicated top-level module so Windows 'spawn' can reimport it cleanly
from benchmarks.cpu._worker import _worker_task

logger = logging.getLogger("BenchMind.CPU.MultiCore")


def run_multi_core_subtest(
    name: str,
    category: str,
    workload_profile: str,
    raw_metric_name: str,
    workload_type: str,
    validate_fn: Any,
    num_workers: int,
    chunks_per_worker: int = 4
) -> SubtestResult:
    """
    Executes multi-core subtest via dynamic ProcessPoolExecutor task queue chunking.
    Uses time.perf_counter() for precise multi-core performance duration measurement.
    Worker dispatch is handled by benchmarks.cpu._worker._worker_task, which is
    a dedicated spawn-safe top-level module on Windows.
    """
    total_tasks = num_workers * chunks_per_worker
    start_total = time.perf_counter()

    outputs = []
    total_ops_combined = 0.0
    validation_passed = True

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_task, workload_type) for _ in range(total_tasks)]

            for future in concurrent.futures.as_completed(futures):
                out, ops = future.result()
                outputs.append(out)
                total_ops_combined += ops

                if not validate_fn(out):
                    validation_passed = False

    except Exception as e:
        logger.error("Multi-core subtest '%s' execution failed: %s", name, e, exc_info=True)
        return SubtestResult(
            name=name,
            category=category,
            workload_profile=workload_profile,
            status="failed",
            execution_time=round(time.perf_counter() - start_total, 6),
            repetitions=total_tasks,
            best_time=0.0,
            median_time=0.0,
            worst_time=0.0,
            std_dev=0.0,
            stability_pct=0.0,
            raw_metric_name=raw_metric_name,
            raw_metric_value=0.0,
            score=0.0,
            validation_passed=False,
            error_message=str(e)
        )

    elapsed_time = max(time.perf_counter() - start_total, 1e-6)
    raw_metric_value = round(total_ops_combined / elapsed_time, 2)
    score = calculate_subtest_score(category, raw_metric_value) if validation_passed else 0.0
    status = "passed" if validation_passed else "failed"

    return SubtestResult(
        name=name,
        category=category,
        workload_profile=workload_profile,
        status=status,
        execution_time=round(elapsed_time, 6),
        repetitions=total_tasks,
        best_time=round(elapsed_time, 6),
        median_time=round(elapsed_time, 6),
        worst_time=round(elapsed_time, 6),
        std_dev=0.0,
        stability_pct=100.0 if validation_passed else 0.0,
        raw_metric_name=raw_metric_name,
        raw_metric_value=raw_metric_value,
        score=score,
        validation_passed=validation_passed
    )


def run_multi_core_suite() -> Tuple[List[SubtestResult], float, float, int]:
    """
    Executes multi-core suite using all available logical cores.
    Returns (results, composite_multi_core_score, total_time, logical_cores).
    """
    logical_cores = psutil.cpu_count(logical=True) or 1
    start_time = time.perf_counter()
    results: List[SubtestResult] = []

    results.append(run_multi_core_subtest(
        name="Multi-Core Integer Compute",
        category="integer",
        workload_profile="compute_bound",
        raw_metric_name="Mops/sec",
        workload_type="integer",
        validate_fn=validate_integer_workload,
        num_workers=logical_cores
    ))

    results.append(run_multi_core_subtest(
        name="Multi-Core Floating-Point",
        category="floating_point",
        workload_profile="compute_bound",
        raw_metric_name="MFLOPS",
        workload_type="floating_point",
        validate_fn=validate_floating_point_workload,
        num_workers=logical_cores
    ))

    results.append(run_multi_core_subtest(
        name="Multi-Core Matrix Compute",
        category="matrix",
        workload_profile="mixed",
        raw_metric_name="GFLOPS",
        workload_type="matrix",
        validate_fn=validate_matrix_workload,
        num_workers=logical_cores
    ))

    results.append(run_multi_core_subtest(
        name="Multi-Core Compression",
        category="compression",
        workload_profile="compression",
        raw_metric_name="MB/s",
        workload_type="compression",
        validate_fn=validate_compression_workload,
        num_workers=logical_cores
    ))

    results.append(run_multi_core_subtest(
        name="Multi-Core Cryptographic Hashing",
        category="hashing",
        workload_profile="crypto",
        raw_metric_name="MB/s",
        workload_type="hashing",
        validate_fn=validate_hashing_workload,
        num_workers=logical_cores
    ))

    total_time = time.perf_counter() - start_time
    passed_scores = [r.score for r in results if r.validation_passed]
    composite_score = geometric_mean(passed_scores)

    return results, composite_score, round(total_time, 6), logical_cores
