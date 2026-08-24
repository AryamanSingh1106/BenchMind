import time
import threadpoolctl
from typing import List, Tuple

from benchmarks.cpu.common import SubtestResult, geometric_mean
from benchmarks.cpu.integer import run_integer_subtest
from benchmarks.cpu.floating_point import run_floating_point_subtest
from benchmarks.cpu.matrix import run_matrix_subtest
from benchmarks.cpu.vector_simd import run_vector_simd_subtest
from benchmarks.cpu.compression import run_compression_subtest
from benchmarks.cpu.hashing import run_hashing_subtest
from benchmarks.cpu.branch_heavy import run_branch_heavy_subtest


def run_single_core_suite(target_duration_per_subtest: float = 1.0) -> Tuple[List[SubtestResult], float, float]:
    """
    Executes the 7 subtests sequentially on a single thread (main core).
    Uses threadpoolctl to strictly isolate BLAS/OpenBLAS operations to 1 worker thread.
    Returns (list of SubtestResult, composite_single_core_score, total_single_core_time).
    """
    start_time = time.perf_counter()
    results: List[SubtestResult] = []

    # Scope BLAS/OpenBLAS thread limits strictly to single-core execution context
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        results.append(run_integer_subtest(target_duration=target_duration_per_subtest))
        results.append(run_floating_point_subtest(target_duration=target_duration_per_subtest))
        results.append(run_matrix_subtest(target_duration=target_duration_per_subtest))
        results.append(run_vector_simd_subtest(target_duration=target_duration_per_subtest))
        results.append(run_compression_subtest(target_duration=target_duration_per_subtest))
        results.append(run_hashing_subtest(target_duration=target_duration_per_subtest))
        results.append(run_branch_heavy_subtest(target_duration=target_duration_per_subtest))

    total_time = time.perf_counter() - start_time

    # Composite single core score = Geometric Mean of normalized subtest scores
    passed_scores = [r.score for r in results if r.validation_passed]
    composite_score = geometric_mean(passed_scores)

    return results, composite_score, round(total_time, 6)
