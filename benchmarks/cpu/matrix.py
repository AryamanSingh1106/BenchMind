import numpy as np
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def matrix_workload(size: int = 768, loops: int = 3) -> Tuple[Dict[str, Any], float]:
    """
    Deterministic Matrix Multiplication (C = A x B)
    - Mixed compute + memory-hierarchy workload
    - Size: 768x768 float64 matrices (3 loops)
    - FLOP Count: loops * 2 * N^3 floating point operations
    """
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(67890)

    A = rng_a.uniform(0.1, 2.0, size=(size, size)).astype(np.float64)
    B = rng_b.uniform(0.1, 2.0, size=(size, size)).astype(np.float64)

    C = A
    for _ in range(loops):
        C = np.matmul(A, B)

    # loops * 2 * N^3 FLOPS
    total_flops = float(loops) * 2.0 * (size ** 3)

    output = {
        "matrix_sum": float(np.sum(C)),
        "matrix_trace": float(np.trace(C)),
        "size": size,
        "c_00": float(C[0, 0]),
        "c_mid": float(C[size // 2, size // 2])
    }

    # Scale total ops to GigaFLOPS (GFLOPS)
    return output, total_flops / 1e9


def validate_matrix_workload(output: Dict[str, Any]) -> bool:
    """
    Validate matrix multiplication output outside the timed block:
    Checks trace, element sum, and finite value bounds.
    """
    if not isinstance(output, dict):
        return False

    matrix_sum = output.get("matrix_sum")
    matrix_trace = output.get("matrix_trace")
    c_00 = output.get("c_00")

    if matrix_sum is None or matrix_trace is None or c_00 is None:
        return False
    if np.isnan(matrix_sum) or np.isinf(matrix_sum):
        return False
    if np.isnan(matrix_trace) or np.isinf(matrix_trace):
        return False
    if matrix_sum <= 0 or matrix_trace <= 0:
        return False
    return True


def run_matrix_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Matrix Compute (Deterministic A x B)",
        category="matrix",
        workload_profile="mixed",
        raw_metric_name="GFLOPS",
        workload_fn=matrix_workload,
        validate_fn=validate_matrix_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
