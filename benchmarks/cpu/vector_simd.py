import numpy as np
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def vector_simd_workload(vector_size: int = 2_000_000, loops: int = 25) -> Tuple[Dict[str, Any], float]:
    """
    Vectorized NumPy/SIMD-capable Workload:
    - Contiguous array Fused Multiply-Add (FMA) & Vector Dot Product
    - Vector size: 2,000,000 float32 elements
    """
    rng = np.random.default_rng(999)
    v1 = rng.uniform(0.1, 1.0, size=vector_size).astype(np.float32)
    v2 = rng.uniform(0.1, 1.0, size=vector_size).astype(np.float32)
    v3 = rng.uniform(0.01, 0.5, size=vector_size).astype(np.float32)
    res = np.zeros(vector_size, dtype=np.float32)

    # Vectorized FMA loop (2 FLOPS per element per loop)
    for _ in range(loops):
        res = v1 * v2 + v3

    # Dot product (2 FLOPS per element)
    dot_val = float(np.dot(v1, v2))

    total_flops = (2.0 * vector_size * loops) + (2.0 * vector_size)

    output = {
        "res_sum": float(np.sum(res)),
        "dot_val": dot_val,
        "vector_size": vector_size,
        "loops": loops
    }

    # Scale total ops to GigaFLOPS (GFLOPS)
    return output, total_flops / 1e9


def validate_vector_simd_workload(output: Dict[str, Any]) -> bool:
    """
    Validate vectorized workload output:
    Checks result array sum and dot product bounds outside the timed block.
    """
    if not isinstance(output, dict):
        return False
    res_sum = output.get("res_sum")
    dot_val = output.get("dot_val")

    if res_sum is None or dot_val is None:
        return False
    if np.isnan(res_sum) or np.isinf(res_sum):
        return False
    if np.isnan(dot_val) or np.isinf(dot_val):
        return False
    if res_sum <= 0 or dot_val <= 0:
        return False
    return True


def run_vector_simd_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Vectorized NumPy/SIMD-Capable",
        category="vector_simd",
        workload_profile="vectorized",
        raw_metric_name="GFLOPS",
        workload_fn=vector_simd_workload,
        validate_fn=validate_vector_simd_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
