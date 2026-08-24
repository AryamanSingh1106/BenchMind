import numpy as np
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def floating_point_workload(array_size: int = 1_000_000, loops: int = 20) -> Tuple[Dict[str, Any], float]:
    """
    Controlled FP32 and FP64 arithmetic throughput workload:
    - FP32 FMA / Multiply-Add loop: x = x * a + b
    - FP64 FMA / Multiply-Add loop: y = y * c + d
    """
    # Deterministic array generation with fixed seed
    rng = np.random.default_rng(42)
    a32 = rng.uniform(0.5, 1.5, size=array_size).astype(np.float32)
    b32 = rng.uniform(0.01, 0.1, size=array_size).astype(np.float32)
    x32 = np.ones(array_size, dtype=np.float32)

    a64 = rng.uniform(0.5, 1.5, size=array_size).astype(np.float64)
    b64 = rng.uniform(0.01, 0.1, size=array_size).astype(np.float64)
    y64 = np.ones(array_size, dtype=np.float64)

    # Arithmetic throughput loop (2 FLOPS per element per loop for FP32, 2 FLOPS for FP64)
    for _ in range(loops):
        x32 = x32 * a32 + b32
        y64 = y64 * a64 + b64

    # Calculate total floating point operations
    # Each loop does (2 FLOPS * array_size) for FP32 + (2 FLOPS * array_size) for FP64
    total_flops = 4.0 * array_size * loops

    output = {
        "x32_sum": float(np.sum(x32)),
        "y64_sum": float(np.sum(y64)),
        "array_size": array_size,
        "loops": loops
    }

    # Scale total ops to Millions (MFLOPS)
    return output, total_flops / 1e6


def validate_floating_point_workload(output: Dict[str, Any]) -> bool:
    """
    Validate floating point output:
    Checks that outputs are valid finite floats and positive sums.
    """
    if not isinstance(output, dict):
        return False
    x32_sum = output.get("x32_sum")
    y64_sum = output.get("y64_sum")

    if x32_sum is None or y64_sum is None:
        return False
    if np.isnan(x32_sum) or np.isinf(x32_sum):
        return False
    if np.isnan(y64_sum) or np.isinf(y64_sum):
        return False
    if x32_sum <= 0 or y64_sum <= 0:
        return False
    return True


def run_floating_point_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Floating-Point Compute (FP32/FP64 Ops)",
        category="floating_point",
        workload_profile="compute_bound",
        raw_metric_name="MFLOPS",
        workload_fn=floating_point_workload,
        validate_fn=validate_floating_point_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
