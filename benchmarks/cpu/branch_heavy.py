import numpy as np
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def branch_heavy_workload(array_size: int = 500_000) -> Tuple[Dict[str, Any], float]:
    """
    Branch-Heavy Workload:
    - Pseudo-random 50% branch splits (odd/even partitioning, threshold branch, and array sorting)
    - Exercises CPU branch prediction & branch target buffer
    """
    rng = np.random.default_rng(777)
    arr = rng.integers(0, 1_000_000, size=array_size, dtype=np.int64)
    original_sum = int(np.sum(arr))

    # 1. Threshold conditional count (50% branch probability)
    pivot = 500_000
    above_pivot = int(np.sum(arr > pivot))

    # 2. Sorting array (exercises internal comparison branch logic)
    sorted_arr = np.sort(arr)
    sorted_sum = int(np.sum(sorted_arr))

    total_branch_ops = array_size * 4

    output = {
        "is_sorted": bool(np.all(sorted_arr[:-1] <= sorted_arr[1:])),
        "original_sum": original_sum,
        "sorted_sum": sorted_sum,
        "above_pivot": above_pivot,
        "array_size": array_size
    }

    # Scale total ops to Millions (Mops)
    return output, total_branch_ops / 1e6


def validate_branch_heavy_workload(output: Dict[str, Any]) -> bool:
    """
    Validate branch-heavy workload output outside the timed block:
    Verify array is strictly sorted and element sum is perfectly preserved.
    """
    if not isinstance(output, dict):
        return False
    is_sorted = output.get("is_sorted")
    original_sum = output.get("original_sum")
    sorted_sum = output.get("sorted_sum")

    if not is_sorted:
        return False
    if original_sum is None or sorted_sum is None:
        return False
    if original_sum != sorted_sum:
        return False
    return True


def run_branch_heavy_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Branch-Heavy Computation & Sorting",
        category="branch_heavy",
        workload_profile="branch_bound",
        raw_metric_name="Mops/sec",
        workload_fn=branch_heavy_workload,
        validate_fn=validate_branch_heavy_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
