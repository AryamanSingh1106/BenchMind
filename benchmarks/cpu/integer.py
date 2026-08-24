import math
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def integer_workload(iterations: int = 300_000) -> Tuple[Dict[str, Any], float]:
    """
    Integer compute workload:
    - Bitwise shifts, bit rotation, 64-bit mask ops
    - Sieve of Eratosthenes prime generation up to N=500,000
    """
    # 1. Bit manipulation loop
    accumulator = 0x123456789ABCDEF0
    for i in range(1, iterations + 1):
        accumulator = ((accumulator << 7) | (accumulator >> 57)) & 0xFFFFFFFFFFFFFFFF
        accumulator ^= (i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        accumulator = (accumulator + (i & 0xFFFF)) & 0xFFFFFFFFFFFFFFFF

    # 2. Sieve of Eratosthenes up to N=500,000
    n = 500_000
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if sieve[p]:
            for multiple in range(p * p, n, p):
                sieve[multiple] = False

    prime_count = sum(sieve)
    total_ops = (iterations * 10) + n  # Operations performed

    output = {
        "accumulator": accumulator,
        "prime_count": prime_count,
        "sieve_limit": n
    }

    # Scale total ops to Millions (Mops)
    return output, total_ops / 1e6


def validate_integer_workload(output: Dict[str, Any]) -> bool:
    """
    Validate integer output:
    Prime count up to 500,000 is mathematically known to be exactly 41,538.
    Check that accumulator is an integer.
    """
    if not isinstance(output, dict):
        return False
    prime_count = output.get("prime_count")
    accumulator = output.get("accumulator")
    if prime_count != 41538:
        return False
    if not isinstance(accumulator, int):
        return False
    return True


def run_integer_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Integer Compute (Bitwise & Sieve)",
        category="integer",
        workload_profile="compute_bound",
        raw_metric_name="Mops/sec",
        workload_fn=integer_workload,
        validate_fn=validate_integer_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
