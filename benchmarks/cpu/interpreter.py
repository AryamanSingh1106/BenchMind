"""
CPython interpreter throughput. Reported, but NOT counted in the CPU Index.

This is the old 1.x integer workload, preserved deliberately and labelled
honestly. It is a pure-Python bit-manipulation loop plus a Python-list Sieve of
Eratosthenes, so its score is dominated by interpreter dispatch cost. Upgrade
from CPython 3.12 to 3.13 and this number moves with no hardware change at all.

That makes it useless as a hardware metric and genuinely useful as a runtime
metric, which is why BenchMind keeps it, flags `measures_runtime=True`, and
excludes it from the composite index.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

ITERATIONS = 120_000
SIEVE_LIMIT = 300_000

# pi(300000) = 25997. Fixed mathematical fact, safe as a constant.
EXPECTED_PRIMES = 25_997


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # Sieve limit fixed so the prime count stays an exact, known constant;
    # only the interpreter loop count scales.
    return {
        "iterations": max(1_000, int(ITERATIONS * scale)),
        "sieve_limit": SIEVE_LIMIT,
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    iterations = ctx["iterations"]
    n = ctx["sieve_limit"]

    accumulator = 0x123456789ABCDEF0
    for i in range(1, iterations + 1):
        accumulator = ((accumulator << 7) | (accumulator >> 57)) & 0xFFFFFFFFFFFFFFFF
        accumulator ^= (i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        accumulator = (accumulator + (i & 0xFFFF)) & 0xFFFFFFFFFFFFFFFF

    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for p in range(2, math.isqrt(n) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(range(p * p, n, p)))
    prime_count = sum(sieve)

    total_ops = (iterations * 10) + n
    return (
        {"accumulator": accumulator, "prime_count": prime_count, "sieve_limit": n},
        total_ops / 1e6,
    )


def validate(output: Dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False
    if not isinstance(output.get("accumulator"), int):
        return False
    if output.get("sieve_limit") == SIEVE_LIMIT:
        return output.get("prime_count") == EXPECTED_PRIMES
    # Scaled run: just check the count is plausible against the prime number theorem.
    n = output.get("sieve_limit") or 0
    count = output.get("prime_count") or 0
    if n < 100:
        return count > 0
    approx = n / math.log(n)
    return 0.7 * approx < count < 1.5 * approx
