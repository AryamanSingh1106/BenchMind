"""
Integer ALU throughput.

BenchMind 1.x used a pure-Python bit-rotation loop plus a Python-list Sieve of
Eratosthenes. That measured the CPython interpreter, not the integer units:
the score moved when you upgraded Python without touching the hardware.

2.0 splits the concern in two:
  * this module      -> int64 SIMD ALU work through NumPy, counted in the CPU Index
  * interpreter.py   -> the old-style pure-Python loop, reported but NOT counted

Working set sizing (revised in 2.1.0)
-------------------------------------
This workload keeps four int64 arrays live. At the original 65,536 elements
that is exactly 2.00 MB, which is exactly the L2 capacity of a Raptor Lake
P-core. Sitting precisely on the cache cliff is the worst possible place to
be: any small change in what else is resident pushes data in and out of L2,
and the measurement swings wildly as a result.

Measured on an i5-13450HX over five calibration passes:

    65,536 elements (2.00 MB, at the boundary)   spread 15.57%
    32,768 elements (1.00 MB, comfortably in)    see CHANGELOG

The loop count is raised to compensate, so repetition duration is unchanged
and only the cache behaviour differs. All masks are chosen so no int64
operation can overflow, which keeps the checksum exactly reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# 32,768 int64 elements x 4 live arrays = 1.00 MB, half of a Raptor Lake
# P-core's 2 MB L2. Leaves headroom rather than balancing on the cliff.
ELEMENTS = 32_768
# 500 loops keeps a repetition above the 20 ms stability floor even on fast
# hardware. Halving the working set without raising this would have pushed
# repetitions down to ~13 ms, trading one source of noise for another.
LOOPS = 500
VALUE_MASK = 0x0003_FFFF_FFFF_FFFF   # ~2^50, so (x << 7) stays inside int64
SEED_BASE = 12345
SEED_ODD = 67890


def _make_arrays(n: int):
    base = np.random.default_rng(SEED_BASE).integers(
        1, VALUE_MASK, size=n, dtype=np.int64)
    odd = np.random.default_rng(SEED_ODD).integers(
        1, 2**30, size=n, dtype=np.int64)
    return base, odd


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # SCALE TIME, NOT WORKING SET. The array size is fixed so the workload
    # stays L2-resident in every mode; `scale` only changes how many passes
    # are made. Shrinking the buffer instead would change which cache level
    # the workload lives in, and quick-mode scores would measure something
    # different from standard-mode scores.
    n = ELEMENTS
    base, odd = _make_arrays(n)
    return {
        "n": n,
        "loops": max(1, int(LOOPS * scale)),
        "base": base,
        "odd": odd,
        "acc": np.empty(n, dtype=np.int64),
        "tmp": np.empty(n, dtype=np.int64),
    }


def _kernel(acc: np.ndarray, tmp: np.ndarray, odd: np.ndarray, loops: int) -> None:
    """Six int64 operations per element per loop, fully in place."""
    for _ in range(loops):
        np.left_shift(acc, 7, out=tmp)
        np.right_shift(acc, 5, out=acc)
        np.bitwise_or(tmp, acc, out=acc)
        np.bitwise_xor(acc, odd, out=acc)
        np.add(acc, odd, out=acc)
        np.bitwise_and(acc, VALUE_MASK, out=acc)


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    acc, tmp = ctx["acc"], ctx["tmp"]
    n, loops = ctx["n"], ctx["loops"]

    np.copyto(acc, ctx["base"])
    _kernel(acc, tmp, ctx["odd"], loops)

    checksum = int(np.bitwise_xor.reduce(acc))
    total_ops = 6.0 * n * loops
    return {"checksum": checksum, "n": n, "loops": loops}, total_ops / 1e6


_REFERENCE_CACHE: Dict[Tuple[int, int], int] = {}


def _reference_checksum(n: int, loops: int) -> int:
    base, odd = _make_arrays(n)
    acc = base.copy()
    tmp = np.empty(n, dtype=np.int64)
    _kernel(acc, tmp, odd, loops)
    return int(np.bitwise_xor.reduce(acc))


def validate(output: Dict[str, Any]) -> bool:
    """
    Integer arithmetic is exact, so the checksum is deterministic for a given
    (n, loops). The reference is recomputed once per shape, outside the timed
    region, rather than hardcoded as a magic constant.
    """
    if not isinstance(output, dict):
        return False
    checksum, n, loops = output.get("checksum"), output.get("n"), output.get("loops")
    if not isinstance(checksum, int) or n is None or loops is None:
        return False

    key = (int(n), int(loops))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = _reference_checksum(*key)
    return checksum == _REFERENCE_CACHE[key]
