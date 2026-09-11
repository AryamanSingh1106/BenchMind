"""
Integer ALU throughput.

BenchMind 1.x used a pure-Python bit-rotation loop plus a Python-list Sieve of
Eratosthenes. That measured the CPython interpreter, not the integer units:
the score moved when you upgraded Python without touching the hardware.

2.0 splits the concern in two:
  * this module      -> int64 SIMD ALU work through NumPy, counted in the CPU Index
  * interpreter.py   -> the old-style pure-Python loop, reported but NOT counted

Working set is sized to stay inside L2 on a typical core, so the result
reflects ALU throughput rather than memory bandwidth. All masks are chosen so
no int64 operation can overflow, which keeps the checksum exactly reproducible
across platforms.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

ELEMENTS = 65_536           # 512 KB per int64 array
LOOPS = 120
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
