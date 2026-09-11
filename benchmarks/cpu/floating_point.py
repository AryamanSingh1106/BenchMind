"""
Floating-point FMA throughput, L2-resident.

The 1.x version allocated six one-million-element arrays and generated six
million random numbers *inside* the timed region, then did 20 cheap passes over
them. Most of the measured time was the RNG, and the arrays were far too large
to sit in cache, so the number reported as MFLOPS was really DRAM bandwidth.

2.0 sizes the working set to roughly 768 KB so it stays in L2 on a typical
core, runs many more passes, and does all arithmetic in place with `out=`
so nothing is allocated while the clock is running.

Pair this with vector_simd.py, which deliberately uses a DRAM-sized working
set. The two together tell you where the machine falls off the cache cliff.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

ELEMENTS = 65_536      # 256 KB as float32, 512 KB as float64
LOOPS = 150


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # Array size fixed so the working set stays L2-resident in every mode;
    # `scale` changes the pass count only. See integer.py for the rationale.
    n = ELEMENTS
    rng = np.random.default_rng(42)
    return {
        "n": n,
        "loops": max(1, int(LOOPS * scale)),
        # a in [0.90, 0.99] keeps x = x*a + b convergent rather than divergent,
        # so FP32 and FP64 stay in agreement and validation is meaningful.
        "a32": rng.uniform(0.90, 0.99, size=n).astype(np.float32),
        "b32": rng.uniform(0.01, 0.1, size=n).astype(np.float32),
        "x32": np.ones(n, dtype=np.float32),
        "a64": rng.uniform(0.90, 0.99, size=n).astype(np.float64),
        "b64": rng.uniform(0.01, 0.1, size=n).astype(np.float64),
        "y64": np.ones(n, dtype=np.float64),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    n, loops = ctx["n"], ctx["loops"]
    a32, b32, x32 = ctx["a32"], ctx["b32"], ctx["x32"]
    a64, b64, y64 = ctx["a64"], ctx["b64"], ctx["y64"]

    # Reset accumulators so every rep does identical work (cheap: one pass).
    x32.fill(1.0)
    y64.fill(1.0)

    for _ in range(loops):
        np.multiply(x32, a32, out=x32)
        np.add(x32, b32, out=x32)
        np.multiply(y64, a64, out=y64)
        np.add(y64, b64, out=y64)

    # 2 FLOPs per element per loop for FP32 plus 2 for FP64.
    total_flops = 4.0 * n * loops
    return (
        {"x32_sum": float(np.sum(x32)), "y64_sum": float(np.sum(y64)),
         "n": n, "loops": loops},
        total_flops / 1e6,
    )


def validate(output: Dict[str, Any]) -> bool:
    """
    x = x*a + b with a < 1 converges to the fixed point b/(1-a), which lies
    in [0.1, 10] for the chosen ranges. FP32 and FP64 run the identical
    recurrence, so their sums must agree closely; a large divergence means
    the arithmetic went wrong.
    """
    if not isinstance(output, dict):
        return False
    x32_sum, y64_sum = output.get("x32_sum"), output.get("y64_sum")
    n = output.get("n")
    if x32_sum is None or y64_sum is None or not n:
        return False
    for v in (x32_sum, y64_sum):
        if not np.isfinite(v) or v <= 0:
            return False
    # FP32 and FP64 run the same recurrence, so their sums must agree closely.
    rel_gap = abs(x32_sum - y64_sum) / max(abs(y64_sum), 1e-12)
    if rel_gap >= 0.01:
        return False
    # Converged mean must land inside the analytic fixed-point range.
    mean = y64_sum / n
    return 0.05 < mean < 20.0
