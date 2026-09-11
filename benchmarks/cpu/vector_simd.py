"""
Streaming vector throughput, DRAM-resident.

This workload is deliberately memory bound. Its working set is far larger than
last-level cache, so what it measures is sustained memory bandwidth plus
vectorized execution, not peak SIMD compute. The 1.x docstring claimed SIMD
compute; the profile below now says `memory_bound`, which is the truth.

Comparing this against floating_point.py (L2-resident, same kernel shape) is
what drives the roofline analysis in ai/analysis.py.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

ELEMENTS = 8_000_000     # 32 MB per float32 array, past LLC on most machines
LOOPS = 6


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # The array size is deliberately NOT scaled. This workload only means
    # anything while its working set is larger than last-level cache; shrink
    # it for quick mode and it quietly becomes a cache benchmark reporting a
    # much higher number. `scale` changes the pass count instead.
    n = ELEMENTS
    rng = np.random.default_rng(999)
    return {
        "n": n,
        "loops": max(1, int(LOOPS * scale)),
        "v1": rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        "v2": rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        "v3": rng.uniform(0.01, 0.5, size=n).astype(np.float32),
        "res": np.zeros(n, dtype=np.float32),
        "tmp": np.zeros(n, dtype=np.float32),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    n, loops = ctx["n"], ctx["loops"]
    v1, v2, v3 = ctx["v1"], ctx["v2"], ctx["v3"]
    res, tmp = ctx["res"], ctx["tmp"]

    for _ in range(loops):
        np.multiply(v1, v2, out=tmp)
        np.add(tmp, v3, out=res)

    dot_val = float(np.dot(v1, v2))

    total_flops = (2.0 * n * loops) + (2.0 * n)
    return (
        {"res_sum": float(np.sum(res)), "dot_val": dot_val, "n": n, "loops": loops},
        total_flops / 1e9,
    )


def bytes_per_run(n: int = ELEMENTS, loops: int = LOOPS) -> float:
    """3 float32 reads + 1 write per element per loop, plus the dot product."""
    return (4.0 * 4 * n * loops) + (2.0 * 4 * n)


def validate(output: Dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False
    res_sum, dot_val, n = output.get("res_sum"), output.get("dot_val"), output.get("n")
    if res_sum is None or dot_val is None or not n:
        return False
    if not np.isfinite(res_sum) or not np.isfinite(dot_val):
        return False
    if res_sum <= 0 or dot_val <= 0:
        return False
    # v1*v2 in [0.01, 1.0] plus v3 in [0.01, 0.5]; mean result must land in range.
    mean_res = res_sum / n
    return 0.05 < mean_res < 1.6
