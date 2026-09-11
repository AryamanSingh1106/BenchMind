"""
Irregular control flow and random access.

Honest note on what changed. The 1.x version claimed to exercise the branch
predictor, but `np.sum(arr > pivot)` compiles to branchless SIMD and the only
data-dependent control flow in the whole workload was inside `np.sort`. The
score was real, the explanation was not.

2.0 builds a workload that genuinely stresses irregular execution:

  1. comparison sort of unsorted random data (data-dependent branches)
  2. binary search of random keys into a large sorted table
     (unpredictable branch per level, plus a cache miss per level)
  3. random gather through a shuffled index array (pointer-chasing-like,
     defeats the prefetcher)

The category key stays `branch_heavy` for backwards compatibility with stored
results, but the display name and the spec both describe it accurately.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

ELEMENTS = 1_000_000
TABLE_SIZE = 4_000_000
QUERIES = 1_000_000


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # Sizes are fixed. The search table must stay far larger than cache for the
    # binary search to generate real misses, and the gather must stay larger
    # than the prefetcher can hide. Scaling these down would turn the workload
    # into a cache benchmark. `scale` is deliberately ignored.
    n = ELEMENTS
    table_n = TABLE_SIZE
    queries = QUERIES

    rng = np.random.default_rng(777)
    unsorted = rng.integers(0, 2**40, size=n, dtype=np.int64)
    table = np.sort(rng.integers(0, 2**40, size=table_n, dtype=np.int64))
    keys = rng.integers(0, 2**40, size=queries, dtype=np.int64)
    gather_src = rng.uniform(0.0, 1.0, size=table_n).astype(np.float64)
    gather_idx = rng.permutation(table_n)[:queries]

    return {
        "n": n,
        "table_n": table_n,
        "queries": queries,
        "unsorted": unsorted,
        "work": np.empty(n, dtype=np.int64),
        "table": table,
        "keys": keys,
        "gather_src": gather_src,
        "gather_idx": gather_idx,
        "expected_sum": int(np.sum(unsorted)),
        "expected_gather": float(np.sum(gather_src[gather_idx])),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    work = ctx["work"]
    np.copyto(work, ctx["unsorted"])

    # 1. Comparison sort: data-dependent branching inside introsort.
    work.sort()

    # 2. Binary search: ~log2(table_n) unpredictable branches and misses per key.
    positions = np.searchsorted(ctx["table"], ctx["keys"])

    # 3. Random gather: defeats hardware prefetch.
    gathered = ctx["gather_src"][ctx["gather_idx"]]

    n, table_n, queries = ctx["n"], ctx["table_n"], ctx["queries"]
    levels = float(np.log2(max(table_n, 2)))
    total_ops = (n * levels) + (queries * levels) + queries

    return (
        {
            "is_sorted": bool(work[0] <= work[-1] and np.all(work[:-1] <= work[1:])),
            "sorted_sum": int(np.sum(work)),
            "expected_sum": ctx["expected_sum"],
            "position_sum": int(np.sum(positions)),
            "gather_sum": float(np.sum(gathered)),
            "expected_gather": ctx["expected_gather"],
        },
        total_ops / 1e6,
    )


def validate(output: Dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False
    if not output.get("is_sorted"):
        return False
    if output.get("sorted_sum") != output.get("expected_sum"):
        return False
    gathered = output.get("gather_sum")
    expected = output.get("expected_gather")
    if gathered is None or expected is None:
        return False
    return abs(gathered - expected) <= max(1e-6, abs(expected) * 1e-9)
