"""
Irregular control flow and random access.

Honest note on what changed. The 1.x version claimed to exercise the branch
predictor, but `np.sum(arr > pivot)` compiles to branchless SIMD and the only
data-dependent control flow in the whole workload was inside `np.sort`. The
score was real, the explanation was not.

2.0 built a workload that genuinely stresses irregular execution:

  1. comparison sort of unsorted random data (data-dependent branches)
  2. binary search of random keys into a large sorted table
     (unpredictable branch per level, plus a cache miss per level)
  3. random gather through a shuffled index array (pointer-chasing-like,
     defeats the prefetcher)

2.1.0 fixed two things that were inflating its run-to-run spread to ~10%
while other workloads managed under 1.5%:

  * `np.searchsorted(...)` and `src[idx]` each ALLOCATED a fresh 8 MB result
    array on every repetition, inside the timed region. The setup/run split was
    supposed to eliminate exactly this, but the allocations were implicit in
    NumPy's return values rather than visible `np.empty` calls, so they were
    easy to miss. Results now go into buffers allocated in `setup`, and the
    binary search is chunked so its intermediate stays cache-resident.

  * The sortedness check and four array-wide reductions ran inside `run`,
    charging verification work to the measurement. `run` now returns the
    buffers by reference and `validate` does the reductions after the clock
    has stopped.

The category key stays `branch_heavy` for backwards compatibility with stored
results, but the display name and the spec both describe it accurately.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

ELEMENTS = 1_000_000
TABLE_SIZE = 4_000_000
QUERIES = 1_000_000

# Binary search is done in chunks so the intermediate result NumPy allocates
# stays small and the allocator reuses the same block every time, instead of
# requesting a fresh 8 MB region on each repetition.
SEARCH_CHUNK = 65_536


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
        "table": table,
        "keys": keys,
        "gather_src": gather_src,
        "gather_idx": gather_idx,
        # Output buffers, allocated once so the timed region allocates nothing.
        "work": np.empty(n, dtype=np.int64),
        "positions": np.empty(queries, dtype=np.intp),
        "gathered": np.empty(queries, dtype=np.float64),
        # References for validation, computed here rather than in the kernel.
        "expected_sum": int(np.sum(unsorted)),
        "expected_gather": float(np.sum(gather_src[gather_idx])),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    work = ctx["work"]
    positions = ctx["positions"]
    gathered = ctx["gathered"]
    table, keys = ctx["table"], ctx["keys"]
    queries = ctx["queries"]

    np.copyto(work, ctx["unsorted"])

    # 1. Comparison sort: data-dependent branching inside introsort.
    work.sort()

    # 2. Binary search: ~log2(table_n) unpredictable branches and cache misses
    #    per key. Chunked so the intermediate stays small and reused.
    for start in range(0, queries, SEARCH_CHUNK):
        end = min(start + SEARCH_CHUNK, queries)
        positions[start:end] = np.searchsorted(table, keys[start:end])

    # 3. Random gather: defeats hardware prefetch. `np.take` with `out=` writes
    #    into the preallocated buffer; `gather_src[gather_idx]` would not.
    np.take(ctx["gather_src"], ctx["gather_idx"], out=gathered)

    n, table_n = ctx["n"], ctx["table_n"]
    levels = float(np.log2(max(table_n, 2)))
    total_ops = (n * levels) + (queries * levels) + queries

    # Buffers returned by reference. Every check and reduction happens in
    # validate(), after the clock has stopped. The `_out` suffix marks them as
    # bulk buffers that the repeatability test compares by other means.
    return (
        {
            "work_out": work,
            "positions_out": positions,
            "gathered_out": gathered,
            "expected_sum": ctx["expected_sum"],
            "expected_gather": ctx["expected_gather"],
            "table_n": table_n,
        },
        total_ops / 1e6,
    )


def validate(output: Dict[str, Any]) -> bool:
    """All verification lives here, outside the timed region."""
    if not isinstance(output, dict):
        return False

    work = output.get("work_out")
    gathered = output.get("gathered_out")
    positions = output.get("positions_out")
    expected_sum = output.get("expected_sum")
    expected_gather = output.get("expected_gather")

    if work is None or gathered is None or positions is None:
        return False
    if expected_sum is None or expected_gather is None:
        return False

    # The sort must have actually sorted, and must not have lost or altered any
    # element: an exact sum over int64 catches both.
    if not bool(np.all(work[:-1] <= work[1:])):
        return False
    if int(np.sum(work)) != expected_sum:
        return False

    # Every search result must be a valid insertion point into the table.
    table_n = output.get("table_n")
    if not table_n:
        return False
    if int(positions.min()) < 0 or int(positions.max()) > table_n:
        return False

    got = float(np.sum(gathered))
    return abs(got - expected_gather) <= max(1e-6, abs(expected_gather) * 1e-9)
