"""
benchmarks/cpu/_worker.py

Worker-process side of the multi-core suite.

Two 1.x problems are fixed here.

1. SPAWN COST WAS INSIDE THE MEASUREMENT.
   `run_multi_core_subtest` started its perf_counter before creating the
   ProcessPoolExecutor. On Windows (spawn start method) that means a fresh
   interpreter plus a NumPy import per worker, which can easily cost several
   hundred milliseconds each. Against ~0.3 s of actual work, process startup
   was a large fraction of the "benchmark".

   Now the pool is created and warmed by the caller before any timing starts,
   and `warmup()` below is what the caller pings to prove every worker is alive
   and has NumPy loaded.

2. SETUP WAS INSIDE THE MEASUREMENT.
   Each worker task used to build its own arrays and payloads on every call.
   Now each process builds a workload context exactly once and caches it in
   `_CONTEXTS`, so repeated chunks reuse it and the timed region contains only
   the kernel.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

# Per-process cache of prepared workload contexts, keyed by (workload, scale).
_CONTEXTS: Dict[Tuple[str, float], Any] = {}


def init_worker() -> None:
    """
    Pool initializer. Imports the heavy modules once at process start so the
    cost is paid before the caller begins timing.
    """
    import numpy  # noqa: F401
    from benchmarks.cpu import registry  # noqa: F401


def warmup() -> int:
    """Cheap ping used to confirm a worker process is fully started."""
    return os.getpid()


def prepare(workload_key: str, scale: float = 1.0) -> int:
    """
    Build and cache the workload context in this process. Called once per
    worker before timing starts. Returns the PID so the caller can confirm
    every distinct worker was reached.
    """
    from benchmarks.cpu import registry

    key = (workload_key, float(scale))
    if key not in _CONTEXTS:
        spec = registry.get(workload_key)
        _CONTEXTS[key] = spec.setup_fn(scale)
        # One untimed warmup run to fault in pages and warm caches.
        spec.run_fn(_CONTEXTS[key])
    return os.getpid()


def execute(workload_key: str, scale: float = 1.0, chunk_reps: int = 1) -> Dict[str, Any]:
    """
    Run the workload `chunk_reps` times against the cached context.

    Returns per-chunk timing measured inside the worker with perf_counter, so
    the caller can separate real compute time from dispatch overhead.
    """
    from benchmarks.cpu import registry

    spec = registry.get(workload_key)
    key = (workload_key, float(scale))
    if key not in _CONTEXTS:
        prepare(workload_key, scale)
    ctx = _CONTEXTS[key]

    total_work = 0.0
    last_output = None

    t0 = time.perf_counter()
    for _ in range(max(1, chunk_reps)):
        last_output, work_units = spec.run_fn(ctx)
        total_work += work_units
    elapsed = time.perf_counter() - t0

    return {
        "pid": os.getpid(),
        "work_units": total_work,
        "worker_elapsed": elapsed,
        "valid": bool(spec.validate_fn(last_output)),
    }
