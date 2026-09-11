"""
Single-core CPU suite.

Isolation measures, in order of importance:

1. BLAS thread limiting via threadpoolctl. NumPy's OpenBLAS/MKL backend will
   happily spawn one thread per core inside `np.matmul`, which turns a
   "single-core" matrix score into a multi-core one. This was already correct
   in 1.x and is preserved.

2. CPU affinity pinning (new in 2.0). threadpoolctl stops BLAS from using other
   cores, but nothing stopped the OS from migrating the process between cores
   mid-measurement. On a hybrid CPU that can mean part of a run lands on an
   E-core. Where the platform supports it, the process is pinned to one core
   for the duration and released afterwards.

3. Process priority elevation (new in 2.0), best effort, so a background
   updater is less likely to preempt the measurement.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import psutil
import threadpoolctl

from benchmarks.cpu import registry
from benchmarks.cpu.common import (
    SubtestResult,
    combine_ci_geometric,
    geometric_mean,
    run_timed_subtest,
)

logger = logging.getLogger("BenchMind.CPU.SingleCore")


@contextmanager
def pinned_to_core(core: Optional[int] = None):
    """
    Pin this process to a single logical core, if the platform allows it.

    Not available on macOS and some BSDs, in which case this is a no-op and the
    result carries `core_pinned: False` so the report can say so honestly.
    """
    proc = psutil.Process()
    original: Optional[List[int]] = None
    pinned = False
    try:
        if hasattr(proc, "cpu_affinity"):
            original = proc.cpu_affinity()
            target = core if core is not None else (original[0] if original else 0)
            proc.cpu_affinity([target])
            pinned = True
            logger.info("Pinned single-core suite to logical core %d", target)
    except Exception as e:  # noqa: BLE001
        logger.info("CPU affinity pinning unavailable on this platform: %s", e)
        pinned = False

    try:
        yield pinned
    finally:
        if original is not None:
            try:
                proc.cpu_affinity(original)
            except Exception:  # noqa: BLE001
                pass


@contextmanager
def elevated_priority():
    """Best-effort priority bump. Silently skipped without permission."""
    proc = psutil.Process()
    original = None
    try:
        original = proc.nice()
        if psutil.WINDOWS:
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            proc.nice(max(-5, original - 5))
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not raise process priority: %s", e)
        original = None
    try:
        yield
    finally:
        if original is not None:
            try:
                proc.nice(original)
            except Exception:  # noqa: BLE001
                pass


def run_single_core_suite(
    target_duration_per_subtest: float = 0.5,
    scale: float = 1.0,
    pin_core: bool = True,
    include_runtime_bound: bool = True,
) -> Tuple[List[SubtestResult], float, float, Dict[str, Any]]:
    """
    Execute every registered workload on one thread.

    Returns (results, composite_score, total_time, meta) where meta carries the
    isolation flags that were actually achieved, not the ones requested.
    """
    start_time = time.perf_counter()
    results: List[SubtestResult] = []

    specs = registry.WORKLOADS if include_runtime_bound else registry.INDEX_WORKLOADS

    blas_info: List[Dict[str, Any]] = []
    pinned = False

    with pinned_to_core() if pin_core else _nullcontext() as pin_state:
        pinned = bool(pin_state)
        with elevated_priority():
            # BLAS limited to one thread for the whole single-core window.
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                try:
                    blas_info = threadpoolctl.threadpool_info()
                except Exception:  # noqa: BLE001
                    blas_info = []

                for spec in specs:
                    logger.info("Single-core: %s", spec.name)
                    results.append(
                        run_timed_subtest(
                            spec,
                            scale=scale,
                            target_duration=target_duration_per_subtest,
                        )
                    )

    total_time = time.perf_counter() - start_time

    scored = [r for r in results if r.validation_passed and r.counted_in_index]
    composite = geometric_mean([r.score for r in scored])
    composite_ci = combine_ci_geometric([r.score_ci_pct for r in scored])

    meta = {
        "core_pinned": pinned,
        "blas_threads_limited": True,
        "blas_backends": [
            {
                "library": b.get("internal_api"),
                "version": b.get("version"),
                "num_threads": b.get("num_threads"),
            }
            for b in blas_info
        ],
        "composite_ci_pct": composite_ci,
        "subtests_counted": len(scored),
    }

    return results, composite, round(total_time, 6), meta


class _nullcontext:
    def __enter__(self):
        return False

    def __exit__(self, *exc):
        return False
