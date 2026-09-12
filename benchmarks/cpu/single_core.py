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
from monitoring.topology import get_topology, select_benchmark_core

logger = logging.getLogger("BenchMind.CPU.SingleCore")


@contextmanager
def pinned_to_core(core: Optional[int] = None):
    """
    Pin this process to one deliberately chosen logical core.

    Yields a dict describing what actually happened, so the result records the
    core that was used and why, rather than only that pinning was attempted.
    Affinity is unavailable on macOS and some BSDs; there this is a no-op and
    `pinned` is False.
    """
    proc = psutil.Process()
    original: Optional[List[int]] = None

    selection = select_benchmark_core()
    target = core if core is not None else selection.get("logical_id")

    state: Dict[str, Any] = {
        "pinned": False,
        "logical_id": target,
        "requested_explicitly": core is not None,
        **{k: v for k, v in selection.items() if k != "logical_id"},
    }

    try:
        if hasattr(proc, "cpu_affinity") and target is not None:
            original = proc.cpu_affinity()
            if target not in original:
                logger.info(
                    "Chosen core %d is outside this process's affinity mask; "
                    "falling back to %d.", target, original[0])
                target = original[0]
                state["logical_id"] = target
                state["reason"] = "chosen core was outside the process affinity mask"
            proc.cpu_affinity([target])
            state["pinned"] = True
            logger.info("Pinned single-core suite to logical core %d (%s)",
                        target, state.get("reason", ""))
    except Exception as e:  # noqa: BLE001
        logger.info("CPU affinity pinning unavailable on this platform: %s", e)
        state["pinned"] = False
        state["reason"] = f"affinity unavailable: {e}"

    try:
        yield state
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
    pin_state: Dict[str, Any] = {"pinned": False, "reason": "pinning disabled"}

    with (pinned_to_core() if pin_core else _nullcontext()) as state:
        pin_state = state
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

    topology = get_topology()
    meta = {
        "core_pinned": bool(pin_state.get("pinned")),
        "pinned_logical_core": pin_state.get("logical_id"),
        "pinned_core_type": pin_state.get("core_type"),
        "pinned_core_reason": pin_state.get("reason"),
        "avoided_core_zero": pin_state.get("avoided_core_zero"),
        "smt_sibling_unreserved": pin_state.get("smt_sibling"),
        "topology": topology.to_dict(),
        "topology_summary": topology.describe(),
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
    def __enter__(self) -> Dict[str, Any]:
        return {"pinned": False, "reason": "pinning disabled by caller"}

    def __exit__(self, *exc):
        return False
