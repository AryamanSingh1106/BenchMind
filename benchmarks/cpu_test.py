"""
benchmarks/cpu_test.py

Backwards-compatible entry point for the CPU benchmark.

Existing callers (the API, the UI, old scripts) keep working. The returned dict
contains every 1.x key plus the 2.0 additions, so nothing that read
`single_core_score` or `cpu_index` breaks.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, Callable, Dict, Optional

from benchmarks.cpu.cpu_suite import run_full_cpu_suite
from benchmarks.cpu.integer import setup as _integer_setup, run as _integer_run


def warmup_cpu(seconds: float = 2.0) -> Dict[str, Any]:
    """
    Bring the CPU out of an idle clock state before measuring.

    Uses monotonic only as a wall-clock deadline for the warmup loop; no
    performance number is derived from it.
    """
    ctx = _integer_setup(0.5)
    deadline = time.monotonic() + seconds
    iterations = 0
    while time.monotonic() < deadline:
        _integer_run(ctx)
        iterations += 1
    return {"warmup_seconds": seconds, "warmup_iterations": iterations}


def run_cpu_test(mode: str = "standard",
                 progress: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """Run the CPU suite and return a result dict with 1.x-compatible keys."""
    suite = run_full_cpu_suite(mode=mode, progress=progress)

    return {
        # --- 1.x keys, preserved ---
        "single_core_score": round(suite["single_core_score"]),
        "multi_core_score": round(suite["multi_core_score"]),
        "single_core_time": suite["single_core_time"],
        "multi_core_time": suite["multi_core_time"],
        "cores_used": suite["cores_used"],
        "cpu_index": round(suite["cpu_index"]),
        "category_scores": suite["category_scores"],
        "subtests": suite["subtests"],
        "telemetry_summary": suite["telemetry_summary"],

        # --- 2.0 additions ---
        "cpu_index_ci_pct": suite["cpu_index_ci_pct"],
        "single_core_ci_pct": suite["single_core_ci_pct"],
        "multi_core_ci_pct": suite["multi_core_ci_pct"],
        "multi_core_scaling": suite["multi_core_scaling"],
        "schema_version": suite["schema_version"],
        "mode": suite["mode"],
        "baseline_version": suite["baseline_version"],
        "environment": suite["environment"],
        "isolation": suite["isolation"],
        "multi_core_meta": suite["multi_core_meta"],
        "telemetry": suite["telemetry"],
        "total_suite_time": suite["total_suite_time"],
    }


if __name__ == "__main__":
    multiprocessing.freeze_support()
    import json
    result = run_cpu_test(mode="quick")
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("subtests", "telemetry")},
        indent=2, default=str))
