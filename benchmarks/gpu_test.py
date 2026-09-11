"""
benchmarks/gpu_test.py

Backwards-compatible entry point for the GPU benchmark.

1.x returned a bare list of {gpu_name, gpu_score, execution_time}. Callers that
expect a list still get one from `run_gpu_test()`. Use `run_gpu_suite()` from
benchmarks.gpu.gpu_suite for the full multi-workload result.
"""

from __future__ import annotations

from typing import Any, Dict, List

from benchmarks.gpu.gpu_suite import run_gpu_suite


def run_gpu_test() -> List[Dict[str, Any]]:
    """Legacy shape: a list of per-device summaries."""
    suite = run_gpu_suite()
    return [
        {
            "gpu_name": d.get("gpu_name"),
            "gpu_score": d.get("gpu_score", 0.0),
            "gpu_score_ci_pct": d.get("gpu_score_ci_pct", 0.0),
            "workloads_passed": d.get("workloads_passed", 0),
        }
        for d in suite.get("devices", [])
    ]


def clean_gpu_name(name: str) -> str:
    """Preserved for compatibility; now generalized across vendor codenames."""
    from monitoring.system_info import normalize_device_name
    return normalize_device_name(name)


if __name__ == "__main__":
    import json
    print(json.dumps(run_gpu_suite(), indent=2, default=str))
