"""
db/crud.py

Optional cloud persistence.

Every function returns None (rather than raising) when Supabase is not
configured, so the benchmark pipeline never fails because of a missing API key.

Results are only uploaded when the validity gate marked the run `valid` and the
environment fingerprint is recorded, because a leaderboard of incomparable
numbers is worse than no leaderboard.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from db.client import get_client, is_configured

logger = logging.getLogger("BenchMind.DB.CRUD")


def save_system_profile(system_info: Dict[str, Any],
                        environment: Optional[Dict[str, Any]] = None) -> Optional[str]:
    client = get_client()
    if client is None:
        return None

    gpus = system_info.get("gpus", []) or []
    environment = environment or {}

    try:
        response = client.table("systems").insert({
            "cpu_name": system_info.get("cpu"),
            "physical_cores": system_info.get("physical_cores"),
            "logical_cores": system_info.get("logical_cores"),
            "ram_gb": system_info.get("ram"),
            "primary_gpu": gpus[0] if gpus else None,
            "secondary_gpu": gpus[1] if len(gpus) > 1 else None,
            "os_info": system_info.get("os"),
            "architecture": system_info.get("architecture"),
            "fingerprint_hash": environment.get("fingerprint_hash"),
        }).execute()
        return response.data[0]["id"]
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not save system profile: %s", e)
        return None


def save_benchmark_result(system_id: str, result: Dict[str, Any]) -> Optional[str]:
    """Upload one run. Refuses runs the validity gate did not mark valid."""
    client = get_client()
    if client is None or not system_id:
        return None

    validity = result.get("validity", {}) or {}
    if validity.get("verdict") not in (None, "valid"):
        logger.info("Run verdict is %s; not uploading to the shared leaderboard.",
                    validity.get("verdict"))
        return None

    cpu = result.get("cpu_benchmark", {}) or {}
    environment = result.get("environment", {}) or {}
    stability = result.get("stability", {}) or {}
    throttle = result.get("throttling", {}) or {}

    gpu_score = 0.0
    gpu = result.get("gpu_benchmark") or {}
    devices = gpu.get("devices") if isinstance(gpu, dict) else gpu
    if devices:
        try:
            gpu_score = max(float(d.get("gpu_score") or 0) for d in devices)
        except (ValueError, TypeError):
            gpu_score = 0.0

    try:
        response = client.table("benchmark_runs").insert({
            "system_id": system_id,
            "mode": cpu.get("mode"),
            "baseline_version": cpu.get("baseline_version"),
            "fingerprint_hash": environment.get("fingerprint_hash"),
            "cpu_index": cpu.get("cpu_index"),
            "cpu_index_ci_pct": cpu.get("cpu_index_ci_pct"),
            "cpu_single_score": cpu.get("single_core_score"),
            "cpu_multi_score": cpu.get("multi_core_score"),
            "gpu_score": gpu_score,
            "stability_score": stability.get("overall_score"),
            "throttled": bool(throttle.get("throttling_detected")),
            "peak_cpu_temp": (cpu.get("telemetry_summary") or {}).get("max_cpu_temp"),
            "summary_text": (result.get("summary") or {}).get("headline"),
            "status": "finished",
        }).execute()
        return response.data[0]["id"]
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not save benchmark result: %s", e)
        return None


def save_timeline_logs(benchmark_id: str, timeline: Dict[str, List]) -> bool:
    """
    Upload the telemetry timeline.

    1.x expected a list of per-sample dicts but the telemetry layer produces
    column arrays, so this never matched the real data shape. It now accepts
    the column format that get_logs_format() actually returns.
    """
    client = get_client()
    if client is None or not benchmark_id or not timeline:
        return False

    elapsed = timeline.get("elapsed") or []
    records = []
    for i, offset in enumerate(elapsed):
        def at(key):
            series = timeline.get(key) or []
            return series[i] if i < len(series) else None

        records.append({
            "benchmark_id": benchmark_id,
            "time_offset": offset,
            "cpu_usage": at("cpu"),
            "ram_usage": at("ram"),
            "cpu_temp": at("cpu_temp"),
            "gpu_temp": at("gpu_temp"),
            "cpu_freq": at("cpu_freq"),
        })

    if not records:
        return False

    try:
        # Chunked so a long run does not exceed the request size limit.
        for start in range(0, len(records), 500):
            client.table("benchmark_logs").insert(records[start:start + 500]).execute()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not save timeline logs: %s", e)
        return False


def get_recent_benchmarks(limit: int = 10,
                          fingerprint_hash: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Recent runs from the shared store.

    When `fingerprint_hash` is given, only runs from a matching software stack
    are returned, which is the only case where the comparison is meaningful.
    """
    client = get_client()
    if client is None:
        return []

    try:
        query = (client.table("benchmark_runs")
                 .select("id, cpu_index, cpu_index_ci_pct, cpu_single_score, "
                         "cpu_multi_score, gpu_score, stability_score, throttled, "
                         "created_at, systems(cpu_name, primary_gpu)"))
        if fingerprint_hash:
            query = query.eq("fingerprint_hash", fingerprint_hash)
        response = query.order("created_at", desc=True).limit(limit).execute()
        return response.data or []
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch recent benchmarks: %s", e)
        return []


def upload_run(result: Dict[str, Any]) -> Optional[str]:
    """Convenience wrapper: profile, run and timeline in one call."""
    if not is_configured():
        return None
    system_id = save_system_profile(result.get("system", {}), result.get("environment"))
    if not system_id:
        return None
    run_id = save_benchmark_result(system_id, result)
    if run_id:
        save_timeline_logs(run_id, result.get("timeline", {}))
    return run_id
