"""
ai/analysis.py

Two diagnostics that turn BenchMind from a score generator into something that
answers a question.

1. THERMAL THROTTLE DETECTION
   BenchMind already logged temperature and timestamps; it just never looked at
   them. This module splits the run window into quarters, compares throughput
   and clock speed in the first quarter against the last, and correlates the
   drop with the temperature rise. The output is a sentence like "sustained 87%
   of opening throughput; clock fell 18% as package temperature rose 31 C from
   61 C to 92 C, onset at 94 s".

2. ROOFLINE / BOTTLENECK ANALYSIS
   Every workload declares its arithmetic intensity (work per byte of memory
   traffic) in the registry. Plotting achieved throughput against arithmetic
   intensity tells you whether the machine ran out of compute or ran out of
   memory bandwidth. That is the real payoff of the `workload_profile` field,
   and it answers the question in BenchMind's own philosophy section:
   "what kind of work is this hardware good at?"
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.Analysis")

# A clock drop this large between the start and end of a run is treated as
# throttling rather than noise.
CLOCK_DROP_THROTTLE_PCT = 5.0
PERF_DROP_THROTTLE_PCT = 5.0
HOT_TEMP_C = 85.0


def _quartiles(values: List[Any]) -> List[List[Any]]:
    clean = [v for v in values if v is not None]
    if len(clean) < 8:
        return []
    q = len(clean) // 4
    return [clean[0:q], clean[q:2 * q], clean[2 * q:3 * q], clean[3 * q:]]


def analyze_throttling(telemetry: Dict[str, List]) -> Dict[str, Any]:
    """
    Detect thermal or power throttling from a telemetry window.

    Needs `cpu_freq` and ideally `cpu_temp`. Without a frequency source the
    analysis degrades to temperature-only and says so instead of inventing a
    verdict.
    """
    freqs = [f for f in telemetry.get("cpu_freq", []) if f is not None]
    temps = [t for t in telemetry.get("cpu_temp", []) if t is not None]
    elapsed = telemetry.get("elapsed", []) or []

    result: Dict[str, Any] = {
        "throttling_detected": False,
        "confidence": "none",
        "performance_retention_pct": None,
        "clock_drop_pct": None,
        "temp_rise_c": None,
        "peak_temp_c": max(temps) if temps else None,
        "onset_seconds": None,
        "frequency_source_available": bool(freqs),
        "temperature_source_available": bool(temps),
        "summary": "",
        "quartile_clocks_mhz": [],
        "quartile_temps_c": [],
    }

    if not freqs and not temps:
        result["summary"] = (
            "No frequency or temperature source available, so throttling could "
            "not be assessed. On Windows, run LibreHardwareMonitor with its web "
            "server enabled on port 8085."
        )
        return result

    freq_q = _quartiles(freqs)
    temp_q = _quartiles(temps)

    if freq_q:
        q_means = [round(statistics.fmean(q), 1) for q in freq_q]
        result["quartile_clocks_mhz"] = q_means
        opening, closing = q_means[0], q_means[-1]
        if opening > 0:
            drop = (opening - closing) / opening * 100.0
            result["clock_drop_pct"] = round(drop, 2)
            result["performance_retention_pct"] = round(100.0 - max(0.0, drop), 2)

    if temp_q:
        t_means = [round(statistics.fmean(q), 1) for q in temp_q]
        result["quartile_temps_c"] = t_means
        result["temp_rise_c"] = round(t_means[-1] - t_means[0], 1)

    # Onset: first sample where the clock has fallen more than the threshold
    # below the opening average and stays down.
    if freqs and freq_q and elapsed and len(elapsed) == len(telemetry.get("cpu_freq", [])):
        opening = statistics.fmean(freq_q[0])
        threshold = opening * (1.0 - CLOCK_DROP_THROTTLE_PCT / 100.0)
        for i, f in enumerate(telemetry.get("cpu_freq", [])):
            if f is not None and f < threshold and i < len(elapsed):
                result["onset_seconds"] = round(float(elapsed[i]), 1)
                break

    drop = result.get("clock_drop_pct") or 0.0
    peak = result.get("peak_temp_c")

    if drop >= PERF_DROP_THROTTLE_PCT and peak is not None and peak >= HOT_TEMP_C:
        result["throttling_detected"] = True
        result["confidence"] = "high"
    elif drop >= PERF_DROP_THROTTLE_PCT:
        result["throttling_detected"] = True
        result["confidence"] = "medium"
    elif peak is not None and peak >= HOT_TEMP_C:
        result["throttling_detected"] = True
        result["confidence"] = "low"

    result["summary"] = _throttle_summary(result)
    return result


def _throttle_summary(r: Dict[str, Any]) -> str:
    retention = r.get("performance_retention_pct")
    drop = r.get("clock_drop_pct")
    peak = r.get("peak_temp_c")
    rise = r.get("temp_rise_c")
    onset = r.get("onset_seconds")

    if not r["throttling_detected"]:
        if retention is not None:
            base = f"No throttling detected. Clock held within {abs(drop or 0):.1f}% across the run"
        else:
            base = "No throttling detected"
        if peak is not None:
            base += f", peaking at {peak} C"
        return base + "."

    parts = []
    if retention is not None:
        parts.append(f"Sustained {retention}% of opening clock speed")
    if drop:
        parts.append(f"clock fell {drop}%")
    if rise is not None and peak is not None:
        parts.append(f"package temperature rose {rise} C to a peak of {peak} C")
    if onset is not None:
        parts.append(f"onset at {onset}s")

    confidence = r.get("confidence")
    return (f"Throttling detected ({confidence} confidence). "
            + ", ".join(parts) + ".")


# --------------------------------------------------------------------------
# Roofline
# --------------------------------------------------------------------------
def analyze_bottleneck(subtests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classify the machine as compute-limited or bandwidth-limited by comparing
    a cache-resident workload against a DRAM-resident one.

    The key pair is floating_point (L2-resident FMA) versus vector_simd (the
    same kernel shape on arrays far larger than last-level cache). The ratio
    between them is the cache cliff: a large ratio means the memory subsystem
    is the limiter for streaming work.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    for s in subtests:
        if not s.get("validation_passed"):
            continue
        # Prefer single-thread numbers for the roofline; threads confound it.
        if s.get("threads", 1) != 1:
            continue
        by_key[s.get("category", "")] = s

    fp = by_key.get("floating_point")
    vec = by_key.get("vector_simd")
    matrix = by_key.get("matrix")

    points = []
    for s in by_key.values():
        ai = s.get("arithmetic_intensity") or 0.0
        if ai <= 0:
            continue
        points.append({
            "category": s.get("category"),
            "name": s.get("name"),
            "profile": s.get("workload_profile"),
            "arithmetic_intensity": round(ai, 3),
            "throughput": s.get("raw_metric_value"),
            "metric": s.get("raw_metric_name"),
            "working_set_mb": round((s.get("working_set_bytes") or 0) / (1024 * 1024), 2),
        })
    points.sort(key=lambda p: p["arithmetic_intensity"])

    result: Dict[str, Any] = {
        "points": points,
        "cache_cliff_ratio": None,
        "verdict": "insufficient_data",
        "summary": "Not enough passing single-thread subtests to classify the bottleneck.",
        "estimated_memory_bandwidth_gbs": None,
    }

    if vec:
        # vector_simd reports GFLOPS at 2 FLOPs per 16 bytes touched.
        gflops = vec.get("raw_metric_value") or 0.0
        ai = vec.get("arithmetic_intensity") or 0.0
        if gflops > 0 and ai > 0:
            result["estimated_memory_bandwidth_gbs"] = round(gflops / ai, 2)

    if fp and vec:
        fp_gflops = (fp.get("raw_metric_value") or 0.0) / 1000.0   # MFLOPS -> GFLOPS
        vec_gflops = vec.get("raw_metric_value") or 0.0
        if vec_gflops > 0:
            ratio = fp_gflops / vec_gflops
            result["cache_cliff_ratio"] = round(ratio, 2)

            if ratio >= 4.0:
                result["verdict"] = "memory_bandwidth_limited"
                result["summary"] = (
                    f"Cache-resident FP work runs {ratio:.1f}x faster than the same "
                    "kernel on DRAM-sized arrays. This machine is strongly limited by "
                    "memory bandwidth on streaming workloads: faster RAM or better "
                    "cache blocking will help more than more cores."
                )
            elif ratio >= 2.0:
                result["verdict"] = "mixed"
                result["summary"] = (
                    f"Cache-resident FP work runs {ratio:.1f}x faster than DRAM-resident "
                    "work. A normal cache cliff. Compute and bandwidth are reasonably "
                    "balanced for this class of machine."
                )
            else:
                result["verdict"] = "compute_limited"
                result["summary"] = (
                    f"Only a {ratio:.1f}x gap between cache-resident and DRAM-resident FP "
                    "work, so the memory subsystem is keeping up with the cores. "
                    "Compute throughput is the limiter here."
                )

    if matrix and fp:
        matrix_gflops = matrix.get("raw_metric_value") or 0.0
        fp_gflops = (fp.get("raw_metric_value") or 0.0) / 1000.0
        if fp_gflops > 0:
            result["blas_efficiency_ratio"] = round(matrix_gflops / fp_gflops, 2)

    return result


def analyze_scaling(scaling: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn the raw thread-scaling curve into a plain-language reading."""
    if not scaling or not scaling.get("points"):
        return {"summary": "Thread scaling curve was not measured.", "verdict": "unknown"}

    points = scaling["points"]
    max_threads = scaling.get("max_threads", points[-1]["threads"])
    peak_speedup = scaling.get("peak_speedup", 0.0)
    efficiency = scaling.get("efficiency_at_max", 0.0)
    saturation = scaling.get("saturation_threads", 1)

    if efficiency >= 0.85:
        verdict = "near_linear"
        summary = (f"Scaling is close to linear: {peak_speedup:.1f}x on {max_threads} threads "
                   f"({efficiency * 100:.0f}% efficiency). The workload is not contended.")
    elif efficiency >= 0.55:
        verdict = "good"
        summary = (f"{peak_speedup:.1f}x on {max_threads} threads ({efficiency * 100:.0f}% "
                   "efficiency). Typical for SMT plus a mix of core types.")
    elif saturation < max_threads:
        verdict = "saturated_early"
        summary = (f"Throughput stopped improving around {saturation} threads and reached only "
                   f"{peak_speedup:.1f}x at {max_threads}. Adding threads past that point does "
                   "not help; the workload is contending on a shared resource, most likely "
                   "memory bandwidth or last-level cache.")
    else:
        verdict = "poor"
        summary = (f"Only {peak_speedup:.1f}x from {max_threads} threads "
                   f"({efficiency * 100:.0f}% efficiency). Check for background load, "
                   "power limits, or a thermally constrained chassis.")

    return {"verdict": verdict, "summary": summary,
            "saturation_threads": saturation, "efficiency_at_max": efficiency}
