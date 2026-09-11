"""
ai/analysis.py

Two diagnostics that turn BenchMind from a score generator into something that
answers a question.

1. THERMAL AND POWER THROTTLE DETECTION
   Splits the run window into quarters and compares opening against closing
   clock speed and package power, correlated with the temperature rise.

   Revised in 2.0.1 after a false positive on a mobile i5-13450HX. The 2.0.0
   rules fired on peak temperature alone at a threshold of 85 C, and produced
   the self-contradicting verdict:

       "Throttling detected (low confidence). Sustained 100.0% of opening
        clock speed, package temperature rose 6.1 C to a peak of 88.0 C."

   Two things were wrong. A mobile HX part sitting at 88 C under sustained
   all-core load is behaving normally, not throttling; and the clock series
   came from psutil, which on Windows returns a constant, so "held 100%" meant
   "we never had a clock signal" rather than "the clock held".

   The rules now are:
     * a throttle verdict REQUIRES a measured drop in clock or package power
     * temperature alone never produces a verdict, only a note
     * a clock series with no variance is treated as absent, not as steady
     * the threshold is 95 C, and is relative to the part where known

2. ROOFLINE / BOTTLENECK ANALYSIS
   Every workload declares its arithmetic intensity in the registry. Plotting
   achieved throughput against arithmetic intensity says whether the machine
   ran out of compute or ran out of memory bandwidth.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.Analysis")

# A throttle verdict requires one of these to be exceeded. Temperature is
# corroborating evidence only; it is never sufficient on its own.
CLOCK_DROP_THROTTLE_PCT = 7.0
POWER_DROP_THROTTLE_PCT = 12.0

# Raised from 85. Mobile H- and HX-class parts routinely sustain high 80s
# under all-core load by design; that is the cooling solution working at its
# operating point, not a fault.
HOT_TEMP_C = 95.0
WARM_TEMP_C = 85.0

# A clock series flatter than this is a nominal value being reported as if it
# were live, not a genuinely steady clock. See monitoring/temp_reader.py.
MIN_CLOCK_VARIATION_PCT = 0.5


def _quartiles(values: List[Any]) -> List[List[Any]]:
    clean = [v for v in values if v is not None]
    if len(clean) < 8:
        return []
    q = len(clean) // 4
    return [clean[0:q], clean[q:2 * q], clean[2 * q:3 * q], clean[3 * q:]]


def _series_is_usable(values: List[float]) -> bool:
    """
    A series with effectively no variance is not a measurement.

    psutil on Windows reports the registry's nominal base clock, so every
    sample is identical. Treating that as "the clock held steady" produced the
    false positive this function now guards against.
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 8:
        return False
    mean = statistics.fmean(clean)
    if mean <= 0:
        return False
    spread = (max(clean) - min(clean)) / mean * 100.0
    return spread >= MIN_CLOCK_VARIATION_PCT


def _quartile_drop(values: List[float]) -> Optional[Dict[str, Any]]:
    """Opening versus closing quartile means, as a percentage drop."""
    quarters = _quartiles(values)
    if not quarters:
        return None
    means = [round(statistics.fmean(q), 1) for q in quarters]
    opening, closing = means[0], means[-1]
    if opening <= 0:
        return None
    return {
        "quartile_means": means,
        "opening": opening,
        "closing": closing,
        "drop_pct": round((opening - closing) / opening * 100.0, 2),
    }


def analyze_throttling(telemetry: Dict[str, List]) -> Dict[str, Any]:
    """
    Detect thermal or power throttling from a telemetry window.

    Requires a usable clock or package-power series. Without one it reports
    that throttling could not be assessed, and says which source was missing,
    rather than inferring a verdict from temperature.
    """
    raw_freqs = telemetry.get("cpu_freq", []) or []
    raw_power = telemetry.get("cpu_power", []) or []
    freqs = [f for f in raw_freqs if f is not None]
    powers = [p for p in raw_power if p is not None]
    temps = [t for t in telemetry.get("cpu_temp", []) if t is not None]
    elapsed = telemetry.get("elapsed", []) or []

    clock_usable = _series_is_usable(freqs)
    power_usable = _series_is_usable(powers)

    result: Dict[str, Any] = {
        "throttling_detected": False,
        "confidence": "none",
        "mechanism": None,               # thermal | power | None
        "performance_retention_pct": None,
        "clock_drop_pct": None,
        "power_drop_pct": None,
        "temp_rise_c": None,
        "peak_temp_c": max(temps) if temps else None,
        "onset_seconds": None,
        "frequency_source_available": bool(freqs),
        "frequency_signal_usable": clock_usable,
        "power_source_available": bool(powers),
        "temperature_source_available": bool(temps),
        "quartile_clocks_mhz": [],
        "quartile_power_w": [],
        "quartile_temps_c": [],
        "summary": "",
        "notes": [],
    }

    if freqs and not clock_usable:
        result["notes"].append(
            "The clock series has no variance, which means it is a nominal value "
            "rather than a live reading. On Windows that is psutil reporting the "
            "registry base clock. Run LibreHardwareMonitor with its web server "
            "enabled to get real per-core clocks."
        )

    clock = _quartile_drop(freqs) if clock_usable else None
    power = _quartile_drop(powers) if power_usable else None
    temp = _quartile_drop(temps) if temps else None

    if clock:
        result["quartile_clocks_mhz"] = clock["quartile_means"]
        result["clock_drop_pct"] = clock["drop_pct"]
        result["performance_retention_pct"] = round(100.0 - max(0.0, clock["drop_pct"]), 2)
    if power:
        result["quartile_power_w"] = power["quartile_means"]
        result["power_drop_pct"] = power["drop_pct"]
    if temp:
        result["quartile_temps_c"] = temp["quartile_means"]
        result["temp_rise_c"] = round(temp["quartile_means"][-1] - temp["quartile_means"][0], 1)

    # Onset: first sample where the clock falls below the opening quartile
    # average by more than the threshold.
    if clock and elapsed and len(elapsed) == len(raw_freqs):
        threshold = clock["opening"] * (1.0 - CLOCK_DROP_THROTTLE_PCT / 100.0)
        for i, f in enumerate(raw_freqs):
            if f is not None and f < threshold:
                result["onset_seconds"] = round(float(elapsed[i]), 1)
                break

    clock_drop = result["clock_drop_pct"] or 0.0
    power_drop = result["power_drop_pct"] or 0.0
    peak = result["peak_temp_c"]

    clock_throttled = clock_drop >= CLOCK_DROP_THROTTLE_PCT
    power_throttled = power_drop >= POWER_DROP_THROTTLE_PCT

    if clock_throttled or power_throttled:
        result["throttling_detected"] = True
        result["mechanism"] = "thermal" if (peak is not None and peak >= WARM_TEMP_C) else "power"
        if clock_throttled and power_throttled:
            result["confidence"] = "high"
        elif peak is not None and peak >= WARM_TEMP_C:
            result["confidence"] = "high"
        else:
            result["confidence"] = "medium"
    elif not clock_usable and not power_usable:
        result["confidence"] = "unmeasured"

    if peak is not None and peak >= HOT_TEMP_C and not result["throttling_detected"]:
        result["notes"].append(
            f"Peak package temperature reached {peak} C without a measurable drop "
            "in clock or power. The cooling is at its limit but the chip is still "
            "holding its operating point."
        )

    result["summary"] = _throttle_summary(result)
    return result


def _throttle_summary(r: Dict[str, Any]) -> str:
    peak = r.get("peak_temp_c")
    rise = r.get("temp_rise_c")

    if r["confidence"] == "unmeasured":
        base = ("Throttling could not be assessed: no usable clock or package-power "
                "signal was available")
        if peak is not None:
            base += f". Package temperature peaked at {peak} C"
        return base + "."

    if not r["throttling_detected"]:
        parts = []
        if r.get("clock_drop_pct") is not None:
            drop = r["clock_drop_pct"]
            if drop > 0:
                parts.append(f"clock held within {drop:.1f}% across the run")
            else:
                parts.append(f"clock rose {abs(drop):.1f}% across the run")
        if r.get("power_drop_pct") is not None:
            parts.append(f"package power within {abs(r['power_drop_pct']):.1f}%")
        if peak is not None:
            parts.append(f"peaking at {peak} C")
        return "No throttling detected" + (": " + ", ".join(parts) if parts else "") + "."

    parts = []
    retention = r.get("performance_retention_pct")
    if retention is not None:
        parts.append(f"sustained {retention}% of opening clock speed")
    if r.get("power_drop_pct"):
        parts.append(f"package power fell {r['power_drop_pct']:.1f}%")
    if rise is not None and peak is not None:
        parts.append(f"temperature rose {rise} C to a peak of {peak} C")
    if r.get("onset_seconds") is not None:
        parts.append(f"onset at {r['onset_seconds']}s")

    mechanism = r.get("mechanism") or "unknown"
    return (f"{mechanism.title()} throttling detected ({r['confidence']} confidence): "
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
