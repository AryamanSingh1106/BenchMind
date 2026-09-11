"""
ai/summary_engine.py

Human-readable narrative for a completed benchmark.

1.x used hardcoded thresholds ("multi > 12000 means high performance") that
were tied to nothing in particular, and an `overall` rating computed as
`multi/1000 + stability/10`, which mixes two incompatible units and would
change meaning the moment the baselines were recalibrated.

2.0 states results relative to the documented reference machine, because that
is what a score of 1000 points actually means. Every claim that could be wrong
is hedged by the confidence interval that produced it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Performance tiers expressed as a multiple of the reference machine, so they
# stay meaningful if the baselines are recalibrated.
TIERS = [
    (2.00, "high-end"),
    (1.25, "strong"),
    (0.75, "mainstream"),
    (0.40, "entry-level"),
    (0.00, "low-power"),
]


def _tier(score: float) -> str:
    ratio = score / 1000.0
    for threshold, label in TIERS:
        if ratio >= threshold:
            return label
    return "low-power"


def _with_ci(score: float, ci_pct: Optional[float]) -> str:
    if not ci_pct:
        return f"{score:,.0f}"
    return f"{score:,.0f} +/- {ci_pct:.1f}%"


def generate_summary(
    cpu_result: Dict[str, Any],
    gpu_result: Any = None,
    stability: Any = None,
    throttle: Optional[Dict[str, Any]] = None,
    bottleneck: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Any]] = None,
    validity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the narrative. Accepts the 1.x positional signature so old callers
    keep working, and uses the new analyses when they are supplied.
    """
    insights: List[str] = []
    warnings: List[str] = []

    single = cpu_result.get("single_core_score", 0) or 0
    multi = cpu_result.get("multi_core_score", 0) or 0
    index = cpu_result.get("cpu_index", 0) or 0
    index_ci = cpu_result.get("cpu_index_ci_pct")
    single_ci = cpu_result.get("single_core_ci_pct")
    multi_ci = cpu_result.get("multi_core_ci_pct")

    insights.append(
        f"CPU Index {_with_ci(index, index_ci)} - a {_tier(index)} result, "
        f"where 1000 points equals the BenchMind reference machine."
    )
    insights.append(
        f"Single-core {_with_ci(single, single_ci)}, "
        f"multi-core {_with_ci(multi, multi_ci)}."
    )

    scaling_ratio = cpu_result.get("multi_core_scaling")
    cores = cpu_result.get("cores_used")
    if scaling_ratio and cores:
        insights.append(
            f"Multi-core throughput is {scaling_ratio:.1f}x the single-core result "
            f"across {cores} logical cores."
        )

    # Categories the machine is unusually good or bad at.
    categories = cpu_result.get("category_scores", {}) or {}
    scored = [
        (name, data.get("single_core", 0))
        for name, data in categories.items()
        if isinstance(data, dict) and data.get("counted_in_index") and data.get("single_core", 0) > 0
    ]
    if len(scored) >= 3:
        scored.sort(key=lambda x: x[1], reverse=True)
        best, worst = scored[0], scored[-1]
        insights.append(
            f"Strongest category: {best[0].replace('_', ' ')} ({best[1]:,.0f}). "
            f"Weakest: {worst[0].replace('_', ' ')} ({worst[1]:,.0f})."
        )

    if bottleneck and bottleneck.get("summary"):
        insights.append(bottleneck["summary"])

    if scaling and scaling.get("summary"):
        insights.append(scaling["summary"])

    if throttle:
        insights.append(throttle.get("summary", ""))
        if throttle.get("throttling_detected"):
            warnings.append(
                "Sustained performance is limited by thermals or power. Scores from "
                "a short run will overstate what this machine does under a long load."
            )
        if not throttle.get("temperature_source_available"):
            warnings.append(
                "No temperature source was available, so thermal behaviour is unverified."
            )

    # GPU
    if gpu_result:
        devices = gpu_result.get("devices") if isinstance(gpu_result, dict) else gpu_result
        if devices:
            try:
                best_gpu = max(devices, key=lambda d: d.get("gpu_score", 0))
                insights.append(
                    f"Primary GPU: {best_gpu.get('gpu_name', 'unknown')} "
                    f"scoring {best_gpu.get('gpu_score', 0):,.0f}."
                )
            except (ValueError, TypeError, AttributeError):
                pass

    # Stability
    stability_grade = None
    if isinstance(stability, dict):
        stability_grade = stability.get("grade")
        if stability.get("notes"):
            insights.extend(n for n in stability["notes"] if n)
    elif isinstance(stability, (int, float)):
        stability_grade = "good" if stability >= 75 else "fair"

    # Validity
    if validity:
        verdict = validity.get("verdict")
        if verdict == "invalid":
            warnings.append(
                "Run conditions were unusable. This score should not be compared "
                "against anything, including your own earlier runs."
            )
        elif verdict == "tainted":
            warnings.append(
                "Run conditions were imperfect, so this score is not stored as a "
                "comparison baseline."
            )
        for issue in validity.get("issues", []):
            msg = issue.get("message") if isinstance(issue, dict) else None
            if msg:
                warnings.append(msg)

    rating = f"{_tier(index).title()} System"
    if throttle and throttle.get("throttling_detected"):
        rating += " (thermally limited)"

    return {
        "rating": rating,
        "tier": _tier(index),
        "headline": f"CPU Index {_with_ci(index, index_ci)}",
        "insights": [i for i in insights if i],
        "warnings": warnings,
        "stability_grade": stability_grade,
        "comparable": bool(validity is None or validity.get("verdict") == "valid"),
    }
