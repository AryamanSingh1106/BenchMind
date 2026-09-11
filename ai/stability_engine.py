"""
ai/stability_engine.py

What "stability" should mean, and what it used to mean.

BenchMind 1.x computed:

    stability = 100 - std(cpu_utilization_samples)

During a BenchMind run, CPU utilization swings from roughly 100/N percent
during the single-core phase to 100 percent during the multi-core phase, by
design. So that formula was mostly measuring BenchMind's own phase
transitions. A run that behaved perfectly could score badly, and a run that
throttled hard could score well.

2.0 measures the two things that actually matter:

  1. REPEATABILITY - how much does the score move when you run it again?
     This is the honest answer to "can I trust this number?"

  2. SUSTAINED PERFORMANCE - did throughput hold up across the run, or did it
     decay? A machine that starts fast and finishes slow is thermally limited,
     which is a hardware property worth reporting.

The legacy `calculate_stability(cpu_log)` entry point is kept so existing
callers do not break, but it now delegates to the utilization-steadiness
metric and is clearly labelled as the weakest of the three signals.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.Stability")


@dataclass
class StabilityReport:
    overall_score: float = 0.0              # 0-100
    grade: str = "unknown"                  # excellent | good | fair | poor | unknown
    repeatability_score: Optional[float] = None
    sustained_score: Optional[float] = None
    utilization_steadiness: Optional[float] = None
    score_spread_pct: Optional[float] = None
    performance_retention_pct: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def repeatability_from_scores(scores: List[float]) -> Dict[str, Any]:
    """
    Given the composite scores of several consecutive runs, report the spread.

    Interpretation used by BenchMind:
        under 2%  -> excellent, differences below 2% between machines are noise
        2-5%      -> good, usable for coarse comparisons
        5-10%     -> fair, something is interfering
        over 10%  -> poor, the measurement is not trustworthy
    """
    clean = [s for s in scores if s and s > 0]
    if len(clean) < 2:
        return {"runs": len(clean), "spread_pct": None, "score": None,
                "note": "Need at least two runs to judge repeatability."}

    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean)
    spread_pct = (sd / mean) * 100.0 if mean else 0.0

    # Map coefficient of variation onto 0-100. 0% spread -> 100, 10% -> 0.
    score = max(0.0, min(100.0, 100.0 - spread_pct * 10.0))

    return {
        "runs": len(clean),
        "mean_score": round(mean, 2),
        "std_dev": round(sd, 2),
        "spread_pct": round(spread_pct, 2),
        "min_score": round(min(clean), 2),
        "max_score": round(max(clean), 2),
        "score": round(score, 2),
    }


def utilization_steadiness(cpu_log: List[float]) -> Optional[float]:
    """
    Legacy-style metric, retained but demoted.

    Now computed on the *coefficient of variation* rather than raw standard
    deviation, so it is at least scale-free, and documented as weak evidence:
    a multi-phase benchmark is supposed to vary its CPU load.
    """
    clean = [v for v in cpu_log if v is not None]
    if len(clean) < 2:
        return None
    mean = statistics.fmean(clean)
    if mean <= 0:
        return 0.0
    cv = statistics.stdev(clean) / mean
    return round(max(0.0, min(100.0, (1.0 - cv) * 100.0)), 2)


def build_stability_report(
    run_scores: Optional[List[float]] = None,
    cpu_log: Optional[List[float]] = None,
    throttle_report: Optional[Dict[str, Any]] = None,
) -> StabilityReport:
    """
    Combine every available stability signal into one report.

    Weights reflect how much each signal is actually worth:
        repeatability      50%   (strongest evidence)
        sustained perf     35%   (thermal behaviour)
        utilization steady 15%   (weakest, kept for continuity)
    """
    report = StabilityReport()
    components: List[tuple] = []

    if run_scores:
        rep = repeatability_from_scores(run_scores)
        if rep.get("score") is not None:
            report.repeatability_score = rep["score"]
            report.score_spread_pct = rep["spread_pct"]
            components.append((rep["score"], 0.50))
            report.notes.append(
                f"Score varied {rep['spread_pct']}% across {rep['runs']} runs."
            )
        else:
            report.notes.append(rep.get("note", ""))

    if throttle_report:
        retention = throttle_report.get("performance_retention_pct")
        if retention is not None:
            report.performance_retention_pct = retention
            report.sustained_score = round(max(0.0, min(100.0, retention)), 2)
            components.append((report.sustained_score, 0.35))
            report.notes.append(
                f"Sustained {retention}% of opening throughput through the run."
            )

    if cpu_log:
        steady = utilization_steadiness(cpu_log)
        if steady is not None:
            report.utilization_steadiness = steady
            components.append((steady, 0.15))

    if not components:
        report.grade = "unknown"
        report.notes.append("Not enough telemetry to assess stability.")
        return report

    total_weight = sum(w for _, w in components)
    report.overall_score = round(sum(v * w for v, w in components) / total_weight, 2)

    score = report.overall_score
    if score >= 90:
        report.grade = "excellent"
    elif score >= 75:
        report.grade = "good"
    elif score >= 55:
        report.grade = "fair"
    else:
        report.grade = "poor"

    return report


# --------------------------------------------------------------------------
# Backwards-compatible entry point
# --------------------------------------------------------------------------
def calculate_stability(cpu_log: List[float]) -> float:
    """
    Deprecated. Kept so 1.x callers keep working.

    Returns the utilization-steadiness metric, which is the weakest of the
    three stability signals. Prefer `build_stability_report`.
    """
    value = utilization_steadiness(cpu_log or [])
    return value if value is not None else 0.0
