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
    drift_verdict: Optional[str] = None
    drift_pct: Optional[float] = None
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

    drift = detect_drift(clean)

    # A session that drifts is not repeatable, however small its standard
    # deviation. Penalise the score by the magnitude of the trend.
    if drift.get("verdict") in ("drift_down", "drift_up"):
        score = min(score, max(0.0, 100.0 - abs(drift["total_change_pct"]) * 8.0))

    return {
        "runs": len(clean),
        "mean_score": round(mean, 2),
        "std_dev": round(sd, 2),
        "spread_pct": round(spread_pct, 2),
        "min_score": round(min(clean), 2),
        "max_score": round(max(clean), 2),
        "score": round(score, 2),
        "drift": drift,
    }


# --------------------------------------------------------------------------
# Drift versus scatter
# --------------------------------------------------------------------------
# Added in 2.0.1. A run of five scores that falls steadily is a completely
# different finding from five scores that bounce around a mean, but standard
# deviation cannot tell them apart. Real measured example:
#
#     2060, 2109, 2081, 1983, 1936
#
# Standard deviation is 3.54%, which the 2.0.0 report called "GOOD". But the
# scores fall monotonically after run 2 and single-core drops 19% from best to
# worst: that is thermal soak across the session, not measurement noise, and
# 30 s of cooldown was not clearing the heat.
#
# Scatter means the measurement is imprecise. Drift means the machine is
# changing while you measure it. Only the second one tells you something about
# the hardware, so they are reported separately.

DRIFT_THRESHOLD_PCT = 3.0      # total change across the session
DRIFT_CORRELATION = 0.6        # |Spearman rho| between run order and score


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Rank correlation. No SciPy dependency for a dozen data points."""
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def detect_drift(scores: List[float]) -> Dict[str, Any]:
    """
    Distinguish a systematic trend across a session from random scatter.

    Uses least-squares slope for magnitude and Spearman rank correlation
    against run order for direction confidence. Both must clear their
    threshold before a trend is called.
    """
    clean = [s for s in scores if s and s > 0]
    n = len(clean)
    if n < 3:
        return {"verdict": "insufficient_runs", "runs": n,
                "message": "At least three runs are needed to separate drift from scatter."}

    xs = list(range(1, n + 1))
    mean_score = statistics.fmean(clean)
    mean_x = statistics.fmean(xs)

    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((x - mean_x) * (y - mean_score) for x, y in zip(xs, clean)) / denominator
             if denominator else 0.0)
    total_change_pct = (slope * (n - 1) / mean_score * 100.0) if mean_score else 0.0
    rho = _spearman([float(x) for x in xs], clean)

    drifting = abs(total_change_pct) >= DRIFT_THRESHOLD_PCT and abs(rho) >= DRIFT_CORRELATION

    result = {
        "runs": n,
        "slope_per_run": round(slope, 2),
        "total_change_pct": round(total_change_pct, 2),
        "rank_correlation": round(rho, 3),
        "first_score": round(clean[0], 2),
        "last_score": round(clean[-1], 2),
        "best_score": round(max(clean), 2),
        "worst_score": round(min(clean), 2),
    }

    if not drifting:
        result["verdict"] = "scatter"
        result["message"] = (
            "No systematic trend across the session; the spread is measurement "
            "scatter rather than the machine changing under you."
        )
        return result

    direction = "downward" if total_change_pct < 0 else "upward"
    result["verdict"] = "drift_down" if total_change_pct < 0 else "drift_up"

    if total_change_pct < 0:
        result["message"] = (
            f"Scores drifted {direction} by {abs(total_change_pct):.1f}% across "
            f"{n} runs (rank correlation {rho:.2f}). This is not scatter: the machine "
            "got slower as the session went on. The usual cause is heat not clearing "
            "between runs, so try a longer cooldown. The spread figure below "
            "understates the problem, because standard deviation treats a steady "
            "decline as if it were noise."
        )
    else:
        result["message"] = (
            f"Scores drifted {direction} by {total_change_pct:.1f}% across {n} runs "
            f"(rank correlation {rho:.2f}). Early runs were slower, which usually means "
            "the first run paid a warmup cost: caches cold, files not yet in the page "
            "cache, or the CPU still at an idle clock state."
        )
    return result


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
            drift = rep.get("drift", {})
            report.drift_verdict = drift.get("verdict")
            report.drift_pct = drift.get("total_change_pct")
            if drift.get("verdict") in ("drift_down", "drift_up"):
                report.notes.append(drift["message"])
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
