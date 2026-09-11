"""
monitoring/validity.py

Pre-run validity gate.

A benchmark run taken while Chrome is indexing, on battery, with the machine
already at 85 C, is not a measurement of the hardware. It is a measurement of
that moment. Serious benchmark suites refuse to certify such runs; hobby
projects usually just report the number anyway and leave the user confused
about why their score dropped 30%.

BenchMind checks the conditions, records what it found, and marks the run
`valid`, `tainted` or `invalid`. Tainted and invalid runs are still shown,
because the user asked for a run, but the history and comparison layers refuse
to use them as a baseline.

This is the highest-value check for the laptop case specifically: on battery,
most laptops cap the CPU package power and the score can halve.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger("BenchMind.Validity")

# Thresholds. Tuned to be permissive enough that a normal desktop passes.
MAX_BACKGROUND_CPU_PCT = 15.0     # above this, something else is competing
MAX_BACKGROUND_CPU_HARD = 40.0    # above this, the run is not usable at all
MIN_FREE_RAM_PCT = 15.0
MAX_START_TEMP_C = 75.0           # already hot => throttling likely from the start
SAMPLE_SECONDS = 2.0


@dataclass
class ValidityIssue:
    code: str
    severity: str          # "warning" (tainted) or "error" (invalid)
    message: str
    measured: Any = None
    threshold: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidityReport:
    verdict: str = "valid"              # valid | tainted | invalid
    issues: List[ValidityIssue] = field(default_factory=list)
    background_cpu_pct: float = 0.0
    free_ram_pct: float = 0.0
    start_cpu_temp: Optional[float] = None
    on_ac_power: bool = True
    has_battery: bool = False
    process_count: int = 0
    top_processes: List[Dict[str, Any]] = field(default_factory=list)
    checked_at: float = 0.0

    @property
    def is_comparable(self) -> bool:
        """Only `valid` runs may be used as a baseline or published."""
        return self.verdict == "valid"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.to_dict() if isinstance(i, ValidityIssue) else i for i in self.issues]
        d["is_comparable"] = self.is_comparable
        return d


def _top_cpu_processes(limit: int = 5) -> List[Dict[str, Any]]:
    """Identify what is competing for the CPU, so the user can go close it."""
    procs = []
    try:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                p.cpu_percent(interval=None)
                procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.5)
        rows = []
        for p in procs:
            try:
                pct = p.cpu_percent(interval=None)
                if pct > 1.0:
                    rows.append({"pid": p.pid, "name": p.info.get("name"),
                                 "cpu_percent": round(pct, 1)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        rows.sort(key=lambda r: r["cpu_percent"], reverse=True)
        return rows[:limit]
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not enumerate processes: %s", e)
        return []


def check_run_validity(sample_seconds: float = SAMPLE_SECONDS,
                       read_temperature: bool = True) -> ValidityReport:
    """
    Sample the machine's idle state and decide whether a benchmark taken right
    now would produce a comparable number.
    """
    report = ValidityReport(checked_at=time.time())

    # Background CPU load, averaged over a short window.
    psutil.cpu_percent(interval=None)
    time.sleep(max(0.2, sample_seconds))
    report.background_cpu_pct = round(psutil.cpu_percent(interval=None), 2)

    vm = psutil.virtual_memory()
    report.free_ram_pct = round(100.0 - vm.percent, 2)
    report.process_count = len(psutil.pids())

    battery = None
    try:
        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        battery = None
    if battery is not None:
        report.has_battery = True
        report.on_ac_power = bool(battery.power_plugged)

    if read_temperature:
        try:
            from monitoring.temp_reader import get_temperatures
            report.start_cpu_temp = get_temperatures().get("cpu_temp")
        except Exception:  # noqa: BLE001
            report.start_cpu_temp = None

    # --- evaluate ---
    if report.background_cpu_pct >= MAX_BACKGROUND_CPU_HARD:
        report.top_processes = _top_cpu_processes()
        report.issues.append(ValidityIssue(
            code="background_load_severe",
            severity="error",
            message=(f"Background CPU load is {report.background_cpu_pct}%. "
                     "Close other applications before benchmarking; this run "
                     "cannot be compared against anything."),
            measured=report.background_cpu_pct,
            threshold=MAX_BACKGROUND_CPU_HARD,
        ))
    elif report.background_cpu_pct >= MAX_BACKGROUND_CPU_PCT:
        report.top_processes = _top_cpu_processes()
        report.issues.append(ValidityIssue(
            code="background_load",
            severity="warning",
            message=(f"Background CPU load is {report.background_cpu_pct}%. "
                     "Scores will be depressed and run-to-run spread will widen."),
            measured=report.background_cpu_pct,
            threshold=MAX_BACKGROUND_CPU_PCT,
        ))

    if report.has_battery and not report.on_ac_power:
        report.issues.append(ValidityIssue(
            code="on_battery",
            severity="warning",
            message=("Running on battery. Most laptops cap package power when "
                     "unplugged, which can halve the score. Plug in and re-run "
                     "before comparing against anything."),
            measured="battery",
            threshold="ac_power",
        ))

    if report.free_ram_pct < MIN_FREE_RAM_PCT:
        report.issues.append(ValidityIssue(
            code="low_memory",
            severity="warning",
            message=(f"Only {report.free_ram_pct}% of RAM is free. Memory-bound "
                     "subtests may be affected by paging."),
            measured=report.free_ram_pct,
            threshold=MIN_FREE_RAM_PCT,
        ))

    if report.start_cpu_temp is not None and report.start_cpu_temp > MAX_START_TEMP_C:
        report.issues.append(ValidityIssue(
            code="hot_start",
            severity="warning",
            message=(f"CPU is already at {report.start_cpu_temp} C before the run. "
                     "Throttling is likely to begin almost immediately. Let the "
                     "machine cool and re-run."),
            measured=report.start_cpu_temp,
            threshold=MAX_START_TEMP_C,
        ))

    if any(i.severity == "error" for i in report.issues):
        report.verdict = "invalid"
    elif report.issues:
        report.verdict = "tainted"
    else:
        report.verdict = "valid"

    logger.info("Run validity: %s (%d issue(s))", report.verdict, len(report.issues))
    return report


def describe_verdict(report: ValidityReport) -> str:
    if report.verdict == "valid":
        return "Conditions were clean; this score is comparable."
    if report.verdict == "tainted":
        return ("Conditions were imperfect; the score is shown but is not stored "
                "as a comparison baseline.")
    return "Conditions were unusable; treat this score as meaningless."
