"""
monitoring/temp_reader.py

Hardware sensor source with a circuit breaker.

Two jobs, both fed by the same HTTP poll of LibreHardwareMonitor:

  1. TEMPERATURES (as in 2.0.0)
  2. CLOCK SPEED AND PACKAGE POWER (new in 2.0.1)

Why clocks moved here from psutil
---------------------------------
On Windows, `psutil.cpu_freq()` reads the *nominal* base clock from the
registry, not the live frequency. On an i5-13450HX it returns a constant
2400 MHz forever, while the chip is actually running anywhere from 800 MHz to
4600 MHz. Throttle detection built on that number is reading a flat line and
can never fire.

LibreHardwareMonitor reads the real per-core clocks from MSRs, and also
reports CPU package power, which on a laptop is usually the *earlier* throttle
signal: PL1 drops from its short-burst value to the sustained limit well
before the clock visibly collapses.

Sensor naming
-------------
Matching is deliberately tolerant, because sensor names differ across LHM
versions and CPU generations:

  hybrid Intel   "P-Core #1" ... "E-Core #4"
  older Intel    "CPU Core #1"
  AMD            "Core #1" / "CCD1 (Tdie)"

Anything under Clocks that is not core-like (Bus Speed, Memory, GPU) is
ignored.
"""

from __future__ import annotations

import logging
import platform
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.TempReader")

LHM_URL = "http://localhost:8085/data.json"
REQUEST_TIMEOUT = 0.5
FAILURES_BEFORE_TRIP = 3
COOLDOWN_SECONDS = 30.0

_P_CORE = re.compile(r"^p[- ]?core\s*#?\d+", re.I)
_E_CORE = re.compile(r"^e[- ]?core\s*#?\d+", re.I)
_GENERIC_CORE = re.compile(r"^(cpu\s+)?core\s*#?\d+", re.I)


class _CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.open_until and time.monotonic() < self.open_until:
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            if self.failures:
                logger.info("Sensor source recovered.")
            self.failures = 0
            self.open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold and not self.open_until:
                self.open_until = time.monotonic() + self.cooldown
                logger.info(
                    "Sensor source unavailable after %d attempts; "
                    "pausing probes for %.0fs.", self.failures, self.cooldown)
            elif self.open_until and time.monotonic() >= self.open_until:
                self.open_until = time.monotonic() + self.cooldown

    @property
    def state(self) -> str:
        if self.open_until and time.monotonic() < self.open_until:
            return "open"
        return "closed" if self.failures == 0 else "half_open"


_breaker = _CircuitBreaker(FAILURES_BEFORE_TRIP, COOLDOWN_SECONDS)


def _parse_number(value: Any) -> Optional[float]:
    """
    Pull a float out of an LHM value string.

    LHM formats values with their unit attached and a locale-dependent decimal
    separator: '62.0 <degree>C', '4,192.5 MHz', '45.3 W'. Strip everything that
    is not part of the number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        match = re.search(r"-?\d[\d,]*\.?\d*", text)
        if not match:
            return None
        return float(match.group(0).replace(",", ""))
    except (ValueError, TypeError):
        logger.debug("Could not parse sensor value %r", value)
        return None


def clean_temp(value: Any) -> Optional[float]:
    """Backwards-compatible alias kept for 2.0.0 callers and tests."""
    return _parse_number(value)


def _empty_reading() -> Dict[str, Any]:
    return {
        "cpu_temp": None,
        "gpu_temp": None,
        "cpu_clock_mhz": None,       # mean across performance cores
        "cpu_clock_max_mhz": None,   # fastest single core
        "p_core_clock_mhz": None,
        "e_core_clock_mhz": None,
        "cpu_power_w": None,
        "core_count_seen": 0,
        "source": "unavailable",
        "clock_source": "unavailable",
    }


def _read_librehardwaremonitor() -> Optional[Dict[str, Any]]:
    if not _breaker.allow():
        return None
    try:
        import requests
    except ImportError:
        _breaker.record_failure()
        return None

    try:
        response = requests.get(LHM_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("LibreHardwareMonitor probe failed: %s", e)
        _breaker.record_failure()
        return None

    reading = _empty_reading()
    p_clocks: List[float] = []
    e_clocks: List[float] = []
    generic_clocks: List[float] = []
    fallback_temps: List[float] = []

    def scan(node):
        if not isinstance(node, dict):
            return
        for child in node.get("Children", []) or []:
            scan(child)

        stype = node.get("Type")
        name = str(node.get("Text", "")).strip()
        lowered = name.lower()

        if stype == "Temperature":
            value = _parse_number(node.get("Value"))
            if value is None:
                return
            if "cpu package" in lowered and reading["cpu_temp"] is None:
                reading["cpu_temp"] = value
            elif ("gpu core" in lowered or "gpu hot spot" in lowered) \
                    and reading["gpu_temp"] is None:
                reading["gpu_temp"] = value
            elif "core max" in lowered or "core average" in lowered or "tdie" in lowered:
                fallback_temps.append(value)

        elif stype == "Clock":
            # Bus Speed, memory and GPU clocks are not CPU core clocks.
            if not (_P_CORE.match(name) or _E_CORE.match(name)
                    or _GENERIC_CORE.match(name)):
                return
            value = _parse_number(node.get("Value"))
            if value is None or value <= 0:
                return
            if _P_CORE.match(name):
                p_clocks.append(value)
            elif _E_CORE.match(name):
                e_clocks.append(value)
            else:
                generic_clocks.append(value)

        elif stype == "Power":
            if "cpu package" in lowered and reading["cpu_power_w"] is None:
                reading["cpu_power_w"] = _parse_number(node.get("Value"))

    scan(data)

    if reading["cpu_temp"] is None and fallback_temps:
        reading["cpu_temp"] = max(fallback_temps)

    if p_clocks:
        reading["p_core_clock_mhz"] = round(sum(p_clocks) / len(p_clocks), 1)
    if e_clocks:
        reading["e_core_clock_mhz"] = round(sum(e_clocks) / len(e_clocks), 1)

    # The headline clock is the performance-core mean. On a hybrid CPU the
    # E-cores run several hundred MHz slower by design, so averaging all cores
    # together would make a shift in the P/E work split look like throttling.
    primary = p_clocks or generic_clocks or e_clocks
    if primary:
        reading["cpu_clock_mhz"] = round(sum(primary) / len(primary), 1)
        reading["cpu_clock_max_mhz"] = round(max(primary), 1)
        reading["core_count_seen"] = len(p_clocks) + len(e_clocks) + len(generic_clocks)
        reading["clock_source"] = "librehardwaremonitor"

    _breaker.record_success()
    return reading


def _read_psutil_sensors() -> Optional[Dict[str, Any]]:
    """
    Native fallback. Works for temperatures on most Linux systems.

    Deliberately does NOT supply a clock: psutil's frequency is the nominal
    base clock on Windows and is useless for throttle detection. Reporting no
    clock is better than reporting a constant that looks like a measurement.
    """
    try:
        import psutil
        readings = psutil.sensors_temperatures()
    except Exception:  # noqa: BLE001
        return None
    if not readings:
        return None

    result = _empty_reading()
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz"):
        if key in readings and readings[key]:
            entries = readings[key]
            pkg = next((e for e in entries if "package" in (e.label or "").lower()), None)
            result["cpu_temp"] = float((pkg or entries[0]).current)
            break
    for key in ("amdgpu", "nouveau", "nvidia"):
        if key in readings and readings[key]:
            result["gpu_temp"] = float(readings[key][0].current)
            break

    if result["cpu_temp"] is None and result["gpu_temp"] is None:
        return None

    # Linux exposes live per-core frequency through scaling_cur_freq, which
    # psutil reads correctly. Windows does not, so gate on platform.
    if platform.system() == "Linux":
        try:
            import psutil
            per_core = psutil.cpu_freq(percpu=True)
            live = [f.current for f in per_core if f and f.current]
            if live:
                result["cpu_clock_mhz"] = round(sum(live) / len(live), 1)
                result["cpu_clock_max_mhz"] = round(max(live), 1)
                result["core_count_seen"] = len(live)
                result["clock_source"] = "psutil"
        except Exception:  # noqa: BLE001
            pass

    return result


def read_sensors() -> Dict[str, Any]:
    """
    Full sensor reading: temperatures, clocks and package power.

    `source` and `clock_source` are reported separately, because temperature
    may be available while a usable clock is not. The UI shows both so that a
    missing clock reads as "no clock source" rather than as a flat line.
    """
    if platform.system() == "Windows":
        order = (_read_librehardwaremonitor, _read_psutil_sensors)
        names = ("librehardwaremonitor", "psutil")
    else:
        order = (_read_psutil_sensors, _read_librehardwaremonitor)
        names = ("psutil", "librehardwaremonitor")

    for reader, name in zip(order, names):
        result = reader()
        if result and any(result.get(k) is not None
                          for k in ("cpu_temp", "gpu_temp", "cpu_clock_mhz")):
            result["source"] = name
            return result

    return _empty_reading()


def get_temperatures() -> Dict[str, Any]:
    """
    Backwards-compatible entry point returning just the temperature fields.

    Prefer `read_sensors()`, which also carries clocks and package power.
    """
    reading = read_sensors()
    return {
        "cpu_temp": reading["cpu_temp"],
        "gpu_temp": reading["gpu_temp"],
        "source": reading["source"],
    }


def breaker_state() -> str:
    return _breaker.state


def reset_breaker() -> None:
    """Test hook."""
    _breaker.failures = 0
    _breaker.open_until = 0.0
