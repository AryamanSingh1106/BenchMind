"""
monitoring/temp_reader.py

Hardware temperature source with a circuit breaker.

The 1.x version issued a blocking HTTP GET to LibreHardwareMonitor with a 1 s
timeout on EVERY telemetry sample. The sampler runs at 5 Hz. If LHM was not
running -- which is the normal case on a fresh machine, on Linux, and on macOS
-- every single sample could block for up to a second, so the "0.2 s interval"
silently became an irregular multi-second interval and the telemetry timeline
became garbage without anything visibly failing.

2.0 adds:
  * a circuit breaker: after N consecutive failures the source is disabled for
    a cool-off period instead of being retried 5 times a second
  * a much shorter timeout
  * psutil.sensors_temperatures() as a native fallback on Linux
  * an explicit `source` field so the UI can say where the number came from
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("BenchMind.TempReader")

LHM_URL = "http://localhost:8085/data.json"
REQUEST_TIMEOUT = 0.25          # was 1.0 -- far too long for a 5 Hz sampler
FAILURES_BEFORE_TRIP = 3
COOLDOWN_SECONDS = 30.0


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
                logger.info("Temperature source recovered.")
            self.failures = 0
            self.open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold and not self.open_until:
                self.open_until = time.monotonic() + self.cooldown
                logger.info(
                    "Temperature source unavailable after %d attempts; "
                    "pausing probes for %.0fs.", self.failures, self.cooldown)
            elif self.open_until and time.monotonic() >= self.open_until:
                self.open_until = time.monotonic() + self.cooldown

    @property
    def state(self) -> str:
        if self.open_until and time.monotonic() < self.open_until:
            return "open"
        return "closed" if self.failures == 0 else "half_open"


_breaker = _CircuitBreaker(FAILURES_BEFORE_TRIP, COOLDOWN_SECONDS)


def clean_temp(value: Any) -> Optional[float]:
    """Convert '62.0 <degree>C' or 62.0 into a float, or None."""
    if value is None:
        return None
    try:
        text = str(value).replace("\u00b0C", "").replace("C", "").strip()
        return float(text)
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug("Could not parse temperature %r: %s", value, e)
        return None


def _read_librehardwaremonitor() -> Optional[Dict[str, Optional[float]]]:
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

    temps: Dict[str, Optional[float]] = {"cpu_temp": None, "gpu_temp": None}

    def scan(node):
        if not isinstance(node, dict):
            return
        for child in node.get("Children", []) or []:
            scan(child)
        if node.get("Type") == "Temperature":
            name = str(node.get("Text", "")).lower()
            if "cpu package" in name and temps["cpu_temp"] is None:
                temps["cpu_temp"] = clean_temp(node.get("Value"))
            elif "gpu core" in name and temps["gpu_temp"] is None:
                temps["gpu_temp"] = clean_temp(node.get("Value"))

    scan(data)
    _breaker.record_success()
    return temps


def _read_psutil_sensors() -> Optional[Dict[str, Optional[float]]]:
    """Native fallback. Works on most Linux systems, not on Windows."""
    try:
        import psutil
        readings = psutil.sensors_temperatures()
    except Exception:  # noqa: BLE001
        return None
    if not readings:
        return None

    cpu_temp = None
    gpu_temp = None
    preferred = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz")
    for key in preferred:
        if key in readings and readings[key]:
            entries = readings[key]
            pkg = next((e for e in entries if "package" in (e.label or "").lower()), None)
            cpu_temp = float((pkg or entries[0]).current)
            break
    for key in ("amdgpu", "nouveau", "nvidia"):
        if key in readings and readings[key]:
            gpu_temp = float(readings[key][0].current)
            break

    if cpu_temp is None and gpu_temp is None:
        return None
    return {"cpu_temp": cpu_temp, "gpu_temp": gpu_temp}


def get_temperatures() -> Dict[str, Any]:
    """
    Return {'cpu_temp', 'gpu_temp', 'source'}.

    `source` is one of: 'librehardwaremonitor', 'psutil', 'unavailable'. The UI
    shows it so a missing temperature reads as "no sensor source" rather than
    as "your CPU is 0 degrees".
    """
    if platform.system() == "Windows":
        order = (_read_librehardwaremonitor, _read_psutil_sensors)
        names = ("librehardwaremonitor", "psutil")
    else:
        order = (_read_psutil_sensors, _read_librehardwaremonitor)
        names = ("psutil", "librehardwaremonitor")

    for reader, name in zip(order, names):
        result = reader()
        if result and (result.get("cpu_temp") is not None or result.get("gpu_temp") is not None):
            return {**result, "source": name}

    return {"cpu_temp": None, "gpu_temp": None, "source": "unavailable"}


def breaker_state() -> str:
    return _breaker.state


def reset_breaker() -> None:
    """Test hook."""
    _breaker.failures = 0
    _breaker.open_until = 0.0
