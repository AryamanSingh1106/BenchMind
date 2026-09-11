"""
monitoring/telemetry_service.py

Centralized telemetry. This remains the single source of system telemetry;
do not create independent sampling loops elsewhere.

Changes in 2.0:

* TEMPERATURE POLLING IS DECOUPLED FROM THE MAIN SAMPLER.
  Temperatures come from an external process over HTTP, which can be slow or
  absent. Previously that call sat inline in the 5 Hz sampling loop, so a slow
  or missing sensor source distorted the sampling interval for everything else.
  Now a second, slower thread polls temperature into a shared slot and the fast
  sampler just reads the slot.

* CPU FREQUENCY AND PER-CORE UTILIZATION ARE RECORDED.
  Frequency over time is what makes thermal throttling visible: you see the
  clock fall while the temperature rises. Without it, throttle detection is
  guesswork.

* SAMPLING INTERVAL DRIFT IS MEASURED AND REPORTED.
  If the loop cannot keep up, the report says so rather than quietly producing
  an irregular timeline.

Timing rule, unchanged:
    time.perf_counter() -> benchmark performance measurement
    time.monotonic()    -> telemetry timestamps and window filtering
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from monitoring.temp_reader import get_temperatures

logger = logging.getLogger("BenchMind.TelemetryService")


@dataclass
class TelemetrySnapshot:
    timestamp: float            # wall clock, for display
    monotonic_time: float       # for window filtering, never for scoring
    cpu_utilization: float
    ram_utilization: float
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    cpu_freq: Optional[float] = None            # MHz, package average
    per_core_utilization: List[float] = field(default_factory=list)
    temp_source: str = "unavailable"
    gpu_utilization: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryService:
    _instance: Optional["TelemetryService"] = None
    _instance_lock = threading.Lock()

    def __init__(self, interval: float = 0.2, max_history: int = 20_000,
                 temp_interval: float = 1.0, per_core: bool = True):
        self.interval = interval
        self.temp_interval = temp_interval
        self.max_history = max_history
        self.per_core = per_core

        self._history: deque = deque(maxlen=max_history)
        self._running = False
        self._sampler: Optional[threading.Thread] = None
        self._temp_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[TelemetrySnapshot] = None

        # Shared temperature slot, written by the slow thread, read by the fast one.
        self._temp_slot: Dict[str, Any] = {"cpu_temp": None, "gpu_temp": None,
                                           "source": "unavailable"}
        self._temp_lock = threading.Lock()

        self._interval_errors: deque = deque(maxlen=2000)

    # ---------------- singleton ----------------
    @classmethod
    def get_instance(cls, interval: float = 0.2, max_history: int = 20_000) -> "TelemetryService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(interval=interval, max_history=max_history)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test hook: stop and drop the singleton."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                logger.debug("TelemetryService already running.")
                return
            self._running = True
            self._sampler = threading.Thread(
                target=self._run_loop, name="BenchMindTelemetrySampler", daemon=True)
            self._temp_thread = threading.Thread(
                target=self._temp_loop, name="BenchMindTemperaturePoller", daemon=True)
            self._sampler.start()
            self._temp_thread.start()
            logger.info("TelemetryService started (sampler %.0f Hz, temperature %.1f Hz).",
                        1.0 / self.interval, 1.0 / self.temp_interval)

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            sampler, temp_thread = self._sampler, self._temp_thread
            self._sampler = self._temp_thread = None

        for t in (sampler, temp_thread):
            if t and t.is_alive():
                t.join(timeout=timeout)
        logger.info("TelemetryService stopped.")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # ---------------- reads ----------------
    def get_current(self) -> Optional[TelemetrySnapshot]:
        with self._lock:
            return self._latest_snapshot

    def get_history(self, count: Optional[int] = None) -> List[TelemetrySnapshot]:
        with self._lock:
            items = list(self._history)
        if count is not None and count > 0:
            return items[-count:]
        return items

    def get_history_window(self, start_time: float, end_time: Optional[float] = None,
                           use_monotonic: bool = True) -> List[TelemetrySnapshot]:
        with self._lock:
            items = list(self._history)
        out = []
        for snap in items:
            t = snap.monotonic_time if use_monotonic else snap.timestamp
            if t < start_time:
                continue
            if end_time is not None and t > end_time:
                continue
            out.append(snap)
        return out

    def get_logs_format(self, start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        use_monotonic: bool = True) -> Dict[str, List]:
        """Column-oriented view of a telemetry window, ready for charting."""
        if start_time is not None:
            snaps = self.get_history_window(start_time, end_time, use_monotonic=use_monotonic)
        else:
            snaps = self.get_history()

        base = snaps[0].monotonic_time if snaps else 0.0
        return {
            "cpu": [s.cpu_utilization for s in snaps],
            "ram": [s.ram_utilization for s in snaps],
            "cpu_temp": [s.cpu_temp for s in snaps],
            "gpu_temp": [s.gpu_temp for s in snaps],
            "cpu_freq": [s.cpu_freq for s in snaps],
            "time": [s.timestamp for s in snaps],
            "elapsed": [round(s.monotonic_time - base, 3) for s in snaps],
            "monotonic": [s.monotonic_time for s in snaps],
        }

    def sampling_health(self) -> Dict[str, Any]:
        """
        How well the sampler actually kept to its interval.

        A large max error means the timeline is not evenly spaced and any
        analysis based on sample index rather than timestamp will be wrong.
        """
        with self._lock:
            errors = list(self._interval_errors)
        if not errors:
            return {"samples": 0, "target_interval": self.interval}
        return {
            "samples": len(errors),
            "target_interval": self.interval,
            "mean_interval": round(statistics.fmean(errors), 4),
            "max_interval": round(max(errors), 4),
            "drift_pct": round(
                (statistics.fmean(errors) - self.interval) / self.interval * 100.0, 2),
            "healthy": max(errors) < self.interval * 3,
        }

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._latest_snapshot = None
            self._interval_errors.clear()

    # ---------------- loops ----------------
    def _temp_loop(self) -> None:
        """Slow poller. Isolated so a stalled sensor source cannot skew sampling."""
        while True:
            with self._lock:
                if not self._running:
                    break
            try:
                temps = get_temperatures()
                with self._temp_lock:
                    self._temp_slot = temps
            except Exception as e:  # noqa: BLE001
                logger.debug("Temperature poll failed: %s", e)
            time.sleep(self.temp_interval)

    def _run_loop(self) -> None:
        psutil.cpu_percent(interval=None)
        if self.per_core:
            psutil.cpu_percent(interval=None, percpu=True)

        last_mono = time.monotonic()

        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                now_wall = time.time()
                now_mono = time.monotonic()

                cpu = psutil.cpu_percent(interval=None)
                per_core = (psutil.cpu_percent(interval=None, percpu=True)
                            if self.per_core else [])
                ram = psutil.virtual_memory().percent

                freq_mhz = None
                try:
                    f = psutil.cpu_freq()
                    if f and f.current:
                        freq_mhz = round(float(f.current), 1)
                except Exception:  # noqa: BLE001
                    freq_mhz = None

                with self._temp_lock:
                    temps = dict(self._temp_slot)

                snapshot = TelemetrySnapshot(
                    timestamp=now_wall,
                    monotonic_time=now_mono,
                    cpu_utilization=cpu,
                    ram_utilization=ram,
                    cpu_temp=temps.get("cpu_temp"),
                    gpu_temp=temps.get("gpu_temp"),
                    cpu_freq=freq_mhz,
                    per_core_utilization=per_core,
                    temp_source=temps.get("source", "unavailable"),
                )

                with self._lock:
                    self._latest_snapshot = snapshot
                    self._history.append(snapshot)
                    self._interval_errors.append(now_mono - last_mono)
                last_mono = now_mono

            except Exception as e:  # noqa: BLE001
                logger.error("Telemetry sampling error: %s", e, exc_info=True)

            time.sleep(self.interval)
