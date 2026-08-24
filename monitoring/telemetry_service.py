import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import psutil

from monitoring.temp_reader import get_temperatures

logger = logging.getLogger("BenchMind.TelemetryService")


@dataclass
class TelemetrySnapshot:
    timestamp: float
    monotonic_time: float
    cpu_utilization: float
    ram_utilization: float
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None

    # Extensible metrics for future expansion
    cpu_freq: Optional[float] = None
    gpu_utilization: Optional[float] = None
    ram_bandwidth: Optional[float] = None
    gpus: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryService:
    _instance: Optional["TelemetryService"] = None
    _instance_lock = threading.Lock()

    def __init__(self, interval: float = 0.2, max_history: int = 2000):
        self.interval = interval
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[TelemetrySnapshot] = None

    @classmethod
    def get_instance(cls, interval: float = 0.2, max_history: int = 2000) -> "TelemetryService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(interval=interval, max_history=max_history)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Helper for testing to reset singleton state."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def start(self):
        with self._lock:
            if self._running:
                logger.debug("TelemetryService already running.")
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                name="TelemetrySamplerThread",
                daemon=True
            )
            self._thread.start()
            logger.info("TelemetryService background sampler thread started.")

    def stop(self, timeout: float = 2.0):
        with self._lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
            self._thread = None

        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            logger.info("TelemetryService background sampler thread stopped.")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_current(self) -> Optional[TelemetrySnapshot]:
        with self._lock:
            return self._latest_snapshot

    def get_history(self, count: Optional[int] = None) -> List[TelemetrySnapshot]:
        with self._lock:
            history_list = list(self._history)
        if count is not None and count > 0:
            return history_list[-count:]
        return history_list

    def get_history_window(
        self,
        start_time: float,
        end_time: Optional[float] = None,
        use_monotonic: bool = True
    ) -> List[TelemetrySnapshot]:
        with self._lock:
            history_list = list(self._history)

        results = []
        for snap in history_list:
            t = snap.monotonic_time if use_monotonic else snap.timestamp
            if t >= start_time:
                if end_time is not None and t > end_time:
                    continue
                results.append(snap)
        return results

    def get_logs_format(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        use_monotonic: bool = True
    ) -> Dict[str, List]:
        if start_time is not None:
            snapshots = self.get_history_window(
                start_time, end_time, use_monotonic=use_monotonic
            )
        else:
            snapshots = self.get_history()

        return {
            "cpu": [s.cpu_utilization for s in snapshots],
            "ram": [s.ram_utilization for s in snapshots],
            "cpu_temp": [s.cpu_temp for s in snapshots],
            "gpu_temp": [s.gpu_temp for s in snapshots],
            "time": [s.timestamp for s in snapshots],
        }

    def clear_history(self):
        with self._lock:
            self._history.clear()
            self._latest_snapshot = None

    def _run_loop(self):
        # Warmup psutil CPU calculation
        psutil.cpu_percent(interval=None)

        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                now_wall = time.time()
                now_mono = time.monotonic()
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                temps = get_temperatures()
                cpu_temp = temps.get("cpu_temp")
                gpu_temp = temps.get("gpu_temp")

                snapshot = TelemetrySnapshot(
                    timestamp=now_wall,
                    monotonic_time=now_mono,
                    cpu_utilization=cpu,
                    ram_utilization=ram,
                    cpu_temp=cpu_temp,
                    gpu_temp=gpu_temp,
                )

                with self._lock:
                    self._latest_snapshot = snapshot
                    self._history.append(snapshot)

            except Exception as e:
                logger.error("Error in TelemetryService sampling loop: %s", e, exc_info=True)

            time.sleep(self.interval)
