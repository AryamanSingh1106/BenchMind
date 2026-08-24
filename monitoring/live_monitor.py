import time
from typing import Dict, List, Optional
from monitoring.telemetry_service import TelemetryService


class TimelineCollector:

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.service = TelemetryService.get_instance(interval=self.interval)
        self.start_mono: Optional[float] = None
        self.end_mono: Optional[float] = None
        self.running = False

    # ===== START =====
    def start(self):
        self.start_mono = time.monotonic()
        self.end_mono = None
        self.running = True
        self.service.start()

    # ===== STOP =====
    def stop(self):
        if self.running:
            self.end_mono = time.monotonic()
            self.running = False

    # ===== GET LOGS =====
    def get_logs(self) -> Dict[str, List]:
        if self.start_mono is None:
            return self.service.get_logs_format()
        
        end_time = self.end_mono if not self.running and self.end_mono is not None else time.monotonic()
        return self.service.get_logs_format(
            start_time=self.start_mono,
            end_time=end_time,
            use_monotonic=True
        )