import unittest
import time
from unittest.mock import patch
import requests

from monitoring.telemetry_service import TelemetryService, TelemetrySnapshot
from monitoring.live_monitor import TimelineCollector


class TestTelemetryService(unittest.TestCase):

    def setUp(self):
        # Reset singleton state before each test
        TelemetryService.reset_instance()

    def tearDown(self):
        # Stop and cleanup after each test
        TelemetryService.reset_instance()

    def test_service_start_and_idempotency(self):
        service = TelemetryService.get_instance(interval=0.05)
        self.assertFalse(service.is_running())

        # Start service
        service.start()
        self.assertTrue(service.is_running())

        initial_thread = service._thread
        self.assertIsNotNone(initial_thread)

        # Call start multiple times - must be idempotent and keep the same thread
        service.start()
        service.start()
        self.assertTrue(service.is_running())
        self.assertEqual(service._thread, initial_thread)

    def test_service_stop(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()
        self.assertTrue(service.is_running())

        service.stop(timeout=1.0)
        self.assertFalse(service.is_running())
        self.assertIsNone(service._thread)

    def test_uninitialized_get_current(self):
        service = TelemetryService.get_instance(interval=0.05)
        # Before sampling starts or completes, get_current must return None (no fabricated zeros)
        self.assertIsNone(service.get_current())

    def test_get_current_telemetry(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()

        # Wait for at least one sample tick
        time.sleep(0.15)

        snapshot = service.get_current()
        self.assertIsNotNone(snapshot)
        self.assertIsInstance(snapshot, TelemetrySnapshot)
        self.assertGreaterEqual(snapshot.cpu_utilization, 0.0)
        self.assertGreaterEqual(snapshot.ram_utilization, 0.0)
        self.assertGreater(snapshot.timestamp, 0.0)
        self.assertGreater(snapshot.monotonic_time, 0.0)

    def test_get_historical_samples(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()

        time.sleep(0.2)

        history = service.get_history()
        self.assertGreaterEqual(len(history), 2)

        # Test count limiter
        limited = service.get_history(count=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0], history[-1])

    def test_get_history_window_monotonic(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()

        time.sleep(0.1)
        start_win = time.monotonic()

        time.sleep(0.15)
        end_win = time.monotonic()

        time.sleep(0.1)

        window_samples = service.get_history_window(
            start_time=start_win,
            end_time=end_win,
            use_monotonic=True
        )

        self.assertGreater(len(window_samples), 0)
        for s in window_samples:
            self.assertGreaterEqual(s.monotonic_time, start_win)
            self.assertLessEqual(s.monotonic_time, end_win)

    @patch("monitoring.temp_reader.requests.get")
    def test_missing_temperature_sensors(self, mock_get):
        # Mock LibreHardwareMonitor HTTP endpoint throwing RequestException
        mock_get.side_effect = requests.RequestException("Connection refused")

        service = TelemetryService.get_instance(interval=0.05)
        service.start()

        time.sleep(0.12)

        snapshot = service.get_current()
        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot.cpu_temp)
        self.assertIsNone(snapshot.gpu_temp)

    def test_timeline_collector_backward_compatibility(self):
        collector = TimelineCollector(interval=0.05)
        collector.start()

        time.sleep(0.15)

        collector.stop()
        logs = collector.get_logs()

        self.assertIn("cpu", logs)
        self.assertIn("ram", logs)
        self.assertIn("cpu_temp", logs)
        self.assertIn("gpu_temp", logs)
        self.assertIn("time", logs)
        self.assertGreater(len(logs["cpu"]), 0)


if __name__ == "__main__":
    unittest.main()
