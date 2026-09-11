"""
tests/test_monitoring.py

Tests for telemetry, the temperature circuit breaker, the validity gate,
history storage and the analysis layer.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from monitoring.telemetry_service import TelemetryService


class TestTelemetryService(unittest.TestCase):
    def setUp(self):
        TelemetryService.reset_instance()

    def tearDown(self):
        TelemetryService.reset_instance()

    def test_singleton(self):
        self.assertIs(TelemetryService.get_instance(), TelemetryService.get_instance())

    def test_start_is_idempotent(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()
        service.start()
        self.assertTrue(service.is_running())
        service.stop()
        self.assertFalse(service.is_running())

    def test_collects_samples(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()
        time.sleep(0.4)
        snapshot = service.get_current()
        service.stop()

        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.cpu_utilization, 0.0)
        self.assertGreater(snapshot.monotonic_time, 0.0)
        self.assertGreater(len(service.get_history()), 2)

    def test_window_filtering_uses_monotonic(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()
        time.sleep(0.2)
        start = time.monotonic()
        time.sleep(0.3)
        end = time.monotonic()
        time.sleep(0.2)
        service.stop()

        window = service.get_history_window(start, end, use_monotonic=True)
        self.assertTrue(window)
        for snap in window:
            self.assertGreaterEqual(snap.monotonic_time, start)
            self.assertLessEqual(snap.monotonic_time, end)
        self.assertLess(len(window), len(service.get_history()))

    def test_logs_format_columns_align(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()
        time.sleep(0.3)
        service.stop()

        logs = service.get_logs_format()
        lengths = {key: len(values) for key, values in logs.items()}
        self.assertEqual(len(set(lengths.values())), 1,
                         f"telemetry columns are ragged: {lengths}")

    def test_sampling_health_reported(self):
        service = TelemetryService.get_instance(interval=0.05)
        service.start()
        time.sleep(0.4)
        health = service.sampling_health()
        service.stop()

        self.assertGreater(health["samples"], 0)
        self.assertIn("healthy", health)

    def test_slow_temperature_source_does_not_stall_sampler(self):
        """
        The 1.x bug: a blocking temperature call inside the sampling loop turned
        a 5 Hz sampler into a 1 Hz one whenever the sensor source was missing.
        """
        def slow_temps():
            time.sleep(0.5)
            return {"cpu_temp": None, "gpu_temp": None, "source": "unavailable"}

        with mock.patch("monitoring.telemetry_service.get_temperatures", slow_temps):
            service = TelemetryService.get_instance(interval=0.05)
            service.start()
            time.sleep(1.0)
            samples = len(service.get_history())
            service.stop()

        self.assertGreater(samples, 8,
                           "sampler was blocked by the temperature source")


class TestTemperatureCircuitBreaker(unittest.TestCase):
    def setUp(self):
        from monitoring import temp_reader
        temp_reader.reset_breaker()

    def test_breaker_trips_after_repeated_failure(self):
        from monitoring import temp_reader

        with mock.patch("requests.get", side_effect=OSError("connection refused")):
            for _ in range(temp_reader.FAILURES_BEFORE_TRIP + 1):
                temp_reader._read_librehardwaremonitor()

        self.assertEqual(temp_reader.breaker_state(), "open")

    def test_open_breaker_stops_probing(self):
        from monitoring import temp_reader

        with mock.patch("requests.get", side_effect=OSError("refused")) as fake:
            for _ in range(temp_reader.FAILURES_BEFORE_TRIP):
                temp_reader._read_librehardwaremonitor()
            calls_after_trip = fake.call_count
            for _ in range(20):
                temp_reader._read_librehardwaremonitor()

        self.assertEqual(fake.call_count, calls_after_trip,
                         "breaker should suppress further probes")

    def test_missing_source_reports_unavailable_not_zero(self):
        from monitoring import temp_reader

        with mock.patch.object(temp_reader, "_read_librehardwaremonitor", return_value=None), \
             mock.patch.object(temp_reader, "_read_psutil_sensors", return_value=None):
            temps = temp_reader.get_temperatures()

        self.assertIsNone(temps["cpu_temp"])
        self.assertEqual(temps["source"], "unavailable")

    def test_clean_temp_parses_degree_strings(self):
        from monitoring.temp_reader import clean_temp
        self.assertEqual(clean_temp("62.0 \u00b0C"), 62.0)
        self.assertEqual(clean_temp(55.5), 55.5)
        self.assertIsNone(clean_temp("not a number"))
        self.assertIsNone(clean_temp(None))


class TestEnvironmentFingerprint(unittest.TestCase):
    def test_fingerprint_is_stable(self):
        from monitoring.environment import get_environment_fingerprint
        a = get_environment_fingerprint(refresh=True)
        b = get_environment_fingerprint(refresh=True)
        self.assertEqual(a["fingerprint_hash"], b["fingerprint_hash"])

    def test_python_version_change_changes_hash(self):
        from monitoring.environment import compute_fingerprint_hash, get_environment_fingerprint

        base = get_environment_fingerprint(refresh=True)
        altered = {**base, "python": {**base["python"], "version": "9.9.9"}}
        self.assertNotEqual(compute_fingerprint_hash(base),
                            compute_fingerprint_hash(altered))

    def test_battery_percent_does_not_change_hash(self):
        """Cosmetic state must not split the comparison pool."""
        from monitoring.environment import compute_fingerprint_hash, get_environment_fingerprint

        base = get_environment_fingerprint(refresh=True)
        altered = {**base, "power": {"has_battery": True, "on_ac_power": True,
                                     "battery_percent": 42.0}}
        self.assertEqual(compute_fingerprint_hash(base),
                         compute_fingerprint_hash(altered))


class TestValidityGate(unittest.TestCase):
    def test_clean_machine_passes(self):
        from monitoring.validity import ValidityReport, check_run_validity

        report = check_run_validity(sample_seconds=0.3, read_temperature=False)
        self.assertIsInstance(report, ValidityReport)
        self.assertIn(report.verdict, ("valid", "tainted", "invalid"))
        self.assertEqual(report.is_comparable, report.verdict == "valid")

    def test_battery_taints_the_run(self):
        from monitoring import validity

        fake_battery = mock.Mock(power_plugged=False, percent=55.0)
        with mock.patch("psutil.sensors_battery", return_value=fake_battery):
            report = validity.check_run_validity(sample_seconds=0.3, read_temperature=False)

        codes = [i.code for i in report.issues]
        self.assertIn("on_battery", codes)
        self.assertNotEqual(report.verdict, "valid")

    def test_heavy_background_load_invalidates(self):
        from monitoring import validity

        with mock.patch("psutil.cpu_percent", return_value=85.0), \
             mock.patch.object(validity, "_top_cpu_processes", return_value=[]):
            report = validity.check_run_validity(sample_seconds=0.3, read_temperature=False)

        self.assertEqual(report.verdict, "invalid")
        self.assertFalse(report.is_comparable)


class TestHistoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from storage.history import HistoryStore
        self.store = HistoryStore(Path(self.tmp.name) / "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, index: float, fingerprint: str = "abc123",
                verdict: str = "valid", mode: str = "standard"):
        return {
            "system": {"cpu": "Test CPU", "os": "TestOS", "logical_cores": 8},
            "environment": {"fingerprint_hash": fingerprint, "benchmind_version": "2.0.0"},
            "validity": {"verdict": verdict},
            "cpu_benchmark": {
                "mode": mode, "cpu_index": index, "cpu_index_ci_pct": 1.0,
                "single_core_score": index * 0.9, "multi_core_score": index * 1.1,
                "baseline_version": "2.0.0", "category_scores": {},
                "telemetry_summary": {"max_cpu_temp": 70.0},
            },
            "throttling": {"throttling_detected": False},
            "stability": {"overall_score": 95.0},
        }

    def test_save_and_read_back(self):
        run_id = self.store.save_run(self._result(1000))
        self.assertIsInstance(run_id, int)
        runs = self.store.recent_runs(limit=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["cpu_index"], 1000)

    def test_first_run_has_no_baseline(self):
        result = self._result(1000)
        run_id = self.store.save_run(result)
        verdict = self.store.detect_regression(result, exclude_run_id=run_id)
        self.assertEqual(verdict["verdict"], "no_baseline")

    def test_detects_regression(self):
        for _ in range(3):
            self.store.save_run(self._result(1000))
        fresh = self._result(820)
        run_id = self.store.save_run(fresh)
        verdict = self.store.detect_regression(fresh, exclude_run_id=run_id)

        self.assertEqual(verdict["verdict"], "regression")
        self.assertLess(verdict["change_pct"], -10)

    def test_small_change_is_called_stable(self):
        for _ in range(3):
            self.store.save_run(self._result(1000))
        fresh = self._result(1020)
        run_id = self.store.save_run(fresh)
        verdict = self.store.detect_regression(fresh, exclude_run_id=run_id)
        self.assertEqual(verdict["verdict"], "stable")

    def test_different_fingerprint_is_not_a_baseline(self):
        """Comparing across software stacks is exactly what must not happen."""
        for _ in range(3):
            self.store.save_run(self._result(1000, fingerprint="stack_a"))
        fresh = self._result(500, fingerprint="stack_b")
        run_id = self.store.save_run(fresh)
        verdict = self.store.detect_regression(fresh, exclude_run_id=run_id)
        self.assertEqual(verdict["verdict"], "no_baseline")

    def test_different_mode_is_not_a_baseline(self):
        for _ in range(3):
            self.store.save_run(self._result(1000, mode="quick"))
        fresh = self._result(500, mode="standard")
        run_id = self.store.save_run(fresh)
        verdict = self.store.detect_regression(fresh, exclude_run_id=run_id)
        self.assertEqual(verdict["verdict"], "no_baseline")

    def test_tainted_run_is_not_used_as_baseline(self):
        for _ in range(3):
            self.store.save_run(self._result(1000, verdict="tainted"))
        fresh = self._result(1000)
        run_id = self.store.save_run(fresh)
        verdict = self.store.detect_regression(fresh, exclude_run_id=run_id)
        self.assertEqual(verdict["verdict"], "no_baseline")

    def test_tainted_run_is_not_compared(self):
        for _ in range(3):
            self.store.save_run(self._result(1000))
        fresh = self._result(400, verdict="tainted")
        verdict = self.store.detect_regression(fresh)
        self.assertEqual(verdict["verdict"], "incomparable")


class TestAnalysis(unittest.TestCase):
    def test_throttling_detected_from_falling_clock(self):
        from ai.analysis import analyze_throttling

        clocks = [4200] * 10 + [4100] * 10 + [3600] * 10 + [3100] * 10
        temps = [55] * 10 + [70] * 10 + [86] * 10 + [95] * 10
        telemetry = {
            "cpu_freq": clocks, "cpu_temp": temps,
            "elapsed": [i * 0.2 for i in range(len(clocks))],
        }
        result = analyze_throttling(telemetry)

        self.assertTrue(result["throttling_detected"])
        self.assertEqual(result["confidence"], "high")
        self.assertGreater(result["clock_drop_pct"], 20)
        self.assertIsNotNone(result["onset_seconds"])

    def test_steady_clock_is_not_throttling(self):
        from ai.analysis import analyze_throttling

        telemetry = {
            "cpu_freq": [4200, 4190, 4205, 4198] * 10,
            "cpu_temp": [62, 63, 64, 63] * 10,
            "elapsed": [i * 0.2 for i in range(40)],
        }
        result = analyze_throttling(telemetry)
        self.assertFalse(result["throttling_detected"])

    def test_missing_sensors_says_so(self):
        from ai.analysis import analyze_throttling

        result = analyze_throttling({"cpu_freq": [], "cpu_temp": [], "elapsed": []})
        self.assertFalse(result["throttling_detected"])
        self.assertIn("not", result["summary"].lower())

    def test_bottleneck_identifies_bandwidth_limit(self):
        from ai.analysis import analyze_bottleneck

        subtests = [
            {"category": "floating_point", "name": "fp", "threads": 1,
             "validation_passed": True, "raw_metric_value": 8000.0,
             "arithmetic_intensity": 30.0, "workload_profile": "compute_bound",
             "raw_metric_name": "MFLOPS", "working_set_bytes": 1e6},
            {"category": "vector_simd", "name": "vec", "threads": 1,
             "validation_passed": True, "raw_metric_value": 1.0,
             "arithmetic_intensity": 0.125, "workload_profile": "memory_bound",
             "raw_metric_name": "GFLOPS", "working_set_bytes": 1.6e8},
        ]
        result = analyze_bottleneck(subtests)
        self.assertEqual(result["verdict"], "memory_bandwidth_limited")
        self.assertIsNotNone(result["estimated_memory_bandwidth_gbs"])

    def test_stability_report_weights_repeatability(self):
        from ai.stability_engine import build_stability_report

        tight = build_stability_report(run_scores=[1000, 1005, 998, 1002])
        loose = build_stability_report(run_scores=[1000, 1300, 700, 1150])
        self.assertGreater(tight.overall_score, loose.overall_score)
        self.assertEqual(tight.grade, "excellent")

    def test_legacy_stability_entry_point_still_works(self):
        from ai.stability_engine import calculate_stability
        value = calculate_stability([50.0, 52.0, 48.0, 51.0])
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)


if __name__ == "__main__":
    unittest.main()
