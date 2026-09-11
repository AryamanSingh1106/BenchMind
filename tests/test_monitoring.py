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
        def slow_sensors():
            time.sleep(0.5)
            return {"cpu_temp": None, "gpu_temp": None, "cpu_clock_mhz": None,
                    "cpu_clock_max_mhz": None, "cpu_power_w": None,
                    "source": "unavailable", "clock_source": "unavailable"}

        with mock.patch("monitoring.telemetry_service.read_sensors", slow_sensors):
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


class TestClockSignalHandling(unittest.TestCase):
    """
    Regression tests for the 2.0.0 false positive on an i5-13450HX.

    psutil on Windows returns the nominal base clock, so the clock series was a
    constant. 2.0.0 read that as "clock held 100%" and then fired a throttle
    verdict on temperature alone, producing a self-contradicting report.
    """

    def test_flat_clock_series_is_treated_as_absent(self):
        from ai.analysis import analyze_throttling

        # Exactly what psutil.cpu_freq() produces on Windows.
        telemetry = {
            "cpu_freq": [2400.0] * 40,
            "cpu_temp": [60 + i * 0.7 for i in range(40)],
            "cpu_power": [],
            "elapsed": [i * 0.2 for i in range(40)],
        }
        result = analyze_throttling(telemetry)

        self.assertFalse(result["frequency_signal_usable"])
        self.assertFalse(result["throttling_detected"],
                         "a constant clock must not produce a throttle verdict")
        self.assertEqual(result["confidence"], "unmeasured")
        self.assertTrue(any("no variance" in n for n in result["notes"]))

    def test_hot_mobile_chip_without_clock_drop_is_not_throttling(self):
        """88 C on a mobile HX part under all-core load is normal operation."""
        from ai.analysis import analyze_throttling

        telemetry = {
            "cpu_freq": [3900 + (i % 5) * 20 for i in range(40)],
            "cpu_temp": [82 + (i % 7) for i in range(40)],
            "cpu_power": [],
            "elapsed": [i * 0.2 for i in range(40)],
        }
        result = analyze_throttling(telemetry)
        self.assertFalse(result["throttling_detected"])
        self.assertIn("No throttling detected", result["summary"])

    def test_real_clock_drop_is_detected(self):
        from ai.analysis import analyze_throttling

        clocks = [4300] * 10 + [4200] * 10 + [3500] * 10 + [3000] * 10
        telemetry = {
            "cpu_freq": clocks,
            "cpu_temp": [60] * 10 + [75] * 10 + [88] * 10 + [96] * 10,
            "cpu_power": [],
            "elapsed": [i * 0.2 for i in range(40)],
        }
        result = analyze_throttling(telemetry)

        self.assertTrue(result["throttling_detected"])
        self.assertEqual(result["mechanism"], "thermal")
        self.assertGreater(result["clock_drop_pct"], 20)
        self.assertIsNotNone(result["onset_seconds"])

    def test_power_limit_throttling_detected_without_clock(self):
        """PL1 stepping down is often the earliest signal on a laptop."""
        from ai.analysis import analyze_throttling

        telemetry = {
            "cpu_freq": [],
            "cpu_power": [55] * 10 + [54] * 10 + [46] * 10 + [42] * 10,
            "cpu_temp": [65] * 20 + [86] * 20,
            "elapsed": [i * 0.2 for i in range(40)],
        }
        result = analyze_throttling(telemetry)

        self.assertTrue(result["throttling_detected"])
        self.assertGreater(result["power_drop_pct"], 12)

    def test_summary_never_contradicts_itself(self):
        """2.0.0 could say 'throttling detected' and 'sustained 100%' at once."""
        from ai.analysis import analyze_throttling

        telemetry = {
            "cpu_freq": [2400.0] * 40,
            "cpu_temp": [82 + (i % 8) for i in range(40)],
            "cpu_power": [],
            "elapsed": [i * 0.2 for i in range(40)],
        }
        summary = analyze_throttling(telemetry)["summary"]
        self.assertFalse("throttling detected" in summary.lower()
                         and "sustained 100" in summary.lower())


class TestSensorParsing(unittest.TestCase):
    def test_parses_localised_and_unit_suffixed_values(self):
        from monitoring.temp_reader import _parse_number

        self.assertEqual(_parse_number("62.0 \u00b0C"), 62.0)
        self.assertEqual(_parse_number("4,192.5 MHz"), 4192.5)
        self.assertEqual(_parse_number("45.3 W"), 45.3)
        self.assertEqual(_parse_number(55.5), 55.5)
        self.assertIsNone(_parse_number("n/a"))
        self.assertIsNone(_parse_number(None))

    def test_hybrid_core_clocks_are_separated(self):
        """
        P-cores and E-cores must not be averaged together. E-cores run several
        hundred MHz slower by design, so a shift in the work split between them
        would otherwise look exactly like throttling.
        """
        from unittest import mock
        from monitoring import temp_reader

        payload = {
            "Text": "Sensor", "Children": [{
                "Text": "CPU", "Children": [
                    {"Text": "Clocks", "Children": [
                        {"Text": "Bus Speed", "Type": "Clock", "Value": "100.0 MHz"},
                        {"Text": "P-Core #1", "Type": "Clock", "Value": "4,200.0 MHz"},
                        {"Text": "P-Core #2", "Type": "Clock", "Value": "4,000.0 MHz"},
                        {"Text": "E-Core #1", "Type": "Clock", "Value": "3,000.0 MHz"},
                    ]},
                    {"Text": "Temperatures", "Children": [
                        {"Text": "CPU Package", "Type": "Temperature", "Value": "88.0 \u00b0C"},
                        {"Text": "P-Core #1", "Type": "Temperature", "Value": "90.0 \u00b0C"},
                    ]},
                    {"Text": "Powers", "Children": [
                        {"Text": "CPU Package", "Type": "Power", "Value": "45.3 W"},
                    ]},
                ]}]}

        temp_reader.reset_breaker()
        fake = mock.Mock(status_code=200)
        fake.json.return_value = payload
        fake.raise_for_status.return_value = None

        with mock.patch("requests.get", return_value=fake):
            reading = temp_reader._read_librehardwaremonitor()

        self.assertEqual(reading["cpu_temp"], 88.0)
        self.assertEqual(reading["cpu_power_w"], 45.3)
        self.assertEqual(reading["p_core_clock_mhz"], 4100.0)
        self.assertEqual(reading["e_core_clock_mhz"], 3000.0)
        # Headline clock is the P-core mean, and excludes Bus Speed entirely.
        self.assertEqual(reading["cpu_clock_mhz"], 4100.0)
        self.assertEqual(reading["clock_source"], "librehardwaremonitor")


class TestDriftDetection(unittest.TestCase):
    """
    Standard deviation cannot tell a steady decline from random scatter, but
    they mean completely different things.
    """

    def test_measured_thermal_soak_is_called_drift(self):
        from ai.stability_engine import detect_drift

        # Real data from an i5-13450HX, five runs with 30s cooldowns.
        result = detect_drift([2060, 2109, 2081, 1983, 1936])

        self.assertEqual(result["verdict"], "drift_down")
        self.assertLess(result["total_change_pct"], -3.0)
        self.assertLess(result["rank_correlation"], -0.6)

    def test_scatter_is_not_called_drift(self):
        from ai.stability_engine import detect_drift

        result = detect_drift([2000, 2060, 1990, 2050, 2010])
        self.assertEqual(result["verdict"], "scatter")

    def test_warmup_drift_is_reported_upward(self):
        from ai.stability_engine import detect_drift

        result = detect_drift([1800, 1950, 2010, 2040, 2060])
        self.assertEqual(result["verdict"], "drift_up")
        self.assertIn("warmup", result["message"])

    def test_too_few_runs_is_honest(self):
        from ai.stability_engine import detect_drift
        self.assertEqual(detect_drift([2000, 1900])["verdict"], "insufficient_runs")

    def test_drift_penalises_repeatability_score(self):
        """
        A drifting session must not score well just because its standard
        deviation is small.
        """
        from ai.stability_engine import repeatability_from_scores

        drifting = repeatability_from_scores([2060, 2109, 2081, 1983, 1936])
        scattered = repeatability_from_scores([2000, 2060, 1990, 2050, 2010])

        self.assertEqual(drifting["drift"]["verdict"], "drift_down")
        self.assertLess(drifting["score"], scattered["score"],
                        "drift must be penalised harder than equivalent scatter")


if __name__ == "__main__":
    unittest.main()
