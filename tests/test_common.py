"""
tests/test_common.py

Tests for the measurement core.

Deliberately absent: any assertion about how FAST something is. CI runners are
noisy shared virtual machines where timings vary by 3x between runs. Asserting
on scores in CI produces flaky failures that teach the team to ignore the test
suite. Speed is verified by scripts/repeatability.py on real hardware; CI only
verifies correctness.
"""

from __future__ import annotations

import unittest

from benchmarks.cpu.common import (
    CATEGORY_BASELINES,
    RUNTIME_BOUND_CATEGORIES,
    calculate_cpu_index,
    calculate_subtest_score,
    combine_ci_geometric,
    combined_index_ci,
    confidence_interval_pct,
    geometric_mean,
    run_timed_subtest,
    scores_are_distinguishable,
    t_critical_95,
)
from benchmarks.cpu import registry


class TestStatistics(unittest.TestCase):
    def test_geometric_mean(self):
        self.assertEqual(geometric_mean([100.0, 10000.0]), 1000.0)
        self.assertEqual(geometric_mean([8.0, 8.0, 8.0]), 8.0)

    def test_geometric_mean_drops_invalid(self):
        self.assertEqual(geometric_mean([0.0, 100.0, 100.0]), 100.0)
        self.assertEqual(geometric_mean([float("nan"), 50.0, 50.0]), 50.0)
        self.assertEqual(geometric_mean([]), 0.0)
        self.assertEqual(geometric_mean([-5.0]), 0.0)

    def test_confidence_interval_needs_two_samples(self):
        self.assertEqual(confidence_interval_pct([100.0]), 0.0)
        self.assertEqual(confidence_interval_pct([]), 0.0)

    def test_confidence_interval_zero_for_identical_samples(self):
        self.assertEqual(confidence_interval_pct([50.0] * 6), 0.0)

    def test_confidence_interval_grows_with_spread(self):
        tight = confidence_interval_pct([100, 101, 99, 100, 100])
        loose = confidence_interval_pct([100, 140, 60, 130, 70])
        self.assertLess(tight, loose)

    def test_confidence_interval_narrows_with_more_samples(self):
        """Same relative spread, more samples -> tighter interval on the mean."""
        few = confidence_interval_pct([90, 110] * 2)
        many = confidence_interval_pct([90, 110] * 10)
        self.assertLess(many, few)

    def test_t_critical_falls_back_sanely(self):
        self.assertEqual(t_critical_95(4), 2.776)
        self.assertAlmostEqual(t_critical_95(500), 1.96)
        self.assertEqual(t_critical_95(0), float("inf"))

    def test_combine_ci_geometric_shrinks(self):
        """Averaging independent measurements reduces relative uncertainty."""
        combined = combine_ci_geometric([4.0, 4.0, 4.0, 4.0])
        self.assertLess(combined, 4.0)
        self.assertAlmostEqual(combined, 2.0, places=1)

    def test_combined_index_ci(self):
        ci = combined_index_ci(3.0, 1.0, 1000.0, 2000.0)
        self.assertGreater(ci, 0.0)
        self.assertLess(ci, 3.0)


class TestDistinguishability(unittest.TestCase):
    """BenchMind must refuse to call a difference real when it is not."""

    def test_overlapping_intervals_are_a_tie(self):
        self.assertFalse(scores_are_distinguishable(1000, 5.0, 1030, 5.0))

    def test_separated_intervals_are_distinguishable(self):
        self.assertTrue(scores_are_distinguishable(1000, 1.0, 1200, 1.0))

    def test_symmetric(self):
        self.assertEqual(
            scores_are_distinguishable(1000, 1.0, 1200, 1.0),
            scores_are_distinguishable(1200, 1.0, 1000, 1.0),
        )

    def test_wide_interval_swallows_large_gap(self):
        self.assertFalse(scores_are_distinguishable(1000, 30.0, 1250, 30.0))


class TestScoring(unittest.TestCase):
    def test_reference_metric_scores_1000(self):
        for category, baseline in CATEGORY_BASELINES.items():
            self.assertAlmostEqual(
                calculate_subtest_score(category, baseline), 1000.0, places=1,
                msg=f"{category} should score 1000 at its own baseline")

    def test_unknown_category_scores_zero(self):
        """An unregistered category must not silently fall back to a guess."""
        self.assertEqual(calculate_subtest_score("not_a_category", 500.0), 0.0)

    def test_zero_metric_scores_zero(self):
        self.assertEqual(calculate_subtest_score("integer", 0.0), 0.0)

    def test_cpu_index_weighting(self):
        self.assertEqual(calculate_cpu_index(1000, 2000, 0.4, 0.6), 1600.0)

    def test_every_workload_has_a_baseline(self):
        for spec in registry.WORKLOADS:
            self.assertIn(spec.category, CATEGORY_BASELINES,
                          f"{spec.key} has no baseline; its score would be suppressed")


class TestRegistry(unittest.TestCase):
    def test_keys_unique(self):
        keys = [w.key for w in registry.WORKLOADS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_runtime_bound_excluded_from_index(self):
        index_categories = {w.category for w in registry.INDEX_WORKLOADS}
        self.assertFalse(index_categories & RUNTIME_BOUND_CATEGORIES)

    def test_interpreter_is_flagged(self):
        spec = registry.get("interpreter")
        self.assertTrue(spec.measures_runtime)
        self.assertFalse(spec.counted_in_index)

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            registry.get("nonexistent")

    def test_specs_are_callable(self):
        for spec in registry.WORKLOADS:
            self.assertTrue(callable(spec.setup_fn))
            self.assertTrue(callable(spec.run_fn))
            self.assertTrue(callable(spec.validate_fn))


class TestTimedRunner(unittest.TestCase):
    """The setup/run split is the single most important correctness property."""

    def test_setup_is_not_timed(self):
        import time
        from benchmarks.cpu.common import WorkloadSpec

        def slow_setup(scale):
            time.sleep(0.25)          # must NOT be charged against the metric
            return {"n": 1000}

        def fast_run(ctx):
            return {"ok": True}, 1000.0

        spec = WorkloadSpec(
            key="probe", name="Probe", category="integer",
            workload_profile="compute_bound", raw_metric_name="ops/sec",
            setup_fn=slow_setup, run_fn=fast_run, validate_fn=lambda o: o["ok"],
        )
        result = run_timed_subtest(spec, target_duration=0.01, min_reps=3, max_reps=3)

        self.assertGreater(result.setup_time, 0.2)
        self.assertLess(result.median_time, 0.05,
                        "setup leaked into the timed region")
        self.assertTrue(result.validation_passed)

    def test_failed_validation_suppresses_score(self):
        from benchmarks.cpu.common import WorkloadSpec

        spec = WorkloadSpec(
            key="bad", name="Bad", category="integer",
            workload_profile="compute_bound", raw_metric_name="ops/sec",
            setup_fn=lambda s: {}, run_fn=lambda c: ({}, 1e9),
            validate_fn=lambda o: False,
        )
        result = run_timed_subtest(spec, target_duration=0.01, min_reps=3, max_reps=3)

        self.assertEqual(result.score, 0.0,
                         "a workload that fails validation must not score")
        self.assertEqual(result.status, "failed")

    def test_setup_exception_is_reported_not_raised(self):
        from benchmarks.cpu.common import WorkloadSpec

        def boom(scale):
            raise RuntimeError("allocation failed")

        spec = WorkloadSpec(
            key="boom", name="Boom", category="integer",
            workload_profile="compute_bound", raw_metric_name="ops/sec",
            setup_fn=boom, run_fn=lambda c: ({}, 1.0), validate_fn=lambda o: True,
        )
        result = run_timed_subtest(spec)

        self.assertEqual(result.status, "failed")
        self.assertIn("allocation failed", result.error_message)
        self.assertEqual(result.score, 0.0)

    def test_warmup_runs_for_a_minimum_duration(self):
        """
        A single warmup repetition is enough for a 500 ms workload and useless
        for a 10 ms one: a laptop CPU takes hundreds of milliseconds to settle
        from peak turbo toward its sustained clock.
        """
        import time
        from benchmarks.cpu.common import WARMUP_MIN_SECONDS, WorkloadSpec

        spec = WorkloadSpec(
            key="tiny", name="Tiny", category="integer",
            workload_profile="compute_bound", raw_metric_name="ops/sec",
            setup_fn=lambda s: {}, run_fn=lambda c: (time.sleep(0.002) or {}, 100.0),
            validate_fn=lambda o: True,
        )
        result = run_timed_subtest(spec, target_duration=0.05, min_reps=3, max_reps=5)

        self.assertGreater(result.warmup_reps, 1,
                           "a 2 ms workload needs many warmup reps, not one")
        self.assertGreaterEqual(result.warmup_time, WARMUP_MIN_SECONDS * 0.8)

    def test_warmup_is_not_charged_to_the_metric(self):
        from benchmarks.cpu.common import WorkloadSpec

        spec = WorkloadSpec(
            key="probe2", name="Probe2", category="integer",
            workload_profile="compute_bound", raw_metric_name="ops/sec",
            setup_fn=lambda s: {}, run_fn=lambda c: ({}, 1000.0),
            validate_fn=lambda o: True,
        )
        result = run_timed_subtest(spec, target_duration=0.02, min_reps=3, max_reps=4)
        self.assertGreater(result.warmup_time, 0.0)
        self.assertLess(result.median_time, result.warmup_time,
                        "warmup time must be separate from the timed repetitions")

    def test_short_repetitions_are_flagged(self):
        """
        No amount of repetition fixes a structurally noisy measurement, so the
        result must say when its repetitions are too brief to be stable.
        """
        from benchmarks.cpu.common import WorkloadSpec

        spec = WorkloadSpec(
            key="instant", name="Instant", category="integer",
            workload_profile="compute_bound", raw_metric_name="ops/sec",
            setup_fn=lambda s: {}, run_fn=lambda c: ({}, 1.0),
            validate_fn=lambda o: True,
        )
        result = run_timed_subtest(spec, target_duration=0.02, min_reps=5, max_reps=6)
        self.assertTrue(result.short_rep_warning)

    def test_real_workloads_have_useful_repetition_lengths(self):
        """
        Every shipped workload must have repetitions above the stability floor.
        This is the test that caught integer and floating_point sitting at
        ~13 ms after their working sets were halved.

        BLAS must be limited to one thread here, exactly as single_core.py does
        it. Without that, `matrix` runs multithreaded: on a 16-thread machine it
        reached ~264 GFLOPS against 47 GFLOPS single-threaded, so the
        repetition came in at 10 ms and the test failed on a configuration the
        suite never actually uses.
        """
        import threadpoolctl
        from benchmarks.cpu.common import MIN_USEFUL_REP_SECONDS

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            for key in ("integer", "floating_point", "matrix", "vector_simd"):
                with self.subTest(workload=key):
                    spec = registry.get(key)
                    result = run_timed_subtest(spec, scale=1.0, target_duration=0.3,
                                               min_reps=3, max_reps=4)
                    self.assertTrue(result.validation_passed)
                    self.assertGreaterEqual(
                        result.median_time, MIN_USEFUL_REP_SECONDS * 0.75,
                        f"{key} repetitions are too short to measure stably")

    def test_reports_confidence_interval(self):
        spec = registry.get("integer")
        result = run_timed_subtest(spec, scale=0.05, target_duration=0.05,
                                   min_reps=5, max_reps=6)
        self.assertGreaterEqual(result.repetitions, 5)
        self.assertIsInstance(result.score_ci_pct, float)


if __name__ == "__main__":
    unittest.main()
