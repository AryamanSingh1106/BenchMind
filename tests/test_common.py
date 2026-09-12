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


class TestContentionHandling(unittest.TestCase):
    """
    Interference in a pinned single-thread measurement is one-directional: a
    competing thread can only make it slower. The median treats a slow
    repetition as equally likely to be signal, which is why reference machine
    R2 reported integer at 1,185 Mops/sec against a 2,106 baseline while five
    other categories sat within 2%.
    """

    def test_estimator_ignores_a_slow_tail(self):
        from benchmarks.cpu.common import robust_throughput

        clean = robust_throughput([2100.0] * 9)
        partly = robust_throughput([2100.0] * 7 + [1150.0] * 2)

        self.assertAlmostEqual(clean["estimate"], partly["estimate"], delta=1.0,
                               msg="a slow tail must not drag the estimate down")
        self.assertEqual(partly["contention_pct"], 0.0)

    def test_sustained_contention_is_detected(self):
        from benchmarks.cpu.common import CONTENTION_RETRY_PCT, robust_throughput

        result = robust_throughput([2100.0] * 3 + [1150.0] * 6)
        self.assertGreater(result["contention_pct"], CONTENTION_RETRY_PCT,
                           "widespread contention must be flagged for retry")
        self.assertGreater(result["estimate"], 1150.0)

    def test_estimator_is_not_just_the_maximum(self):
        """
        Best-of-N would flatter a machine that got one lucky repetition. The
        mean of the fastest half is robust without being optimistic.
        """
        from benchmarks.cpu.common import robust_throughput

        result = robust_throughput([3000.0] + [2000.0] * 8)
        self.assertLess(result["estimate"], 3000.0)
        self.assertGreater(result["estimate"], 2000.0)

    def test_few_samples_fall_back_to_median(self):
        from benchmarks.cpu.common import robust_throughput

        result = robust_throughput([100.0, 200.0, 150.0])
        self.assertEqual(result["estimate"], 150.0)

    def test_empty_input_is_safe(self):
        from benchmarks.cpu.common import robust_throughput
        self.assertEqual(robust_throughput([])["estimate"], 0.0)
        self.assertEqual(robust_throughput([0.0, float("nan")])["estimate"], 0.0)

    def test_result_reports_attempts_and_contention(self):
        from benchmarks.cpu import registry

        result = run_timed_subtest(registry.get("integer"), scale=0.05,
                                   target_duration=0.05, min_reps=5, max_reps=6)
        self.assertGreaterEqual(result.attempts, 1)
        self.assertLessEqual(result.attempts, 3)
        self.assertGreaterEqual(result.contention_pct, 0.0)


class TestMemoryHierarchySizing(unittest.TestCase):
    """
    Pure-function tests for the sweep's pass sizing.

    These replace an earlier pair of tests that asserted on measured timings
    and therefore flaked: on a battery-powered laptop the curve came back
    non-monotonic and the suite failed, then passed on mains. That violates
    the project's own rule that no test may assert on how fast anything runs
    (PROJECT_CONTEXT.md rule 13). A flaky test is worse than no test, because
    it teaches people to re-run instead of to look.
    """

    def test_pass_sizing_hits_the_target(self):
        from benchmarks.cpu.memory_hierarchy import gathers_for_target

        # A 0.5 ms gather needs ~50 repeats to fill a 25 ms pass.
        self.assertEqual(gathers_for_target(0.0005, target_seconds=0.025), 50)
        # A gather already longer than the target needs exactly one.
        self.assertEqual(gathers_for_target(0.040, target_seconds=0.025), 1)

    def test_pass_sizing_never_returns_zero(self):
        from benchmarks.cpu.memory_hierarchy import gathers_for_target

        for probe in (0.0, -1.0, 1e9, 1e-12):
            self.assertGreaterEqual(gathers_for_target(probe), 1)

    def test_sizing_scales_inversely_with_gather_cost(self):
        from benchmarks.cpu.memory_hierarchy import gathers_for_target

        fast = gathers_for_target(0.0002)
        slow = gathers_for_target(0.002)
        self.assertGreater(fast, slow)


class TestMemoryHierarchyStructure(unittest.TestCase):
    """
    Structural tests on a real sweep. Nothing here asserts on speed; a loaded
    or battery-powered machine changes the numbers but not the shape.
    """

    @classmethod
    def setUpClass(cls):
        from benchmarks.cpu.memory_hierarchy import measure_hierarchy
        cls.result = measure_hierarchy(sizes_mb=[0.5, 8, 64], reps=3)

    def test_every_point_is_fully_described(self):
        for point in self.result["points"]:
            for key in ("size_mb", "elements", "lookups_per_sec_millions",
                        "ns_per_lookup", "memory_ns_per_lookup",
                        "correction_reliable", "gathers_per_pass", "pass_ms"):
                self.assertIn(key, point)
            self.assertGreater(point["lookups_per_sec_millions"], 0)
            self.assertGreaterEqual(point["gathers_per_pass"], 1)

    def test_overhead_floor_is_measured_and_subtracted(self):
        floor = self.result["overhead_ns_per_lookup"]
        self.assertGreater(floor, 0.0,
                           "NumPy call overhead is real and must be quantified")
        for point in self.result["points"]:
            self.assertAlmostEqual(
                point["memory_ns_per_lookup"],
                max(0.0, point["ns_per_lookup"] - floor),
                places=1)

    def test_corrected_ratio_ignores_unreliable_points(self):
        """
        A corrected value smaller than the floor is a difference of similar
        numbers. Using one as a denominator once turned a defensible 11x cliff
        ratio into 42x.
        """
        from benchmarks.cpu.memory_hierarchy import measure_hierarchy

        # Re-derive the ratio from the reported points and check it matches.
        reliable = [p["memory_ns_per_lookup"] for p in self.result["points"]
                    if p["correction_reliable"] and p["memory_ns_per_lookup"] > 0]
        if len(reliable) > 1:
            expected = round(max(reliable) / min(reliable), 2)
            self.assertEqual(self.result["cache_cliff_ratio_corrected"], expected)
        else:
            self.assertIsNone(self.result["cache_cliff_ratio_corrected"])

    def test_largest_working_set_is_not_the_fastest(self):
        """
        The one physical claim safe to assert: a 64 MB working set cannot be
        faster than a 512 KB one. Stated with generous tolerance so a busy
        machine does not fail it.
        """
        rates = [p["lookups_per_sec_millions"] for p in self.result["points"]]
        self.assertGreater(rates[0] * 1.5, rates[-1],
                           f"largest working set measured fastest: {rates}")

    def test_is_not_in_the_cpu_index(self):
        """
        Adding it would change the index's meaning and invalidate every stored
        baseline, for information more useful as a curve than a score.
        """
        from benchmarks.cpu import registry
        self.assertNotIn("memory_hierarchy", [w.category for w in registry.WORKLOADS])

    def test_analysis_handles_a_missing_sweep(self):
        from ai.analysis import analyze_memory_hierarchy

        self.assertEqual(analyze_memory_hierarchy(None)["verdict"], "not_measured")
        self.assertEqual(analyze_memory_hierarchy({})["verdict"], "not_measured")

    def test_analysis_connects_to_branch_heavy(self):
        from ai.analysis import analyze_memory_hierarchy

        hierarchy = {
            "points": [{"size_mb": 0.5, "lookups_per_sec_millions": 400.0,
                        "ns_per_lookup": 2.5},
                       {"size_mb": 256, "lookups_per_sec_millions": 50.0,
                        "ns_per_lookup": 20.0}],
            "cache_cliff_ratio": 8.0,
            "cached_ns_per_lookup": 2.5,
            "dram_ns_per_lookup": 20.0,
            "boundaries": [{"between_mb": [8, 16], "drop_pct": 40.0}],
        }
        subtests = [{"category": "branch_heavy", "threads": 1,
                     "validation_passed": True, "score": 802.0}]

        result = analyze_memory_hierarchy(hierarchy, subtests)
        self.assertEqual(result["verdict"], "latency_limited")
        self.assertIn("branch_heavy", result["branch_heavy_note"])
        self.assertIn("802", result["branch_heavy_note"])


class TestAdaptiveChunking(unittest.TestCase):
    """
    Multi-core wave selection. Through 2.1.1 this was fixed at 2 waves, which
    left a real 6P+4E machine at +/- 5.6% multi-core against +/- 0.7%
    single-core: every repetition was dominated by whichever worker drew the
    slowest chunk.
    """

    def test_fast_workload_gets_maximum_waves(self):
        from benchmarks.cpu.multi_core import MAX_WAVES, _choose_waves
        self.assertEqual(_choose_waves(0.026), MAX_WAVES)

    def test_slow_workload_is_time_bounded(self):
        from benchmarks.cpu.multi_core import MIN_WAVES, _choose_waves
        # A 1.3 s chunk cannot afford 8 waves; 8 x 1.3 x 5 reps is over a minute.
        self.assertEqual(_choose_waves(1.287), MIN_WAVES)

    def test_waves_respect_the_time_budget(self):
        from benchmarks.cpu.multi_core import TARGET_REP_SECONDS, _choose_waves

        for chunk in (0.05, 0.135, 0.3, 0.5):
            with self.subTest(chunk=chunk):
                waves = _choose_waves(chunk)
                self.assertLessEqual(waves * chunk, TARGET_REP_SECONDS * 1.6)

    def test_clamped_at_both_ends(self):
        from benchmarks.cpu.multi_core import MAX_WAVES, MIN_WAVES, _choose_waves
        self.assertEqual(_choose_waves(0.0), MAX_WAVES)      # degenerate
        self.assertEqual(_choose_waves(1e-9), MAX_WAVES)     # absurdly fast
        self.assertEqual(_choose_waves(60.0), MIN_WAVES)     # absurdly slow

    def test_result_reports_its_imbalance_bound(self):
        """A workload forced down to MIN_WAVES must say so, not hide it."""
        import psutil
        from benchmarks.cpu import registry
        from benchmarks.cpu.multi_core import WarmPool, run_multi_core_subtest

        workers = min(2, psutil.cpu_count(logical=True) or 1)
        with WarmPool(workers) as pool:
            result = run_multi_core_subtest(
                pool, registry.get("integer"), workers, scale=0.05, reps=2)

        self.assertGreaterEqual(result.chunk_waves, 2)
        self.assertGreater(result.imbalance_bound_pct, 0.0)
        self.assertAlmostEqual(result.imbalance_bound_pct,
                               100.0 / result.chunk_waves, places=1)


if __name__ == "__main__":
    unittest.main()
