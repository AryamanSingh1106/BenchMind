"""
tests/test_workloads.py

Correctness tests for every registered workload.

The important property being tested is not speed, it is that each workload's
`validate` function would actually notice if the computation went wrong. A
validator that only checks for NaN passes a completely wrong answer, which is
how a benchmark ends up reporting confident nonsense.
"""

from __future__ import annotations

import unittest

from benchmarks.cpu import registry

SMALL_SCALE = 0.05


class TestWorkloadContract(unittest.TestCase):
    """Every workload must honour the setup/run/validate contract."""

    def test_all_workloads_run_and_validate(self):
        for spec in registry.WORKLOADS:
            with self.subTest(workload=spec.key):
                ctx = spec.setup_fn(SMALL_SCALE)
                output, work_units = spec.run_fn(ctx)
                self.assertGreater(work_units, 0,
                                   "work units must be positive to compute throughput")
                self.assertTrue(spec.validate_fn(output),
                                f"{spec.key} failed its own validation")

    def test_run_is_repeatable(self):
        """
        Calling run() twice on the same context must produce identical output.

        This is what allows repetitions to be compared. If a workload mutates
        its context without resetting, rep 2 does different work than rep 1 and
        the standard deviation becomes meaningless.
        """
        for spec in registry.WORKLOADS:
            with self.subTest(workload=spec.key):
                ctx = spec.setup_fn(SMALL_SCALE)
                first, units_first = spec.run_fn(ctx)
                second, units_second = spec.run_fn(ctx)

                self.assertEqual(units_first, units_second,
                                 f"{spec.key} reports different work per run")

                for key in first:
                    if key.endswith("_out"):       # large byte buffers
                        continue
                    a, b = first[key], second[key]
                    if isinstance(a, float):
                        self.assertAlmostEqual(
                            a, b, places=6,
                            msg=f"{spec.key}.{key} is not repeatable")
                    else:
                        self.assertEqual(a, b, f"{spec.key}.{key} is not repeatable")

    def test_setup_is_deterministic_across_contexts(self):
        """Two fresh contexts must produce the same answer: seeds must be fixed."""
        for spec in registry.WORKLOADS:
            with self.subTest(workload=spec.key):
                out_a, _ = spec.run_fn(spec.setup_fn(SMALL_SCALE))
                out_b, _ = spec.run_fn(spec.setup_fn(SMALL_SCALE))
                for key in out_a:
                    if key.endswith("_out"):
                        continue
                    a, b = out_a[key], out_b[key]
                    if isinstance(a, float):
                        self.assertAlmostEqual(a, b, places=6)
                    else:
                        self.assertEqual(a, b)

    def test_validators_reject_garbage(self):
        for spec in registry.WORKLOADS:
            with self.subTest(workload=spec.key):
                self.assertFalse(spec.validate_fn({}))
                self.assertFalse(spec.validate_fn(None))
                self.assertFalse(spec.validate_fn("not a dict"))


class TestValidatorsCatchCorruption(unittest.TestCase):
    """A validator that cannot detect a wrong answer is not a validator."""

    def test_integer_checksum_detects_wrong_result(self):
        from benchmarks.cpu import integer
        ctx = integer.setup(SMALL_SCALE)
        output, _ = integer.run(ctx)
        self.assertTrue(integer.validate(output))

        output["checksum"] += 1
        self.assertFalse(integer.validate(output),
                         "a one-bit change must be caught")

    def test_matrix_detects_wrong_element(self):
        from benchmarks.cpu import matrix
        ctx = matrix.setup(SMALL_SCALE)
        output, _ = matrix.run(ctx)
        self.assertTrue(matrix.validate(output))

        output["c_00"] *= 1.01     # 1% wrong, still finite and positive
        self.assertFalse(matrix.validate(output),
                         "validation must compare against a reference, not just check NaN")

    def test_compression_detects_broken_round_trip(self):
        from benchmarks.cpu import compression
        ctx = compression.setup(SMALL_SCALE)
        output, _ = compression.run(ctx)
        self.assertTrue(compression.validate(output))

        output["zlib_out"] = output["zlib_out"][:-1]
        self.assertFalse(compression.validate(output))

    def test_hashing_detects_wrong_digest(self):
        from benchmarks.cpu import hashing
        ctx = hashing.setup(SMALL_SCALE)
        output, _ = hashing.run(ctx)
        self.assertTrue(hashing.validate(output))

        output["sha256_digest"] = "0" * 64
        self.assertFalse(hashing.validate(output),
                         "a correctly shaped but wrong digest must be rejected")

    def test_branch_heavy_detects_lost_elements(self):
        """
        Corrupt the sorted buffer itself, not a precomputed summary. Since 2.1.0
        `run` returns buffers by reference and all reductions happen in
        `validate`, so this exercises the real verification path.
        """
        import numpy as np
        from benchmarks.cpu import branch_heavy

        ctx = branch_heavy.setup(SMALL_SCALE)
        output, _ = branch_heavy.run(ctx)
        self.assertTrue(branch_heavy.validate(output))

        # Alter one element: breaks both the ordering and the exact sum.
        corrupted = dict(output)
        work = output["work_out"].copy()
        work[len(work) // 2] += 1
        corrupted["work_out"] = work
        self.assertFalse(branch_heavy.validate(corrupted))

        # An out-of-range search result must also be caught.
        bad_positions = dict(output)
        bad_positions["positions_out"] = np.full(8, 10 ** 12, dtype=np.intp)
        self.assertFalse(branch_heavy.validate(bad_positions))

        # And a wrong gather sum.
        bad_gather = dict(output)
        bad_gather["expected_gather"] = output["expected_gather"] * 1.01
        self.assertFalse(branch_heavy.validate(bad_gather))

    def test_floating_point_detects_divergence(self):
        from benchmarks.cpu import floating_point
        ctx = floating_point.setup(SMALL_SCALE)
        output, _ = floating_point.run(ctx)
        self.assertTrue(floating_point.validate(output))

        output["x32_sum"] *= 2.0
        self.assertFalse(floating_point.validate(output),
                         "FP32 and FP64 run the same recurrence and must agree")


class TestCompressionCorpus(unittest.TestCase):
    def test_corpus_is_mixed_entropy(self):
        """
        The 1.x payload was a repeated pattern that deflate crushed at a silly
        ratio, so the benchmark measured run-skipping rather than compression.
        """
        import zlib
        from benchmarks.cpu.payloads import build_mixed_corpus

        data = build_mixed_corpus(1.0)
        ratio = len(data) / len(zlib.compress(data, level=6))
        self.assertGreater(ratio, 1.2, "corpus should be at least somewhat compressible")
        self.assertLess(ratio, 10.0,
                        "corpus compresses too well to represent real data")

    def test_corpus_is_deterministic(self):
        from benchmarks.cpu.payloads import build_mixed_corpus
        self.assertEqual(build_mixed_corpus(0.5), build_mixed_corpus(0.5))


class TestSingleAndMultiCoreUseSameWork(unittest.TestCase):
    """
    The CPU Index blends single-core and multi-core scores, which is only valid
    when both sides measure the same unit of work. In 1.x they did not.
    """

    def test_worker_uses_the_registry(self):
        from benchmarks.cpu import _worker

        pid = _worker.prepare("integer", 0.05)
        self.assertIsInstance(pid, int)

        result = _worker.execute("integer", 0.05, chunk_reps=1)
        self.assertTrue(result["valid"])
        self.assertGreater(result["work_units"], 0)

        spec = registry.get("integer")
        _, direct_units = spec.run_fn(spec.setup_fn(0.05))
        self.assertAlmostEqual(result["work_units"], direct_units, places=6,
                               msg="worker and single-core paths do different work")


if __name__ == "__main__":
    unittest.main()
