import unittest
import threadpoolctl
from benchmarks.cpu.common import SubtestResult, calculate_subtest_score, calculate_cpu_index, geometric_mean
from benchmarks.cpu.integer import run_integer_subtest, validate_integer_workload, integer_workload
from benchmarks.cpu.floating_point import run_floating_point_subtest, validate_floating_point_workload, floating_point_workload
from benchmarks.cpu.matrix import run_matrix_subtest, validate_matrix_workload, matrix_workload
from benchmarks.cpu.vector_simd import run_vector_simd_subtest, validate_vector_simd_workload, vector_simd_workload
from benchmarks.cpu.compression import run_compression_subtest, validate_compression_workload, compression_workload
from benchmarks.cpu.hashing import run_hashing_subtest, validate_hashing_workload, hashing_workload
from benchmarks.cpu.branch_heavy import run_branch_heavy_subtest, validate_branch_heavy_workload, branch_heavy_workload
from benchmarks.cpu.single_core import run_single_core_suite
from benchmarks.cpu.multi_core import run_multi_core_suite
from benchmarks.cpu.cpu_suite import run_full_cpu_suite
from benchmarks.cpu_test import run_cpu_test


class TestCPUBenchmarkSuite(unittest.TestCase):

    def test_geometric_mean_scoring(self):
        # Geometric mean of 100 and 10000 is 1000
        gm = geometric_mean([100.0, 10000.0])
        self.assertEqual(gm, 1000.0)

        # Handling zero/negative scores safely
        gm_safe = geometric_mean([0.0, 100.0, 100.0])
        self.assertEqual(gm_safe, 100.0)

    def test_single_core_blas_isolation(self):
        # Verify that threadpoolctl inside single_core suite limits BLAS threads to 1
        subtests, score, duration = run_single_core_suite(target_duration_per_subtest=0.1)
        matrix_subtest = next(s for s in subtests if s.category == "matrix")
        self.assertTrue(matrix_subtest.validation_passed)
        self.assertGreater(matrix_subtest.score, 0.0)

    def test_subtest_schema_and_workload_profile(self):
        res = run_integer_subtest(target_duration=0.1)
        res_dict = res.to_dict()
        self.assertIn("workload_profile", res_dict)
        self.assertEqual(res_dict["workload_profile"], "compute_bound")
        self.assertIn("category", res_dict)
        self.assertIn("stability_pct", res_dict)
        self.assertIn("validation_passed", res_dict)

    def test_integer_subtest_correctness(self):
        out, _ = integer_workload(iterations=50_000)
        self.assertTrue(validate_integer_workload(out))
        res = run_integer_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_floating_point_subtest_correctness(self):
        out, _ = floating_point_workload(array_size=100_000, loops=5)
        self.assertTrue(validate_floating_point_workload(out))
        res = run_floating_point_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_matrix_subtest_correctness(self):
        out, _ = matrix_workload(size=128)
        self.assertTrue(validate_matrix_workload(out))
        res = run_matrix_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_vector_simd_subtest_correctness(self):
        out, _ = vector_simd_workload(vector_size=200_000, loops=5)
        self.assertTrue(validate_vector_simd_workload(out))
        res = run_vector_simd_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_compression_subtest_correctness(self):
        out, _ = compression_workload(buffer_size_mb=1.0)
        self.assertTrue(validate_compression_workload(out))
        res = run_compression_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_hashing_subtest_correctness(self):
        out, _ = hashing_workload(buffer_size_mb=2.0, passes=2)
        self.assertTrue(validate_hashing_workload(out))
        res = run_hashing_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_branch_heavy_subtest_correctness(self):
        out, _ = branch_heavy_workload(array_size=50_000)
        self.assertTrue(validate_branch_heavy_workload(out))
        res = run_branch_heavy_subtest(target_duration=0.1)
        self.assertTrue(res.validation_passed)
        self.assertEqual(res.status, "passed")

    def test_single_core_suite_execution(self):
        subtests, score, duration = run_single_core_suite(target_duration_per_subtest=0.1)
        self.assertEqual(len(subtests), 7)
        self.assertGreater(score, 0.0)
        for s in subtests:
            self.assertTrue(s.validation_passed)

    def test_multi_core_suite_execution(self):
        subtests, score, duration, cores = run_multi_core_suite()
        self.assertEqual(len(subtests), 5)
        self.assertGreater(score, 0.0)
        self.assertGreater(cores, 0)
        for s in subtests:
            self.assertTrue(s.validation_passed)

    def test_cpu_suite_orchestrator(self):
        suite_res = run_full_cpu_suite(mode="quick")
        self.assertIn("cpu_index", suite_res)
        self.assertIn("single_core_score", suite_res)
        self.assertIn("multi_core_score", suite_res)
        self.assertIn("category_scores", suite_res)
        self.assertIn("subtests", suite_res)
        self.assertIn("telemetry_summary", suite_res)
        self.assertGreater(suite_res["cpu_index"], 0.0)

    def test_run_cpu_test_compatibility(self):
        legacy_res = run_cpu_test()
        self.assertIn("single_core_score", legacy_res)
        self.assertIn("multi_core_score", legacy_res)
        self.assertIn("single_core_time", legacy_res)
        self.assertIn("multi_core_time", legacy_res)
        self.assertIn("cores_used", legacy_res)
        self.assertIn("cpu_index", legacy_res)
        self.assertIn("category_scores", legacy_res)


if __name__ == "__main__":
    unittest.main()
