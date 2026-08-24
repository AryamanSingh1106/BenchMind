import time
import multiprocessing
from benchmarks.cpu.cpu_suite import run_full_cpu_suite
from benchmarks.cpu.integer import integer_workload

multiprocessing.freeze_support()


def warmup_cpu(seconds: float = 2.0):
    """Warmup CPU execution units."""
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        integer_workload(iterations=100_000)


def run_cpu_test() -> dict:
    """
    Backward-compatible entry point for CPU benchmark.
    Executes the modular CPU Benchmark Suite and returns both legacy keys
    and new BenchMind CPU Index metrics.
    """
    suite_result = run_full_cpu_suite(mode="quick")

    # Map to backward-compatible keys expected by existing callers
    return {
        "single_core_score": round(suite_result["single_core_score"]),
        "multi_core_score": round(suite_result["multi_core_score"]),
        "single_core_time": suite_result["single_core_time"],
        "multi_core_time": suite_result["multi_core_time"],
        "cores_used": suite_result["cores_used"],
        "cpu_index": round(suite_result["cpu_index"]),
        "category_scores": suite_result["category_scores"],
        "subtests": suite_result["subtests"],
        "telemetry_summary": suite_result["telemetry_summary"]
    }


if __name__ == "__main__":
    print(run_cpu_test())