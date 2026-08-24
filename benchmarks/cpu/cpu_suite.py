import time
import statistics
import logging
from typing import Dict, Any, List

from monitoring.telemetry_service import TelemetryService
from benchmarks.cpu.common import SubtestResult, calculate_cpu_index, geometric_mean
from benchmarks.cpu.single_core import run_single_core_suite
from benchmarks.cpu.multi_core import run_multi_core_suite

logger = logging.getLogger("BenchMind.CPU.Suite")


def run_full_cpu_suite(
    mode: str = "quick",
    single_weight: float = 0.4,
    multi_weight: float = 0.6
) -> Dict[str, Any]:
    """
    Master CPU Benchmark Suite Orchestrator.
    Clock Separation Rule:
      - time.perf_counter(): High-resolution performance timer used for ALL benchmark timing.
      - time.monotonic(): Used EXCLUSIVELY for background TelemetryService telemetry windowing.
    Modes:
      - 'quick': Target ~0.3s per subtest (~15-25s total run time)
      - 'full': Target ~0.8s per subtest (~30-60s total run time)
    """
    target_duration = 0.3 if mode == "quick" else 0.8

    # 1. Start centralized telemetry service
    telemetry_service = TelemetryService.get_instance()
    telemetry_service.start()

    # Clock separation: perf_counter for performance timing, monotonic for telemetry windowing
    start_perf = time.perf_counter()
    start_mono = time.monotonic()
    logger.info("Starting BenchMind CPU Suite in '%s' mode...", mode)

    # 2. Run Single-Core Suite
    single_results, single_score, single_time = run_single_core_suite(
        target_duration_per_subtest=target_duration
    )

    # 3. Run Multi-Core Suite
    multi_results, multi_score, multi_time, cores_used = run_multi_core_suite()

    end_perf = time.perf_counter()
    end_mono = time.monotonic()

    # 4. Query Telemetry Window using monotonic time
    telemetry_logs = telemetry_service.get_logs_format(
        start_time=start_mono,
        end_time=end_mono,
        use_monotonic=True
    )

    # 5. Compute BenchMind CPU Index
    cpu_index = calculate_cpu_index(
        single_core_score=single_score,
        multi_core_score=multi_score,
        single_weight=single_weight,
        multi_weight=multi_weight
    )

    # 6. Aggregate Category Breakdowns
    all_subtests: List[SubtestResult] = single_results + multi_results
    category_scores: Dict[str, float] = {}

    categories = ["integer", "floating_point", "matrix", "vector_simd", "compression", "hashing", "branch_heavy"]
    for cat in categories:
        cat_subtests = [s for s in single_results if s.category == cat and s.validation_passed]
        if cat_subtests:
            category_scores[cat] = geometric_mean([s.score for s in cat_subtests])
        else:
            category_scores[cat] = 0.0

    # Summary telemetry stats during benchmark window
    cpu_samples = telemetry_logs.get("cpu", [])
    ram_samples = telemetry_logs.get("ram", [])
    cpu_temps = [t for t in telemetry_logs.get("cpu_temp", []) if t is not None]

    telemetry_summary = {
        "sample_count": len(cpu_samples),
        "avg_cpu_utilization": round(statistics.mean(cpu_samples), 2) if cpu_samples else 0.0,
        "max_cpu_utilization": round(max(cpu_samples), 2) if cpu_samples else 0.0,
        "avg_ram_utilization": round(statistics.mean(ram_samples), 2) if ram_samples else 0.0,
        "max_cpu_temp": round(max(cpu_temps), 2) if cpu_temps else None,
        "min_cpu_temp": round(min(cpu_temps), 2) if cpu_temps else None,
    }

    return {
        "cpu_index": cpu_index,
        "single_core_score": single_score,
        "multi_core_score": multi_score,
        "single_core_time": round(single_time, 4),
        "multi_core_time": round(multi_time, 4),
        "cores_used": cores_used,
        "total_suite_time": round(end_perf - start_perf, 4),
        "scoring_weights": {
            "single_core_weight": single_weight,
            "multi_core_weight": multi_weight
        },
        "category_scores": category_scores,
        "subtests": [s.to_dict() for s in all_subtests],
        "telemetry": telemetry_logs,
        "telemetry_summary": telemetry_summary
    }
