import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'db' is always discoverable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db.crud import save_system_profile, save_benchmark_result, save_timeline_logs, get_recent_benchmarks

print("--- Testing BenchMind Database Operations ---")

# 1. Mock System Info
mock_system = {
    "cpu": "13th Gen Intel(R) Core(TM) i5-13450HX",
    "physical_cores": 10,
    "logical_cores": 16,
    "ram": 15.71,
    "gpus": ["NVIDIA GeForce RTX 3050 6GB Laptop GPU", "Intel UHD Graphics"],
    "os": "Windows 11"
}

system_id = save_system_profile(mock_system)
print(f"✅ Saved System Profile with ID: {system_id}")

# 2. Mock Benchmark Results
mock_results = {
    "cpu_single_score": 1618,
    "cpu_multi_score": 12669,
    "gpu_score": 131384,
    "stability": 93.7,
    "summary": {
        "verdict": "High-performance multi-core CPU detected.",
        "thermal_health": "Optimal"
    }
}

benchmark_id = save_benchmark_result(system_id, mock_results)
print(f"✅ Saved Benchmark Run with ID: {benchmark_id}")

# 3. Mock Timeline Points
mock_timeline = [
    {"time": 0.0, "cpu": 15.0, "ram": 71.0, "cpu_temp": 64.0, "gpu_temp": 59.0},
    {"time": 1.0, "cpu": 76.0, "ram": 72.0, "cpu_temp": 72.0, "gpu_temp": 60.0},
    {"time": 2.0, "cpu": 100.0, "ram": 72.0, "cpu_temp": 78.0, "gpu_temp": 62.0}
]

save_timeline_logs(benchmark_id, mock_timeline)
print("✅ Saved Timeline Logs!")

# 4. Fetch Recent Benchmarks
history = get_recent_benchmarks(limit=5)
print(f"✅ Successfully retrieved {len(history)} record(s) from history.")