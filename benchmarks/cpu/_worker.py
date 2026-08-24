"""
benchmarks/cpu/_worker.py

Standalone top-level worker module for ProcessPoolExecutor on Windows.
On Windows, multiprocessing uses the 'spawn' start method. Each worker
process reimports this module, so ALL workload imports must be resolvable
at top level without triggering any process-spawning code paths.
"""


def _worker_task(workload_type: str):
    """Dispatch a single workload chunk to a worker process."""
    if workload_type == "integer":
        from benchmarks.cpu.integer import integer_workload
        return integer_workload(iterations=200_000)
    elif workload_type == "floating_point":
        from benchmarks.cpu.floating_point import floating_point_workload
        return floating_point_workload(array_size=500_000, loops=15)
    elif workload_type == "matrix":
        from benchmarks.cpu.matrix import matrix_workload
        return matrix_workload(size=384)
    elif workload_type == "compression":
        from benchmarks.cpu.compression import compression_workload
        return compression_workload(buffer_size_mb=5.0)
    elif workload_type == "hashing":
        from benchmarks.cpu.hashing import hashing_workload
        return hashing_workload(buffer_size_mb=8.0, passes=3)
    else:
        raise ValueError(f"Unknown workload_type: {workload_type}")
