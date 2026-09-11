"""
benchmarks/gpu/gpu_suite.py

Multi-workload GPU benchmark.

Everything wrong with the 1.x GPU test, and what replaced it:

  1.x: timed with `time.time()`          -> 2.0: OpenCL event profiling, which
       (host wall clock, low resolution,       reads the device's own start and
       includes queue latency)                 end timestamps in nanoseconds

  1.x: `score = (size*loops/elapsed)/100000` -> 2.0: throughput measured in
       an arbitrary constant, exactly what          GFLOPS and GB/s, scored
       PROJECT_CONTEXT.md forbids                   against documented baselines

  1.x: result buffer never read back      -> 2.0: every kernel's output is read
       so a no-op driver would score high       back and checked against a CPU
                                                reference within FP tolerance

  1.x: iterated every OpenCL device       -> 2.0: GPUs only (see devices.py)
       including CPU runtimes

  1.x: single FMA kernel                  -> 2.0: FP32 compute, FP64 compute,
                                                memory bandwidth, matrix multiply

Host-to-device transfer is measured separately and excluded from the compute
numbers, because PCIe bandwidth is a different property from GPU throughput.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.cpu.common import confidence_interval_pct
from benchmarks.gpu import kernels
from benchmarks.gpu.devices import iter_gpu_handles, list_all_devices, opencl_available

logger = logging.getLogger("BenchMind.GPU.Suite")

# Baselines: reference GPU raw metrics, documented in docs/BENCHMARK_SPEC.md.
# 1000 points = reference. Not tuned to make anything look good.
GPU_BASELINES = {
    "fp32_compute": 4000.0,    # GFLOPS
    "fp64_compute": 120.0,     # GFLOPS
    "memory_bandwidth": 180.0,  # GB/s
    "matrix": 900.0,           # GFLOPS
}

DEFAULT_REPS = 7
FP_TOLERANCE = 1e-3


def _score(key: str, value: float) -> float:
    baseline = GPU_BASELINES.get(key)
    if not baseline or value <= 0:
        return 0.0
    return round((value / baseline) * 1000.0, 2)


def _event_seconds(event) -> float:
    """
    Kernel duration from the device's own profiling counters, in seconds.

    This is the whole reason the queue is created with PROFILING_ENABLE. Host
    wall clock around an enqueue measures submission latency and driver
    behaviour as much as it measures the kernel.
    """
    event.wait()
    return (event.profile.end - event.profile.start) * 1e-9


def _bench_fp_compute(cl, ctx, queue, device_info: Dict[str, Any],
                      dtype=np.float32, reps: int = DEFAULT_REPS) -> Dict[str, Any]:
    is_fp64 = dtype == np.float64
    if is_fp64 and not device_info.get("supports_fp64"):
        return {"name": "FP64 Compute", "status": "skipped",
                "reason": "Device does not advertise cl_khr_fp64."}

    size = 4_000_000
    inner_loops = 128
    src = kernels.FP64_FMA if is_fp64 else kernels.FP32_FMA
    entry = "fp64_fma" if is_fp64 else "fp32_fma"

    rng = np.random.default_rng(2026)
    a = rng.uniform(0.9, 1.1, size=size).astype(dtype)
    b = rng.uniform(0.9, 1.1, size=size).astype(dtype)
    out = np.empty_like(a)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

    program = cl.Program(ctx, src).build()
    kernel = cl.Kernel(program, entry)
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, out_buf)
    kernel.set_arg(3, np.int32(inner_loops))

    # Untimed warmup: first launch pays JIT compilation and allocation.
    cl.enqueue_nd_range_kernel(queue, kernel, (size,), None).wait()

    durations: List[float] = []
    for _ in range(reps):
        ev = cl.enqueue_nd_range_kernel(queue, kernel, (size,), None)
        durations.append(_event_seconds(ev))
    queue.finish()

    cl.enqueue_copy(queue, out, out_buf).wait()

    # CPU reference over a small sample: same dependent FMA chain.
    sample = min(256, size)
    ref = a[:sample].astype(np.float64).copy()
    yv = b[:sample].astype(np.float64)
    for _ in range(inner_loops):
        ref = ref * yv + 0.001
    got = out[:sample].astype(np.float64)
    rel_err = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)))
    valid = bool(np.all(np.isfinite(out[:sample]))) and rel_err < (1e-2 if not is_fp64 else 1e-6)

    median = statistics.median(durations)
    flops = 2.0 * size * inner_loops            # fma counts as 2 FLOPs
    gflops = round((flops / median) / 1e9, 2)
    throughputs = [flops / d / 1e9 for d in durations if d > 0]

    key = "fp64_compute" if is_fp64 else "fp32_compute"
    return {
        "name": "FP64 Compute (dependent FMA chain)" if is_fp64
                else "FP32 Compute (dependent FMA chain)",
        "key": key,
        "status": "passed" if valid else "failed",
        "raw_metric_name": "GFLOPS",
        "raw_metric_value": gflops,
        "ci_pct": confidence_interval_pct(throughputs),
        "score": _score(key, gflops) if valid else 0.0,
        "reps": len(durations),
        "median_kernel_seconds": round(median, 6),
        "validation_passed": valid,
        "max_relative_error": rel_err,
        "elements": size,
        "inner_loops": inner_loops,
    }


def _bench_memory_bandwidth(cl, ctx, queue, reps: int = DEFAULT_REPS) -> Dict[str, Any]:
    size = 16_000_000
    scalar = np.float32(3.0)

    rng = np.random.default_rng(31337)
    a = rng.uniform(0.0, 1.0, size=size).astype(np.float32)
    b = rng.uniform(0.0, 1.0, size=size).astype(np.float32)
    out = np.empty_like(a)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

    program = cl.Program(ctx, kernels.MEMORY_TRIAD).build()
    kernel = cl.Kernel(program, "triad")
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, out_buf)
    kernel.set_arg(3, scalar)

    cl.enqueue_nd_range_kernel(queue, kernel, (size,), None).wait()

    durations = []
    for _ in range(reps):
        ev = cl.enqueue_nd_range_kernel(queue, kernel, (size,), None)
        durations.append(_event_seconds(ev))
    queue.finish()

    cl.enqueue_copy(queue, out, out_buf).wait()

    expected = a[:1024] + float(scalar) * b[:1024]
    valid = bool(np.allclose(out[:1024], expected, rtol=FP_TOLERANCE, atol=FP_TOLERANCE))

    median = statistics.median(durations)
    bytes_moved = 3.0 * 4 * size          # two reads plus one write
    gbs = round((bytes_moved / median) / 1e9, 2)
    throughputs = [bytes_moved / d / 1e9 for d in durations if d > 0]

    return {
        "name": "Memory Bandwidth (STREAM triad)",
        "key": "memory_bandwidth",
        "status": "passed" if valid else "failed",
        "raw_metric_name": "GB/s",
        "raw_metric_value": gbs,
        "ci_pct": confidence_interval_pct(throughputs),
        "score": _score("memory_bandwidth", gbs) if valid else 0.0,
        "reps": len(durations),
        "median_kernel_seconds": round(median, 6),
        "validation_passed": valid,
        "elements": size,
    }


def _bench_matrix(cl, ctx, queue, reps: int = 5) -> Dict[str, Any]:
    n = 1024      # multiple of the 16x16 tile size
    rng = np.random.default_rng(4242)
    A = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
    B = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
    C = np.empty((n, n), dtype=np.float32)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    program = cl.Program(ctx, kernels.MATRIX_MULTIPLY).build()
    kernel = cl.Kernel(program, "matmul")
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, c_buf)
    kernel.set_arg(3, np.int32(n))

    global_size = (n, n)
    local_size = (16, 16)

    cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()

    durations = []
    for _ in range(reps):
        ev = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        durations.append(_event_seconds(ev))
    queue.finish()

    cl.enqueue_copy(queue, C, c_buf).wait()

    # Verify one row against NumPy rather than trusting the device.
    expected_row = A[0, :].astype(np.float64) @ B.astype(np.float64)
    valid = bool(np.allclose(C[0, :].astype(np.float64), expected_row, rtol=1e-2, atol=1e-2))

    median = statistics.median(durations)
    flops = 2.0 * (n ** 3)
    gflops = round((flops / median) / 1e9, 2)
    throughputs = [flops / d / 1e9 for d in durations if d > 0]

    return {
        "name": "Matrix Multiply (tiled, 1024x1024 FP32)",
        "key": "matrix",
        "status": "passed" if valid else "failed",
        "raw_metric_name": "GFLOPS",
        "raw_metric_value": gflops,
        "ci_pct": confidence_interval_pct(throughputs),
        "score": _score("matrix", gflops) if valid else 0.0,
        "reps": len(durations),
        "median_kernel_seconds": round(median, 6),
        "validation_passed": valid,
        "size": n,
    }


def _bench_transfer(cl, ctx, queue, reps: int = 5) -> Dict[str, Any]:
    """
    Host-to-device and device-to-host bandwidth, measured separately.

    PCIe throughput is a real bottleneck for GPU work but it is not a property
    of the GPU's compute units, so it never enters the compute scores.
    """
    size_mb = 128
    data = np.random.default_rng(11).random(size_mb * 1024 * 1024 // 4).astype(np.float32)
    back = np.empty_like(data)

    mf = cl.mem_flags
    buf = cl.Buffer(ctx, mf.READ_WRITE, data.nbytes)

    cl.enqueue_copy(queue, buf, data).wait()

    h2d, d2h = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        cl.enqueue_copy(queue, buf, data).wait()
        h2d.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cl.enqueue_copy(queue, back, buf).wait()
        d2h.append(time.perf_counter() - t0)

    valid = bool(np.array_equal(back, data))
    gb = data.nbytes / 1e9
    return {
        "name": "Host Transfer Bandwidth",
        "key": "transfer",
        "status": "passed" if valid else "failed",
        "host_to_device_gbs": round(gb / statistics.median(h2d), 2),
        "device_to_host_gbs": round(gb / statistics.median(d2h), 2),
        "validation_passed": valid,
        "payload_mb": size_mb,
        "score": 0.0,          # informational, deliberately not scored
    }


def benchmark_device(device, device_info: Dict[str, Any],
                     reps: int = DEFAULT_REPS) -> Dict[str, Any]:
    import pyopencl as cl

    ctx = cl.Context([device])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    workloads: List[Dict[str, Any]] = []
    for fn in (
        lambda: _bench_fp_compute(cl, ctx, queue, device_info, np.float32, reps),
        lambda: _bench_fp_compute(cl, ctx, queue, device_info, np.float64, reps),
        lambda: _bench_memory_bandwidth(cl, ctx, queue, reps),
        lambda: _bench_matrix(cl, ctx, queue),
        lambda: _bench_transfer(cl, ctx, queue),
    ):
        try:
            workloads.append(fn())
        except Exception as e:  # noqa: BLE001
            logger.error("GPU workload failed on %s: %s", device_info.get("name"), e,
                         exc_info=True)
            workloads.append({"name": "unknown", "status": "failed", "error": str(e),
                              "validation_passed": False, "score": 0.0})

    scored = [w for w in workloads
              if w.get("validation_passed") and (w.get("score") or 0) > 0]
    if scored:
        product = 1.0
        for w in scored:
            product *= w["score"]
        composite = round(product ** (1.0 / len(scored)), 2)
        composite_ci = round(
            (sum((w.get("ci_pct") or 0.0) ** 2 for w in scored) ** 0.5) / len(scored), 2)
    else:
        composite = 0.0
        composite_ci = 0.0

    return {
        "gpu_name": device_info.get("name"),
        "gpu_score": composite,
        "gpu_score_ci_pct": composite_ci,
        "device": device_info,
        "workloads": workloads,
        "workloads_passed": len(scored),
        "workloads_total": len(workloads),
    }


def run_gpu_suite(reps: int = DEFAULT_REPS) -> Dict[str, Any]:
    """
    Benchmark every real GPU on the system.

    Returns a dict rather than a bare list so the absence of a GPU, and the
    reason for it, can be reported instead of silently returning [].
    """
    if not opencl_available():
        return {
            "status": "unavailable",
            "reason": ("PyOpenCL is not installed. Install it and a vendor OpenCL "
                       "runtime to enable GPU benchmarking."),
            "devices": [],
            "all_opencl_devices": [],
        }

    all_devices = list_all_devices()
    results: List[Dict[str, Any]] = []

    for device, info in iter_gpu_handles():
        logger.info("Benchmarking GPU: %s", info.get("name"))
        try:
            results.append(benchmark_device(device, info, reps=reps))
        except Exception as e:  # noqa: BLE001
            logger.error("Skipping %s: %s", info.get("name"), e)
            results.append({
                "gpu_name": info.get("name"), "gpu_score": 0.0,
                "device": info, "workloads": [], "error": str(e),
            })

    if not results:
        non_gpu = [d["name"] for d in all_devices if d["device_class"] != "gpu"]
        return {
            "status": "no_gpu_found",
            "reason": ("No OpenCL device of type GPU was found. "
                       + (f"Non-GPU OpenCL devices present: {', '.join(non_gpu)}. "
                          "BenchMind deliberately does not benchmark CPU OpenCL "
                          "runtimes as if they were graphics devices."
                          if non_gpu else "")),
            "devices": [],
            "all_opencl_devices": all_devices,
        }

    return {
        "status": "ok",
        "devices": results,
        "all_opencl_devices": all_devices,
        "baseline_reference": GPU_BASELINES,
    }
