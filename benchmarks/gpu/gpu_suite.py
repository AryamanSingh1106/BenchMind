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

# Baselines: measured on reference GPU G1, documented in
# docs/BENCHMARK_SPEC.md section 8.1. 1000 points = G1.
#
# G1 is an RTX 3050 6GB Laptop GPU, calibrated over five passes with the
# validity gate reporting 'valid'. Worst spread across the four workloads was
# 1.00%.
#
# Two internal consistency checks give confidence these are real rather than
# merely repeatable:
#
#   FP32:FP64 comes out at 1/59, against consumer Ampere's architectural
#   1/64. Before the 2.2.1 warmup fix the same device measured 1/24.7, which
#   is physically impossible for this architecture -- the FP32 figure was
#   being taken at idle clock.
#
#   Memory bandwidth is 93% of the card's ~168 GB/s theoretical, which is
#   about what a STREAM triad should achieve.
#
# To recalibrate against your own reference:
#     python -m scripts.calibrate_baselines --gpu --passes 5
GPU_BASELINES = {
    "fp32_compute": 7780.44,     # GFLOPS  spread 0.66%
    "fp64_compute": 131.96,      # GFLOPS  spread 0.37%
    "memory_bandwidth": 156.93,  # GB/s    spread 1.00%
    "matrix": 282.86,            # GFLOPS  spread 0.43%
}

REFERENCE_GPU = "BenchMind Reference G1: NVIDIA GeForce RTX 3050 6GB Laptop GPU"

DEFAULT_REPS = 9
FP_TOLERANCE = 1e-3

# A GPU takes 100 ms or more to ramp from its idle clock to a steady boost
# clock. Through 2.2.0 each kernel got exactly ONE warmup launch, so for a
# 7.5 ms kernel the clock was still ramping during the measurement: early
# repetitions slow, later ones fast, and a back-to-back calibration pass
# starting warm produced a different median entirely.
#
# Measured on an RTX 3050 Laptop over five calibration passes:
#
#     workload          launch     spread    median vs a cold single run
#     fp32_compute      0.31 ms     0.67%    3,289 vs 3,291  (stable)
#     memory_bandwidth  1.22 ms     0.12%      157 vs 157    (stable)
#     fp64_compute      7.68 ms    11.11%      133 vs 100    (+33%)
#     matrix            7.49 ms     6.32%      287 vs 215    (+33%)
#
# fp32 was stable only by accident: seven launches of 0.31 ms is 2 ms of work
# total, so the GPU never left its idle clock and was consistently slow.
GPU_WARMUP_SECONDS = 0.35
GPU_WARMUP_MAX_LAUNCHES = 20_000

# Each timed launch is tuned to about this long, so that clock jitter within a
# launch averages out instead of dominating.
TARGET_LAUNCH_SECONDS = 0.020

# Build diagnostics collected during a run, keyed by kernel name. PyOpenCL
# raises a CompilerWarning for non-empty compiler output and then discards the
# text, which meant the OpenCL compiler could be telling us something real
# about a kernel and we would never see it.
_BUILD_LOGS: Dict[str, str] = {}


def _build_program(cl, ctx, source: str, kernel_name: str, device=None):
    """
    Build an OpenCL program and capture its build log.

    Vendor compilers emit register-spill notices, occupancy hints and
    unroll decisions here. On a kernel whose measured throughput looks wrong,
    that text is usually the explanation.
    """
    program = cl.Program(ctx, source)
    try:
        program.build()
    except Exception:
        # On failure the log is the only useful diagnostic, so surface it.
        try:
            for dev in ctx.devices:
                log = program.get_build_info(dev, cl.program_build_info.LOG)
                if log and log.strip():
                    logger.error("OpenCL build FAILED for %s:\n%s", kernel_name, log.strip())
        except Exception:  # noqa: BLE001
            pass
        raise

    try:
        for dev in (ctx.devices if device is None else [device]):
            log = program.get_build_info(dev, cl.program_build_info.LOG)
            if log and log.strip():
                _BUILD_LOGS[kernel_name] = log.strip()
                logger.info("OpenCL build notes for %s: %s", kernel_name, log.strip())
    except Exception:  # noqa: BLE001
        pass

    return program


def build_logs() -> Dict[str, str]:
    """Build diagnostics from the most recent run."""
    return dict(_BUILD_LOGS)


def _score(key: str, value: float) -> float:
    baseline = GPU_BASELINES.get(key)
    if not baseline or value <= 0:
        return 0.0
    return round((value / baseline) * 1000.0, 2)


def _warmup_kernel(cl, queue, kernel, global_size, local_size=None,
                   seconds: float = GPU_WARMUP_SECONDS) -> Dict[str, Any]:
    """
    Launch a kernel repeatedly, untimed, until the GPU reaches a steady clock.

    Returns what it did so the result can report it: a device that needed 900
    launches to fill the warmup window is telling you something about its
    clock behaviour.
    """
    start = time.perf_counter()
    launches = 0
    while True:
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        launches += 1
        if (time.perf_counter() - start) >= seconds or launches >= GPU_WARMUP_MAX_LAUNCHES:
            break
    queue.finish()
    return {"warmup_launches": launches,
            "warmup_seconds": round(time.perf_counter() - start, 4)}


def _tune_inner_loops(cl, queue, kernel, global_size, loop_arg_index: int,
                      probe_loops: int = 64,
                      target_seconds: float = TARGET_LAUNCH_SECONDS) -> int:
    """
    Choose an inner loop count so one launch takes roughly `target_seconds`.

    A fixed loop count cannot suit both an RTX 3050 and an integrated Intel
    GPU: the same 128 iterations run in 0.31 ms on one and far longer on the
    other. Since the reported metric is a rate (GFLOPS), doing a different
    amount of work per device is fine, and it is what keeps every device's
    launch long enough to measure.
    """
    kernel.set_arg(loop_arg_index, np.int32(probe_loops))
    ev = cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
    elapsed = _event_seconds(ev)
    if elapsed <= 0:
        return probe_loops * 16
    scaled = int(probe_loops * (target_seconds / elapsed))
    return int(max(probe_loops, min(scaled, 500_000)))


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
    src = kernels.FP64_FMA if is_fp64 else kernels.FP32_FMA
    entry = "fp64_fma" if is_fp64 else "fp32_fma"

    # The multiplier must stay BELOW 1.0. The kernel runs a dependent chain
    # x = fma(x, y, 0.001), so y > 1 makes x grow as y^n -- fine at the old
    # fixed 128 iterations, but the loop count is now tuned per device and can
    # reach several thousand, where 1.1^8258 overflows to infinity and
    # validation fails on a perfectly healthy GPU.
    #
    # With y < 1 the chain converges toward 0.001/(1-y) and is numerically
    # stable at any loop count. Identical reasoning to the CPU floating_point
    # workload; see benchmarks/cpu/floating_point.py.
    rng = np.random.default_rng(2026)
    a = rng.uniform(0.5, 1.0, size=size).astype(dtype)
    b = rng.uniform(0.90, 0.99, size=size).astype(dtype)
    out = np.empty_like(a)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

    program = _build_program(cl, ctx, src, entry)
    kernel = cl.Kernel(program, entry)
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, out_buf)

    # First launch pays JIT compilation; do it before tuning.
    kernel.set_arg(3, np.int32(64))
    cl.enqueue_nd_range_kernel(queue, kernel, (size,), None).wait()

    inner_loops = _tune_inner_loops(cl, queue, kernel, (size,), loop_arg_index=3)
    kernel.set_arg(3, np.int32(inner_loops))

    # Untimed warmup by DURATION, so the clock has settled before timing.
    warmup = _warmup_kernel(cl, queue, kernel, (size,))

    durations: List[float] = []
    for _ in range(reps):
        ev = cl.enqueue_nd_range_kernel(queue, kernel, (size,), None)
        durations.append(_event_seconds(ev))
    queue.finish()

    cl.enqueue_copy(queue, out, out_buf).wait()

    # CPU reference over a small sample: same dependent FMA chain. Untimed,
    # and cheap because it is vectorized over the sample rather than looped
    # per element.
    sample = min(128, size)
    ref = a[:sample].astype(np.float64).copy()
    yv = b[:sample].astype(np.float64)
    for _ in range(inner_loops):
        ref = ref * yv + 0.001
    got = out[:sample].astype(np.float64)
    rel_err = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)))
    finite = bool(np.all(np.isfinite(out[:sample])))
    # FP32 accumulates visible rounding over thousands of dependent steps, so
    # its tolerance is looser than FP64's. Both are far tighter than the
    # difference between a correct result and a wrong one.
    valid = finite and rel_err < (2e-2 if not is_fp64 else 1e-6)

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
        "launch_ms": round(median * 1000, 3),
        **warmup,
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

    program = _build_program(cl, ctx, kernels.MEMORY_TRIAD, "triad")
    kernel = cl.Kernel(program, "triad")
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, out_buf)
    kernel.set_arg(3, scalar)

    warmup = _warmup_kernel(cl, queue, kernel, (size,))

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
        "launch_ms": round(median * 1000, 3),
        **warmup,
    }


def _bench_matrix(cl, ctx, queue, reps: int = DEFAULT_REPS) -> Dict[str, Any]:
    n = 1024      # multiple of the 16x16 tile size
    rng = np.random.default_rng(4242)
    A = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
    B = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
    C = np.empty((n, n), dtype=np.float32)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    program = _build_program(cl, ctx, kernels.MATRIX_MULTIPLY, "matmul")
    kernel = cl.Kernel(program, "matmul")
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, c_buf)
    kernel.set_arg(3, np.int32(n))

    global_size = (n, n)
    local_size = (16, 16)

    warmup = _warmup_kernel(cl, queue, kernel, global_size, local_size)

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
        "launch_ms": round(median * 1000, 3),
        **warmup,
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

    _BUILD_LOGS.clear()

    ctx = cl.Context([device])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    workloads: List[Dict[str, Any]] = []
    for fn in (
        lambda: _bench_fp_compute(cl, ctx, queue, device_info, np.float32, reps),
        lambda: _bench_fp_compute(cl, ctx, queue, device_info, np.float64, reps),
        lambda: _bench_memory_bandwidth(cl, ctx, queue, reps),
        lambda: _bench_matrix(cl, ctx, queue, reps),
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
        "build_logs": build_logs(),
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


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibrate_gpu(device_index: Optional[int] = None,
                  passes: int = 5,
                  reps: int = DEFAULT_REPS) -> Dict[str, Any]:
    """
    Measure a reference GPU's raw metrics across several passes.

    Mirrors the CPU calibration contract: report per-workload medians together
    with their spread, so the caller can refuse to adopt a baseline captured
    from noisy data. GPU_BASELINES shipped as invented placeholders, which made
    every GPU score a ratio against a number nobody had ever measured.

    `device_index` selects from `list_gpu_devices()`; the default picks the
    device with the most compute units, which on a laptop means the discrete
    GPU rather than the integrated one.
    """
    if not opencl_available():
        return {"status": "unavailable",
                "reason": "PyOpenCL is not installed."}

    handles = list(iter_gpu_handles())
    if not handles:
        return {"status": "no_gpu_found",
                "reason": "No OpenCL device of type GPU was found."}

    if device_index is None:
        device_index = max(
            range(len(handles)),
            key=lambda i: handles[i][1].get("compute_units") or 0,
        )
    if not 0 <= device_index < len(handles):
        return {"status": "bad_device_index",
                "reason": f"device_index {device_index} outside 0..{len(handles) - 1}"}

    device, info = handles[device_index]
    logger.info("Calibrating GPU: %s (%d passes)", info.get("name"), passes)

    samples: Dict[str, List[float]] = {}
    skipped: List[str] = []

    for p in range(1, passes + 1):
        logger.info("  GPU pass %d/%d", p, passes)
        result = benchmark_device(device, info, reps=reps)
        for w in result.get("workloads", []):
            key = w.get("key")
            if not key or key == "transfer":
                continue
            if w.get("status") == "skipped":
                if key not in skipped:
                    skipped.append(key)
                continue
            if w.get("validation_passed") and w.get("raw_metric_value"):
                samples.setdefault(key, []).append(w["raw_metric_value"])

    measured: Dict[str, Dict[str, Any]] = {}
    for key, values in samples.items():
        median = statistics.median(values)
        spread = (statistics.stdev(values) / median * 100.0) if len(values) > 1 else 0.0
        measured[key] = {
            "median": round(median, 2),
            "spread_pct": round(spread, 2),
            "passes": len(values),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    return {
        "status": "ok",
        "device": info,
        "passes": passes,
        "measured": measured,
        "skipped": skipped,
        "build_logs": build_logs(),
    }
