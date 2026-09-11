"""
benchmarks/gpu/devices.py

OpenCL device discovery.

The 1.x `run_gpu_test` iterated every device on every OpenCL platform. Intel
and AMD both ship CPU OpenCL runtimes, so on a typical machine that loop would
happily "benchmark the GPU" by running the kernel on the CPU and reporting it
as a graphics device. This module filters on `cl.device_type.GPU` and records
the device class it found, so nothing is silently misattributed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.GPU.Devices")


def opencl_available() -> bool:
    try:
        import pyopencl  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _device_class(device, cl) -> str:
    dtype = device.type
    if dtype & cl.device_type.GPU:
        return "gpu"
    if dtype & cl.device_type.CPU:
        return "cpu"
    if dtype & cl.device_type.ACCELERATOR:
        return "accelerator"
    return "other"


def describe_device(device, platform, cl) -> Dict[str, Any]:
    from monitoring.system_info import normalize_device_name

    def safe(fn, default=None):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return default

    return {
        "name": normalize_device_name(device.name),
        "raw_name": device.name.strip(),
        "vendor": safe(lambda: device.vendor.strip(), "unknown"),
        "platform": safe(lambda: platform.name.strip(), "unknown"),
        "device_class": _device_class(device, cl),
        "compute_units": safe(lambda: int(device.max_compute_units)),
        "max_clock_mhz": safe(lambda: int(device.max_clock_frequency)),
        "global_mem_mb": safe(lambda: round(device.global_mem_size / (1024 ** 2), 1)),
        "local_mem_kb": safe(lambda: round(device.local_mem_size / 1024, 1)),
        "max_work_group_size": safe(lambda: int(device.max_work_group_size)),
        "driver_version": safe(lambda: device.driver_version.strip()),
        "opencl_version": safe(lambda: device.version.strip()),
        "supports_fp64": safe(lambda: "cl_khr_fp64" in device.extensions, False),
    }


def list_all_devices() -> List[Dict[str, Any]]:
    """Every OpenCL device, with its class labelled."""
    if not opencl_available():
        return []
    import pyopencl as cl

    devices: List[Dict[str, Any]] = []
    try:
        platforms = cl.get_platforms()
    except Exception as e:  # noqa: BLE001
        logger.info("No OpenCL platforms available: %s", e)
        return []

    for platform in platforms:
        try:
            for device in platform.get_devices():
                devices.append(describe_device(device, platform, cl))
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not enumerate devices on %s: %s", platform, e)
    return devices


def list_gpu_devices() -> List[Dict[str, Any]]:
    """Only real GPUs. CPU OpenCL runtimes are excluded by design."""
    return [d for d in list_all_devices() if d["device_class"] == "gpu"]


def iter_gpu_handles():
    """
    Yield (pyopencl device handle, description) pairs for real GPUs.

    Handles cannot be cached across processes, so this is a generator used at
    benchmark time rather than a stored list.
    """
    if not opencl_available():
        return
    import pyopencl as cl

    try:
        platforms = cl.get_platforms()
    except Exception as e:  # noqa: BLE001
        logger.info("No OpenCL platforms available: %s", e)
        return

    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception:  # noqa: BLE001
            continue
        for device in devices:
            if _device_class(device, cl) != "gpu":
                continue
            yield device, describe_device(device, platform, cl)
