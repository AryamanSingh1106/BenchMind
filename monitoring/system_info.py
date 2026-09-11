"""
monitoring/system_info.py

Cross-platform system identification.

1.x shelled out to PowerShell with `shell=True` to get the CPU name, which
meant the function returned "Unknown CPU" on every non-Windows machine, and
also never populated an `os` key even though db/crud.py read one.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger("BenchMind.SystemInfo")


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    """Run a command without a shell. Returns stripped stdout or None."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, shell=False)
        value = (out.stdout or "").strip()
        return value or None
    except Exception as e:  # noqa: BLE001
        logger.debug("Command %s failed: %s", cmd[0], e)
        return None


def get_cpu_name() -> str:
    system = platform.system()

    if system == "Windows":
        value = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
        ])
        if value:
            return value.splitlines()[0].strip()

    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    elif system == "Darwin":
        value = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value

    return platform.processor() or platform.machine() or "Unknown CPU"


def get_gpu_names() -> List[str]:
    """
    GPU names via OpenCL when PyOpenCL is available. Only real GPU devices are
    listed; CPU OpenCL runtimes are filtered out so a CPU never shows up in the
    GPU list.
    """
    try:
        from benchmarks.gpu.devices import list_gpu_devices
        return [d["name"] for d in list_gpu_devices()]
    except Exception as e:  # noqa: BLE001
        logger.debug("GPU enumeration unavailable: %s", e)
        return []


def _cpu_freq() -> Dict[str, Optional[float]]:
    try:
        freq = psutil.cpu_freq()
    except Exception:  # noqa: BLE001
        freq = None
    if freq is None:
        return {"current_mhz": None, "min_mhz": None, "max_mhz": None}
    return {
        "current_mhz": round(freq.current, 1) if freq.current else None,
        "min_mhz": round(freq.min, 1) if freq.min else None,
        "max_mhz": round(freq.max, 1) if freq.max else None,
    }


def get_system_info(include_gpus: bool = True) -> Dict[str, Any]:
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    vm = psutil.virtual_memory()

    info: Dict[str, Any] = {
        "cpu": get_cpu_name(),
        "physical_cores": physical,
        "logical_cores": logical,
        "smt_enabled": bool(physical and logical and logical > physical),
        "ram": round(vm.total / (1024 ** 3), 2),
        "ram_available": round(vm.available / (1024 ** 3), 2),
        "os": f"{platform.system()} {platform.release()}",
        "os_detail": platform.platform(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "cpu_freq": _cpu_freq(),
    }

    if include_gpus:
        gpus = get_gpu_names()
        info["gpus"] = gpus
        info["gpu_count"] = len(gpus)
    else:
        info["gpus"] = []
        info["gpu_count"] = 0

    return info


def normalize_device_name(name: str) -> str:
    """
    Tidy vendor device strings for display.

    Generalized from the 1.x special case that hardcoded one Intel codename.
    """
    name = (name or "").strip()
    codename_map = {
        r"raptorlake": "Intel UHD Graphics (Raptor Lake)",
        r"alderlake": "Intel UHD Graphics (Alder Lake)",
        r"tigerlake": "Intel Iris Xe Graphics (Tiger Lake)",
        r"meteorlake": "Intel Arc Graphics (Meteor Lake)",
    }
    lowered = name.lower().replace(" ", "").replace("-", "")
    for pattern, pretty in codename_map.items():
        if re.search(pattern, lowered):
            return pretty
    return re.sub(r"\s+", " ", name)
