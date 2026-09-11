"""
monitoring/environment.py

Environment fingerprinting.

This is the most important single addition in BenchMind 2.0, and it exists
because of an uncomfortable fact about the whole design: BenchMind measures
your machine *as a Python compute host*, not the bare silicon.

  * the matrix score is really your BLAS build (OpenBLAS vs MKL vs reference)
  * the hashing score is really your OpenSSL build (SHA extensions or not)
  * the compression score is really your zlib and bz2 builds
  * the interpreter score is literally CPython's dispatch speed

Upgrade NumPy and the matrix number moves with no hardware change. That does
not make the benchmark worthless; it makes it a benchmark of a stack. But it
does mean two scores are only comparable when the stacks match.

So every result carries a fingerprint, and `fingerprint_hash` is what the
history and leaderboard layers key on. Two runs with different hashes are
never presented as a like-for-like comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional

from _version import __version__

logger = logging.getLogger("BenchMind.Environment")

_CACHE: Optional[Dict[str, Any]] = None


def _blas_backends() -> List[Dict[str, Any]]:
    try:
        import threadpoolctl
        return [
            {
                "internal_api": b.get("internal_api"),
                "user_api": b.get("user_api"),
                "version": b.get("version"),
                "threading_layer": b.get("threading_layer"),
                "num_threads": b.get("num_threads"),
                "architecture": b.get("architecture"),
            }
            for b in threadpoolctl.threadpool_info()
        ]
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not query BLAS backends: %s", e)
        return []


def _numpy_info() -> Dict[str, Any]:
    try:
        import numpy as np
        info: Dict[str, Any] = {"version": np.__version__}
        try:
            cfg = np.show_config(mode="dicts")  # NumPy >= 2.0
            build = cfg.get("Build Dependencies", {}) if isinstance(cfg, dict) else {}
            blas = build.get("blas", {}) if isinstance(build, dict) else {}
            info["blas_name"] = blas.get("name")
            info["blas_version"] = blas.get("version")
            simd = cfg.get("SIMD Extensions", {}) if isinstance(cfg, dict) else {}
            info["simd_baseline"] = simd.get("baseline")
            info["simd_found"] = simd.get("found")
        except Exception:  # noqa: BLE001
            pass
        return info
    except Exception as e:  # noqa: BLE001
        return {"version": None, "error": str(e)}


def _openssl_version() -> Optional[str]:
    try:
        import ssl
        return ssl.OPENSSL_VERSION
    except Exception:  # noqa: BLE001
        return None


def _zlib_version() -> Optional[str]:
    try:
        import zlib
        return getattr(zlib, "ZLIB_RUNTIME_VERSION", None) or zlib.ZLIB_VERSION
    except Exception:  # noqa: BLE001
        return None


def _power_state() -> Dict[str, Any]:
    """
    Battery and AC state. This matters enormously on laptops: running the same
    benchmark on battery can halve the score, and a result gathered on battery
    should never be compared against one gathered on mains.
    """
    try:
        import psutil
        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        battery = None

    if battery is None:
        return {"has_battery": False, "on_ac_power": True, "battery_percent": None}
    return {
        "has_battery": True,
        "on_ac_power": bool(battery.power_plugged),
        "battery_percent": round(float(battery.percent), 1),
    }


def _windows_power_plan() -> Optional[str]:
    if platform.system() != "Windows":
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True, text=True, timeout=5, shell=False,
        )
        line = (out.stdout or "").strip()
        if "(" in line and ")" in line:
            return line[line.rfind("(") + 1:line.rfind(")")]
        return line or None
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not read Windows power plan: %s", e)
        return None


def get_environment_fingerprint(refresh: bool = False) -> Dict[str, Any]:
    """
    Full environment description plus a stable hash over the parts that affect
    performance comparability.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        # Power state can change mid-session, so refresh that part every call.
        _CACHE = dict(_CACHE)
        _CACHE["power"] = _power_state()
        return _CACHE

    numpy_info = _numpy_info()

    fingerprint: Dict[str, Any] = {
        "benchmind_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build": " ".join(platform.python_build()),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "numpy": numpy_info,
        "blas_backends": _blas_backends(),
        "openssl": _openssl_version(),
        "zlib": _zlib_version(),
        "power": _power_state(),
        "windows_power_plan": _windows_power_plan(),
        "env_thread_vars": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            if os.environ.get(k) is not None
        },
    }

    fingerprint["fingerprint_hash"] = compute_fingerprint_hash(fingerprint)
    _CACHE = fingerprint
    return fingerprint


def compute_fingerprint_hash(fingerprint: Dict[str, Any]) -> str:
    """
    Hash only the fields that change what a score means.

    Deliberately excluded: battery percentage, executable path, OS patch
    version. Those vary run to run without making results incomparable.
    Deliberately included: Python version, NumPy version, BLAS backend and
    version, OpenSSL, zlib, CPU architecture, and whether thread-count
    environment overrides are set.
    """
    comparable = {
        "python": fingerprint.get("python", {}).get("version"),
        "implementation": fingerprint.get("python", {}).get("implementation"),
        "os": fingerprint.get("os", {}).get("system"),
        "machine": fingerprint.get("os", {}).get("machine"),
        "numpy": fingerprint.get("numpy", {}).get("version"),
        "blas": sorted(
            f"{b.get('internal_api')}:{b.get('version')}"
            for b in fingerprint.get("blas_backends", [])
        ),
        "openssl": fingerprint.get("openssl"),
        "zlib": fingerprint.get("zlib"),
        "thread_vars": fingerprint.get("env_thread_vars", {}),
        "benchmind_version": fingerprint.get("benchmind_version"),
    }
    blob = json.dumps(comparable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def fingerprints_comparable(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True only when two runs can be legitimately compared."""
    return a.get("fingerprint_hash") == b.get("fingerprint_hash")


def describe_fingerprint(fingerprint: Dict[str, Any]) -> str:
    py = fingerprint.get("python", {}).get("version", "?")
    npv = fingerprint.get("numpy", {}).get("version", "?")
    blas = fingerprint.get("blas_backends", [])
    blas_desc = blas[0].get("internal_api", "unknown") if blas else "unknown"
    osys = fingerprint.get("os", {}).get("system", "?")
    return f"Python {py} / NumPy {npv} / BLAS {blas_desc} / {osys}"
