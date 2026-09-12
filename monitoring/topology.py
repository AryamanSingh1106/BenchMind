"""
monitoring/topology.py

CPU topology detection: physical cores, SMT siblings, and performance versus
efficiency cores.

Why this exists
---------------
Through 2.0.1, single-core pinning targeted `cpu_affinity()[0]` — logical core
0. That is the worst available choice on Windows, for two separate reasons.

1. CORE 0 HANDLES INTERRUPTS.
   Windows directs a disproportionate share of interrupts and DPCs (network,
   storage, timers) to processor 0. Pinning a benchmark there measures the
   benchmark plus whatever the system happened to be doing. It hurts short
   workloads worst, because one interrupt burst is a larger fraction of a short
   measurement.

   This was visible in real calibration data from an i5-13450HX:

       integer          spread 36.92%   (shortest workload)
       branch_heavy     spread 10.74%
       floating_point   spread  8.38%
       matrix           spread  1.18%   (longest workload)

   A 37% spread is not a measurement. Matrix at 1.18% is what the same machine
   produces when the run is long enough to average the interference out.

2. CORE 0 IS AN SMT SIBLING.
   On a hybrid Intel part, logical 0 and logical 1 are the two threads of the
   same physical P-core. Pinning to logical 0 while Windows schedules anything
   onto logical 1 means the "isolated" single-core run is sharing execution
   resources with another process.

So BenchMind now picks a physical performance core away from core 0, and
records which core it chose and how confident the detection was.

What this cannot do
-------------------
Setting *our* affinity does not stop the OS scheduling other work onto the SMT
sibling of our chosen core. Preventing that needs system-wide affinity changes
or a driver, neither of which belongs in a benchmark. The sibling is reported
so the limitation is visible rather than hidden.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.Topology")

RELATION_PROCESSOR_CORE = 0
LTP_PC_SMT = 0x1


@dataclass
class PhysicalCore:
    index: int                      # ordinal among physical cores
    logical_processors: List[int]   # logical ids belonging to this core
    efficiency_class: int = 0       # higher is faster; 0 on non-hybrid CPUs
    smt: bool = False

    @property
    def primary(self) -> int:
        return self.logical_processors[0]

    @property
    def sibling(self) -> Optional[int]:
        return self.logical_processors[1] if len(self.logical_processors) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Topology:
    physical_cores: List[PhysicalCore] = field(default_factory=list)
    source: str = "unknown"          # windows | linux | fallback
    confidence: str = "low"          # high | medium | low
    hybrid: bool = False

    @property
    def logical_count(self) -> int:
        return sum(len(c.logical_processors) for c in self.physical_cores)

    @property
    def performance_cores(self) -> List[PhysicalCore]:
        """Cores in the fastest efficiency class. All of them on a non-hybrid CPU."""
        if not self.physical_cores:
            return []
        best = max(c.efficiency_class for c in self.physical_cores)
        return [c for c in self.physical_cores if c.efficiency_class == best]

    @property
    def efficiency_cores(self) -> List[PhysicalCore]:
        if not self.hybrid:
            return []
        best = max(c.efficiency_class for c in self.physical_cores)
        return [c for c in self.physical_cores if c.efficiency_class < best]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "hybrid": self.hybrid,
            "physical_core_count": len(self.physical_cores),
            "logical_count": self.logical_count,
            "performance_core_count": len(self.performance_cores),
            "efficiency_core_count": len(self.efficiency_cores),
            "cores": [c.to_dict() for c in self.physical_cores],
        }

    def describe(self) -> str:
        if not self.physical_cores:
            return "CPU topology unknown."
        if self.hybrid:
            return (f"{len(self.performance_cores)}P + {len(self.efficiency_cores)}E cores, "
                    f"{self.logical_count} threads (detected via {self.source}).")
        smt = " with SMT" if any(c.smt for c in self.physical_cores) else ""
        return (f"{len(self.physical_cores)} physical cores{smt}, "
                f"{self.logical_count} threads (detected via {self.source}).")


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def _detect_windows() -> Optional[Topology]:
    """
    Parse GetLogicalProcessorInformationEx(RelationProcessorCore).

    The buffer is a sequence of variable-length records. For a processor-core
    record the layout on x64 is:

        0   DWORD Relationship
        4   DWORD Size
        8   BYTE  Flags            (LTP_PC_SMT)
        9   BYTE  EfficiencyClass  (higher = performance core)
        10  BYTE  Reserved[20]
        30  WORD  GroupCount
        32  GROUP_AFFINITY GroupMask[]   { ULONG_PTR Mask; WORD Group; WORD[3] }

    EfficiencyClass is what distinguishes P-cores from E-cores, and it is the
    only reliable way to do so: core numbering order is not guaranteed.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return None

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = kernel32.GetLogicalProcessorInformationEx
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL

        length = wintypes.DWORD(0)
        fn(RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
        if length.value == 0:
            return None

        buffer = ctypes.create_string_buffer(length.value)
        if not fn(RELATION_PROCESSOR_CORE, buffer, ctypes.byref(length)):
            logger.debug("GetLogicalProcessorInformationEx failed: %d",
                         ctypes.get_last_error())
            return None
    except Exception as e:  # noqa: BLE001
        logger.debug("Windows topology query unavailable: %s", e)
        return None

    import struct

    raw = buffer.raw
    offset = 0
    cores: List[PhysicalCore] = []

    try:
        while offset < length.value:
            relationship, size = struct.unpack_from("<II", raw, offset)
            if size == 0:
                break
            if relationship == RELATION_PROCESSOR_CORE:
                flags = raw[offset + 8]
                efficiency_class = raw[offset + 9]
                group_count = struct.unpack_from("<H", raw, offset + 30)[0]

                logical: List[int] = []
                for g in range(max(1, group_count)):
                    base = offset + 32 + g * 16
                    if base + 10 > offset + size:
                        break
                    mask, group = struct.unpack_from("<QH", raw, base)
                    for bit in range(64):
                        if mask & (1 << bit):
                            logical.append(group * 64 + bit)

                if logical:
                    cores.append(PhysicalCore(
                        index=len(cores),
                        logical_processors=sorted(logical),
                        efficiency_class=int(efficiency_class),
                        smt=bool(flags & LTP_PC_SMT) or len(logical) > 1,
                    ))
            offset += size
    except (struct.error, IndexError) as e:
        logger.debug("Could not parse processor information buffer: %s", e)
        return None

    if not cores:
        return None

    classes = {c.efficiency_class for c in cores}
    return Topology(physical_cores=cores, source="windows", confidence="high",
                    hybrid=len(classes) > 1)


# --------------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------------
def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _detect_linux() -> Optional[Topology]:
    """
    Read sysfs. `thread_siblings_list` groups logical CPUs onto physical cores;
    `cpu_capacity` (or the intel_core / intel_atom type directories on newer
    kernels) distinguishes performance from efficiency cores.
    """
    base = "/sys/devices/system/cpu"
    if not os.path.isdir(base):
        return None

    try:
        cpu_ids = sorted(
            int(name[3:]) for name in os.listdir(base)
            if name.startswith("cpu") and name[3:].isdigit()
        )
    except OSError:
        return None
    if not cpu_ids:
        return None

    groups: Dict[str, List[int]] = {}
    capacities: Dict[str, int] = {}

    for cpu in cpu_ids:
        topo = f"{base}/cpu{cpu}/topology"
        siblings_path = f"{topo}/thread_siblings_list"
        try:
            with open(siblings_path, "r", encoding="utf-8") as fh:
                key = fh.read().strip()
        except OSError:
            key = str(cpu)
        groups.setdefault(key, []).append(cpu)
        capacity = _read_int(f"{base}/cpu{cpu}/cpu_capacity")
        if capacity is not None:
            capacities[key] = capacity

    cores: List[PhysicalCore] = []
    for index, (key, logical) in enumerate(sorted(groups.items(),
                                                  key=lambda kv: min(kv[1]))):
        cores.append(PhysicalCore(
            index=index,
            logical_processors=sorted(logical),
            efficiency_class=capacities.get(key, 0),
            smt=len(logical) > 1,
        ))

    if not cores:
        return None

    # Normalise raw capacity values into ordered classes.
    distinct = sorted({c.efficiency_class for c in cores})
    if len(distinct) > 1:
        for core in cores:
            core.efficiency_class = distinct.index(core.efficiency_class)

    return Topology(physical_cores=cores, source="linux", confidence="high",
                    hybrid=len(distinct) > 1)


# --------------------------------------------------------------------------
# Fallback
# --------------------------------------------------------------------------
def _detect_fallback() -> Topology:
    """
    Guess from physical and logical counts when no real source is available.

    Assumes consecutive logical ids pair onto a physical core, which holds on
    Windows and on x86 Linux but is not guaranteed. Marked low confidence, and
    callers should say so rather than presenting it as detected fact.
    """
    try:
        import psutil
        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:  # noqa: BLE001
        logical = os.cpu_count() or 1
        physical = logical

    cores: List[PhysicalCore] = []
    if physical and logical and logical >= physical * 2:
        per_core = logical // physical
        for i in range(physical):
            cores.append(PhysicalCore(
                index=i,
                logical_processors=list(range(i * per_core, (i + 1) * per_core)),
                smt=True,
            ))
    else:
        for i in range(logical):
            cores.append(PhysicalCore(index=i, logical_processors=[i], smt=False))

    return Topology(physical_cores=cores, source="fallback",
                    confidence="low", hybrid=False)


_CACHE: Optional[Topology] = None


def get_topology(refresh: bool = False) -> Topology:
    """Detect once and cache; topology does not change while the process runs."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    detector = _detect_windows if platform.system() == "Windows" else _detect_linux
    topology = None
    try:
        topology = detector()
    except Exception as e:  # noqa: BLE001
        logger.debug("Topology detection raised: %s", e)

    if topology is None or not topology.physical_cores:
        topology = _detect_fallback()
        logger.info("Using fallback CPU topology; core selection is a guess.")

    _CACHE = topology
    logger.info("Topology: %s", topology.describe())
    return topology


def select_benchmark_core(topology: Optional[Topology] = None) -> Dict[str, Any]:
    """
    Choose the best logical processor to pin a single-core benchmark to.

    Policy, in order:
      1. Prefer a performance core. On a hybrid CPU an E-core would understate
         single-thread performance by a wide margin.
      2. Avoid physical core 0, which fields most interrupts and DPCs on
         Windows. This is the change that matters most for short workloads.
      3. Among the rest, take the highest-numbered performance core: interrupt
         affinity concentrates on low-numbered processors.
      4. Pin to that core's primary logical processor.

    Returns the choice plus the reasoning, so the benchmark result can record
    what was actually done rather than what was intended.
    """
    topology = topology or get_topology()
    candidates = topology.performance_cores or topology.physical_cores

    if not candidates:
        return {"logical_id": None, "reason": "no topology available",
                "core_type": "unknown", "confidence": "low",
                "avoided_core_zero": False, "smt_sibling": None}

    # Rule 2: drop core 0 unless it is all we have.
    non_zero = [c for c in candidates if c.index != 0 and 0 not in c.logical_processors]
    avoided_zero = bool(non_zero)
    pool = non_zero or candidates

    chosen = pool[-1]        # Rule 3
    core_type = "performance" if (topology.hybrid and chosen in topology.performance_cores) \
        else ("physical" if not topology.hybrid else "efficiency")

    reasons = []
    if topology.hybrid:
        reasons.append(f"picked a {core_type} core")
    if avoided_zero:
        reasons.append("avoided core 0, which handles interrupts on Windows")
    else:
        reasons.append("core 0 could not be avoided")
    if chosen.sibling is not None:
        reasons.append(f"logical {chosen.sibling} is its SMT sibling and cannot be reserved")

    return {
        "logical_id": chosen.primary,
        "physical_core_index": chosen.index,
        "core_type": core_type,
        "smt_sibling": chosen.sibling,
        "avoided_core_zero": avoided_zero,
        "confidence": topology.confidence,
        "topology_source": topology.source,
        "reason": "; ".join(reasons),
    }
