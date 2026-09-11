"""
benchmarks/cpu/registry.py

The single source of truth for every CPU workload.

This file exists to fix a specific 1.x defect. The single-core suite ran
integer at 300,000 iterations and matrix at 768x768, while the multi-core
worker ran integer at 200,000 and matrix at 384x384. A 384x384 matmul lives in
a completely different cache regime than a 768x768 one. Those two numbers were
then blended 0.4/0.6 into a single "CPU Index", which made the index a
combination of two things that were not measuring the same unit of work.

Now both suites pull their workload definitions from here, at the same scale.
Thread count is the only variable that differs between them, which is the whole
point of a single-core vs multi-core comparison.

`bytes_per_run` and `arithmetic_intensity` are declared per workload and feed
the roofline analysis in ai/analysis.py.

One further rule, enforced inside each workload's `setup`:

    SCALE TIME, NOT WORKING SET.

`scale` (which the quick/standard/full modes vary) changes how many passes a
workload makes, never how large its buffers are. Shrinking a buffer moves the
workload into a different cache level, so a quick-mode score would measure
something genuinely different from a standard-mode score while looking like
the same metric. Modes are still not comparable to each other -- fewer
repetitions means a wider confidence interval -- but at least each mode is
measuring the same thing.
"""

from __future__ import annotations

from typing import Dict, List

from benchmarks.cpu import (
    branch_heavy,
    compression,
    floating_point,
    hashing,
    integer,
    interpreter,
    matrix,
    vector_simd,
)
from benchmarks.cpu.common import WorkloadSpec

_KB = 1024.0
_MB = 1024.0 * 1024.0


def _fp_bytes() -> float:
    n, loops = floating_point.ELEMENTS, floating_point.LOOPS
    # Per loop: read a32 + read/write x32 (fp32, 4B) and the fp64 equivalent (8B).
    # Cache-resident, so DRAM traffic is effectively just the working set once.
    return (3 * 4 * n) + (3 * 8 * n)


def _vec_bytes() -> float:
    return vector_simd.bytes_per_run()


def _int_bytes() -> float:
    n = integer.ELEMENTS
    return 4 * 8 * n  # base + odd + acc + tmp, int64


WORKLOADS: List[WorkloadSpec] = [
    WorkloadSpec(
        key="integer",
        name="Integer ALU Throughput (int64 SIMD)",
        category="integer",
        workload_profile="compute_bound",
        raw_metric_name="Mops/sec",
        setup_fn=integer.setup,
        run_fn=integer.run,
        validate_fn=integer.validate,
        bytes_per_run=_int_bytes(),
        working_set_bytes=_int_bytes(),
        arithmetic_intensity=(6.0 * integer.ELEMENTS * integer.LOOPS) / max(_int_bytes(), 1.0),
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="floating_point",
        name="Floating-Point FMA, L2-resident (FP32 + FP64)",
        category="floating_point",
        workload_profile="compute_bound",
        raw_metric_name="MFLOPS",
        setup_fn=floating_point.setup,
        run_fn=floating_point.run,
        validate_fn=floating_point.validate,
        bytes_per_run=_fp_bytes(),
        working_set_bytes=_fp_bytes(),
        arithmetic_intensity=(4.0 * floating_point.ELEMENTS * floating_point.LOOPS)
        / max(_fp_bytes(), 1.0),
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="matrix",
        name="Dense Matrix Multiply (BLAS dgemm)",
        category="matrix",
        workload_profile="mixed",
        raw_metric_name="GFLOPS",
        setup_fn=matrix.setup,
        run_fn=matrix.run,
        validate_fn=matrix.validate,
        bytes_per_run=3.0 * 8 * (matrix.SIZE ** 2),
        working_set_bytes=3.0 * 8 * (matrix.SIZE ** 2),
        arithmetic_intensity=(2.0 * matrix.SIZE ** 3) / max(3.0 * 8 * matrix.SIZE ** 2, 1.0),
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="vector_simd",
        name="Streaming Vector FMA, DRAM-resident",
        category="vector_simd",
        workload_profile="memory_bound",
        raw_metric_name="GFLOPS",
        setup_fn=vector_simd.setup,
        run_fn=vector_simd.run,
        validate_fn=vector_simd.validate,
        bytes_per_run=_vec_bytes(),
        working_set_bytes=5.0 * 4 * vector_simd.ELEMENTS,
        arithmetic_intensity=(2.0 * vector_simd.ELEMENTS * vector_simd.LOOPS)
        / max(_vec_bytes(), 1.0),
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="compression",
        name="Compression Round Trip (zlib + bz2, mixed corpus)",
        category="compression",
        workload_profile="compression",
        raw_metric_name="MB/s",
        setup_fn=compression.setup,
        run_fn=compression.run,
        validate_fn=compression.validate,
        bytes_per_run=4.0 * compression.SIZE_MB * _MB,
        working_set_bytes=compression.SIZE_MB * _MB,
        arithmetic_intensity=0.0,
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="hashing",
        name="Cryptographic Hashing (SHA-256 + BLAKE2b)",
        category="hashing",
        workload_profile="crypto",
        raw_metric_name="MB/s",
        setup_fn=hashing.setup,
        run_fn=hashing.run,
        validate_fn=hashing.validate,
        bytes_per_run=2.0 * hashing.SIZE_MB * _MB * hashing.PASSES,
        working_set_bytes=hashing.SIZE_MB * _MB,
        arithmetic_intensity=0.0,
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="branch_heavy",
        name="Irregular Control Flow & Random Access",
        category="branch_heavy",
        workload_profile="branch_bound",
        raw_metric_name="Mops/sec",
        setup_fn=branch_heavy.setup,
        run_fn=branch_heavy.run,
        validate_fn=branch_heavy.validate,
        bytes_per_run=8.0 * (branch_heavy.ELEMENTS + branch_heavy.QUERIES * 2),
        working_set_bytes=8.0 * (branch_heavy.ELEMENTS + branch_heavy.TABLE_SIZE * 2),
        arithmetic_intensity=0.0,
        multi_core_capable=True,
    ),
    WorkloadSpec(
        key="interpreter",
        name="CPython Interpreter Loop (runtime-bound, excluded from index)",
        category="interpreter",
        workload_profile="runtime_bound",
        raw_metric_name="Mops/sec",
        setup_fn=interpreter.setup,
        run_fn=interpreter.run,
        validate_fn=interpreter.validate,
        bytes_per_run=0.0,
        working_set_bytes=float(interpreter.SIEVE_LIMIT),
        arithmetic_intensity=0.0,
        multi_core_capable=True,
        measures_runtime=True,
    ),
]

BY_KEY: Dict[str, WorkloadSpec] = {w.key: w for w in WORKLOADS}

# Workloads used for the composite index. Runtime-bound workloads are reported
# but excluded, because their score reflects the Python build, not the CPU.
INDEX_WORKLOADS: List[WorkloadSpec] = [w for w in WORKLOADS if w.counted_in_index]

CATEGORIES: List[str] = [w.category for w in WORKLOADS]


def get(key: str) -> WorkloadSpec:
    if key not in BY_KEY:
        raise KeyError(f"Unknown workload key: {key!r}. Known: {sorted(BY_KEY)}")
    return BY_KEY[key]
