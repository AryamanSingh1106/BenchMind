# BenchMind Benchmark Specification

Version 2.1.2 · baseline set `2.1.0`

This document is the contract for what a BenchMind score means. If you change
a workload, a baseline, or a scoring rule, change this file in the same commit
and add a CHANGELOG entry. A score whose methodology has drifted from its
specification is not a measurement.

---

## 1. The honest framing

**BenchMind measures your machine as a Python compute host, not as bare
silicon.**

This is not a disclaimer bolted on at the end; it determines the whole design.

| Workload | What it really exercises |
|---|---|
| matrix | your BLAS build (OpenBLAS vs MKL vs reference) |
| hashing | your OpenSSL build, including SHA hardware extensions |
| compression | your zlib and bz2 builds |
| interpreter | CPython's dispatch loop |
| integer, floating_point, vector_simd, branch_heavy | NumPy's compiled kernels over your CPU |

Upgrade NumPy and the matrix score moves with no hardware change. That does not
make the benchmark worthless — it makes it a benchmark of a *stack*. It does
mean two scores are only comparable when the stacks match.

Therefore every result carries an **environment fingerprint**
(`monitoring/environment.py`), and the history and comparison layers refuse to
compare runs whose `fingerprint_hash` differs.

The fingerprint hashes: Python version and implementation, NumPy version, BLAS
backend and version, OpenSSL, zlib, OS, CPU architecture, any thread-count
environment overrides, and **`BASELINE_VERSION`**.

It deliberately excludes battery percentage, executable path, OS patch level,
and **the BenchMind application version**. That last exclusion is deliberate
and was a bug until 2.1.1: hashing the application version meant every
release, including pure bug-fix releases that changed no workload, produced a
new fingerprint and silently orphaned all stored history. What determines
comparability is the baseline set, not the build.

---

## 2. Measurement rules

These are enforced in `benchmarks/cpu/common.py` and must not be relaxed.

### 2.1 Setup is never timed
Every workload is `setup(scale) -> ctx` plus `run(ctx) -> (output, work_units)`.
Allocation, RNG generation and payload construction happen in `setup`. Only
`run` sits inside the `perf_counter` window.

### 2.2 Clock separation
- `time.perf_counter()` — all performance measurement, without exception.
- `time.monotonic()` — telemetry timestamps and telemetry window filtering.
- `time.time()` — display only.

### 2.3 Scale time, not working set
`scale` (varied by quick/standard/full) changes how many passes a workload
makes. It never changes buffer sizes. Shrinking a buffer moves the workload
into a different cache level, so a quick-mode score would measure something
genuinely different while wearing the same label.

### 2.4 Every measurement reports a confidence interval
Minimum 5 repetitions. The 95% CI is computed over per-repetition throughput
using a t-distribution. `scores_are_distinguishable()` returns False when two
intervals overlap, and BenchMind will not claim one machine is faster than
another in that case.

### 2.5 Validation happens outside the timed region, and must be able to fail
A validator that only checks for NaN passes a completely wrong answer.
Each workload verifies against an independently computed reference:

| Workload | Verification |
|---|---|
| integer | exact int64 checksum recomputed from the same seeds |
| floating_point | FP32 and FP64 sums must agree within 1%, and land inside the analytic fixed-point range |
| matrix | `C[0,0]` must equal `dot(A[0,:], B[:,0])` to 1e-9 relative |
| vector_simd | result mean must land inside the analytic bounds |
| compression | SHA-256 round trip on both codecs, plus a compression-ratio sanity bound |
| hashing | digests compared against values computed in `setup` |
| branch_heavy | ordering, exact int64 element sum, every search result a valid insertion point, gather sum |
| interpreter | π(300000) = 25997 |

The test suite (`tests/test_workloads.py`) corrupts each output and asserts the
validator rejects it.

### 2.6 Single-core and multi-core run identical work
Both pull from `benchmarks/cpu/registry.py` at the same scale. Thread count is
the only variable. This is what makes the weighted CPU Index meaningful.

### 2.7 Multi-core timing excludes process startup
The process pool is created and pinged until every worker reports a distinct
PID, and each worker builds and caches its workload context, all before the
`perf_counter` starts.

### 2.8 Single-core isolation
1. BLAS limited to one thread via `threadpoolctl`.
2. Process pinned to one logical core via `cpu_affinity` where the platform
   supports it. The result records whether pinning actually succeeded.
3. Best-effort priority elevation.

---

## 3. Workloads

| Key | Profile | Metric | Working set | Notes |
|---|---|---|---|---|
| `integer` | compute_bound | Mops/sec | 1.00 MB (4 × int64 × 32,768) | 6 int64 ops per element per pass, in place |
| `floating_point` | compute_bound | MFLOPS | 1.13 MB | L2-resident FMA, FP32 + FP64 |
| `matrix` | mixed | GFLOPS | 3 × 4.5 MB f64 | 768×768 dgemm through BLAS |
| `vector_simd` | memory_bound | GFLOPS | 5 × 32 MB f32 | DRAM-resident streaming FMA |
| `compression` | compression | MB/s | 8 MB corpus | zlib level 6 + bz2 level 5, round trip |
| `hashing` | crypto | MB/s | 16 MB corpus | SHA-256 + BLAKE2b, 4 passes |
| `branch_heavy` | branch_bound | Mops/sec | 32 MB table | sort + binary search + random gather |
| `interpreter` | runtime_bound | Mops/sec | — | **excluded from the index** |

### 3.1 On `vector_simd` versus `floating_point`
These run the same kernel shape at two different working-set sizes. The ratio
between them is the machine's cache cliff and drives the roofline verdict in
`ai/analysis.py`. Neither is "the FP score" on its own.

### 3.2 On `branch_heavy`
The 1.x version claimed to exercise the branch predictor, but
`np.sum(arr > pivot)` compiles to branchless SIMD. The honest description of
the 2.0 workload is **irregular control flow and random access**: a comparison
sort, a binary search into a table far larger than cache, and a permuted
gather that defeats the prefetcher. The category key stays `branch_heavy` for
compatibility with stored results.

### 3.3 On the compression corpus
The 1.x payload was a repeated 1 KB pattern that deflate compressed several
hundred to one — that measured the match finder skipping long runs. The 2.0
corpus is deterministic mixed entropy: 40% English-like text, 25% structured
records, 20% binary floats, 15% incompressible random.

### 3.4 On `interpreter`
Deliberately kept, deliberately excluded from the index. It is the old 1.x
integer workload, and its score moves when you upgrade Python without touching
hardware. That makes it useless as a hardware metric and genuinely useful as a
runtime metric, so it is reported with `measures_runtime=True`.

---

## 4. Scoring

```
subtest score      = (measured raw metric / category baseline) × 1000
category composite = geometric mean of that category's passing subtests
single-core score  = geometric mean of all index-counted single-thread subtests
multi-core score   = geometric mean of all index-counted multi-thread subtests
CPU Index          = 0.4 × single-core + 0.6 × multi-core
```

Geometric mean, not arithmetic: one workload producing a very large normalized
score must not dominate the composite.

Uncertainty propagates. For a geometric mean of *n* subtests the relative
uncertainty is `sqrt(Σ rel_i²) / n`; for the weighted index it is the
quadrature sum of the weighted contributions.

A subtest that fails validation scores 0 and is excluded from its composite. An
unregistered category scores 0 rather than falling back to a default baseline.

### 4.1 Reference machine R2

A score of 1000 means "matched R2 on that workload". R2's measured medians are
the baselines in `CATEGORY_BASELINES`.

```
Machine    Lenovo laptop, 13th Gen Intel Core i5-13450HX
           6 P-cores (SMT) + 4 E-cores, 16 threads, 15.71 GB
OS         Windows 11
Python     3.11.0 (CPython)
NumPy      2.4.6
BLAS       OpenBLAS 0.3.31.188.0, limited to 1 thread
Pinning    logical core 10 (last P-core, away from core 0)
Conditions validity gate 'valid', mains power, idle, 5 passes
```

| Category | R2 median | Unit | Spread |
|---|---|---|---|
| integer | 2106.52 | Mops/sec | 0.74% |
| floating_point | 5137.51 | MFLOPS | 1.68% |
| matrix | 46.88 | GFLOPS | 0.16% |
| vector_simd | 1.73 | GFLOPS | 0.45% |
| compression | 24.84 | MB/s | 0.14% |
| hashing | 849.14 | MB/s | 0.28% |
| branch_heavy | 32.30 | Mops/sec | 0.74% |
| interpreter | 45.06 | Mops/sec | 0.49% |

**Why R2 replaced R1.** R1 was a shared cloud instance: noisy neighbours,
unknown turbo behaviour, and no way to control its conditions. R2 is physical
hardware whose state can be verified before a calibration pass.

**The spreads are the point.** Every category is under 1.7% and five are under
0.5%, which is what makes these baselines usable. The same machine's first
calibration attempt, before the 2.0.2 and 2.1.0 measurement fixes, produced
36.92% on `integer` — and adopting that would have permanently miscalibrated
the category while looking perfectly authoritative. The journey from 36.92% to
0.74% is recorded in the CHANGELOG for 2.0.2 and 2.1.0.

R2 is a laptop, which has one genuine drawback as a reference: it is thermally
constrained, and repeated back-to-back runs drift downward as the chassis
soaks up heat (see the drift detection in section 7). Calibration must be run
from cold.

### 4.2 The calibration spread gate

`calibrate_baselines.py` **refuses** to emit a baseline block when any
category's spread across passes exceeds `--max-spread` (default 5%).

A baseline captured at 37% spread is one sample from a very wide distribution,
and every future score in that category is then measured against a number that
could just as easily have been 30% different. That is worse than having no
baseline, because it looks authoritative.

If the gate trips, in order of likelihood: background processes, battery or a
balanced power plan, a machine that started warm, too few passes, or a bad
pinned core. `--force` exists but should be used only when you can explain why
the spread is irreducible on that machine.

Never adjust a single baseline to make a score look better. That is the exact
failure mode the project philosophy warns about.

---

## 5. Modes

| Mode | Scale | Reps | Use |
|---|---|---|---|
| `quick` | 0.5 | 3 | smoke test; wide intervals, do not publish |
| `standard` | 1.0 | 5 | **the only mode whose scores should be stored or compared** |
| `full` | 1.0 | 9 | tightest intervals |

Mode is recorded in every result and the history layer refuses to compare
across modes.

---

## 6. Run validity

`monitoring/validity.py` samples the machine before the benchmark and returns
`valid`, `tainted` or `invalid`.

| Condition | Threshold | Severity |
|---|---|---|
| background CPU load | ≥ 40% | error → invalid |
| background CPU load | ≥ 15% | warning → tainted |
| on battery | any | warning → tainted |
| free RAM | < 15% | warning → tainted |
| CPU temperature at start | > 75 °C | warning → tainted |

Only `valid` runs are used as comparison baselines or uploaded to the shared
store. The battery check matters most: most laptops cap package power when
unplugged, and the score can halve.

---

## 7. Derived analyses

**Throttle detection** splits the telemetry window into quartiles and compares
opening against closing clock speed and package power, correlated with the
temperature rise.

A verdict **requires a measured drop**: clock ≥ 7% or package power ≥ 12%.
Temperature never produces a verdict on its own, only a note. Confidence is
`high` when both clock and power fell, or when a clock drop coincides with a
peak above 85 °C; `medium` otherwise; `unmeasured` when no usable signal
existed.

Two rules exist because of a specific false positive on a mobile i5-13450HX:

* **A clock series with no variance is treated as absent.** `psutil.cpu_freq()`
  on Windows returns the registry's nominal base clock, a constant. Reading
  that as "the clock held steady" is how 2.0.0 produced the self-contradicting
  verdict "throttling detected ... sustained 100% of opening clock speed".
  Clocks come from LibreHardwareMonitor's MSR readings instead.
* **The hot threshold is 95 °C, not 85.** Mobile H- and HX-class parts sustain
  high 80s under all-core load by design.

P-core and E-core clocks are collected separately and the headline figure is
the performance-core mean, because averaging across core types would make a
shift in the work split look like throttling.

**Roofline / bottleneck** compares cache-resident against DRAM-resident FP
throughput at one thread. Ratio ≥ 4 → memory-bandwidth limited; 2–4 → mixed;
< 2 → compute limited. Estimated memory bandwidth is derived from
`vector_simd` throughput divided by its declared arithmetic intensity.

**Stability** is a weighted blend: repeatability across runs 50%, sustained
performance 35%, utilization steadiness 15%. The 1.x formula
(`100 − std(cpu_utilization)`) is retained only as that last 15%, because a
multi-phase benchmark is *supposed* to vary its CPU load.

**Drift versus scatter.** Standard deviation cannot distinguish five scores
falling steadily from five scores bouncing around a mean, but they mean
different things: scatter says the measurement is imprecise, drift says the
machine changed while you measured it. `detect_drift()` uses least-squares
slope for magnitude and Spearman rank correlation against run order for
direction, and calls a trend only when the total change ≥ 3% **and**
|rho| ≥ 0.6. A drifting session is penalised in the repeatability score
regardless of how tight its standard deviation is.

Downward drift usually means heat is not clearing between runs; raise
`--cooldown`. Upward drift usually means the first run paid a warmup cost.

**Regression detection** compares against the median of the last 5 valid runs
on the same fingerprint and mode. A change is only called significant when it
exceeds both 5% and the combined confidence intervals.

---

## 8. GPU

GPU workloads are timed with OpenCL **event profiling** (device-side
nanosecond counters), never host wall clock. Only devices of type
`CL_DEVICE_TYPE_GPU` are benchmarked; CPU OpenCL runtimes are excluded so a CPU
can never be reported as a graphics device. Every kernel's output is read back
and checked against a CPU reference.

| Workload | Metric | R1-GPU baseline |
|---|---|---|
| fp32_compute | GFLOPS | 4000.0 |
| fp64_compute | GFLOPS | 120.0 |
| memory_bandwidth | GB/s | 180.0 |
| matrix | GFLOPS | 900.0 |
| transfer | GB/s | informational, not scored |

The FMA kernels use a dependent chain specifically so the compiler cannot hoist
the loop out; an independent chain would be optimized away and the benchmark
would measure the optimizer.

**The GPU baselines have not been measured on real hardware.** They are
placeholders. Run the suite on your RTX 3050 and recalibrate before treating
GPU scores as meaningful.

---

## 9. What is still not solved

Recorded honestly rather than hidden:

1. **The stack problem remains.** Fingerprinting makes scores comparable within
   a stack; it does not make them comparable across stacks. Truly
   hardware-level measurement needs native kernels via cffi, Cython or a pinned
   Numba.
2. **NumPy ufunc call overhead** is roughly 5–10% of the L2-resident workloads.
   It is inside the measurement.
3. **`compression` and `branch_heavy` multi-core results are load-balance
   limited.** Measuring throughput over a fixed time window instead of
   time-to-complete-fixed-work would remove the straggler effect, but changes
   what the metric means and would invalidate the baselines.
4. **The SMT sibling of the pinned core cannot be reserved.** Other processes
   may be scheduled onto it, sharing execution resources with the measurement.
   Reported, not solved.
5. **No RAM latency, storage, or network benchmark** yet.
6. **The GPU baselines are still untested placeholders.** They have never been measured on real hardware; run the suite on a real GPU and recalibrate before treating GPU scores as meaningful.
