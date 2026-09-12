# BenchMind — Project Context

> **Read this file before modifying the BenchMind project.**
> It is the source of truth for purpose, architecture, current state, design
> decisions and development rules. When you make a significant architectural
> decision, update this file in the same commit.
>
> Companion documents:
> - `docs/ARCHITECTURE.md` — deeper technical structure
> - `docs/BENCHMARK_SPEC.md` — methodology, baselines, scoring (**the contract**)
> - `CHANGELOG.md` — what changed and when

Version 2.4.0

---

## 1. Project overview

BenchMind is a hardware benchmarking and diagnostics suite. It:

- benchmarks CPU performance across eight workload categories
- benchmarks GPU performance across four OpenCL workloads
- monitors system telemetry during benchmarks
- detects thermal and power throttling
- classifies the machine as compute-limited or bandwidth-limited
- measures thread scaling efficiency
- tracks run history locally and detects regressions
- produces scores with confidence intervals
- serves a cross-platform web dashboard

BenchMind is **not** a collection of CPU loops and GPU stress tests. The
long-term goal is a modular platform whose workloads represent distinguishable
kinds of real computational behaviour.

---

## 2. Philosophy

A good benchmark answers **"what kind of work is this hardware good at?"**, not
"how many loops can it complete?".

Four rules follow, in priority order:

1. **Correctness over score.** A workload that fails validation scores zero.
2. **Repeatability over magnitude.** A number without a spread is not a
   measurement. Every score carries a 95% confidence interval, and BenchMind
   refuses to call two overlapping scores different.
3. **Honesty over polish.** Where a workload measures something other than what
   its name suggests, the docstring and the spec say so. The `interpreter`
   category exists specifically to make that visible.
4. **No arbitrary constants.** Baselines come from a documented reference
   machine. Never adjust one to make a score look better.

---

## 3. Technology stack

| Area | Choice |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| CPU | NumPy, `concurrent.futures`, threadpoolctl |
| GPU | PyOpenCL (vendor-neutral, deliberately not CUDA) |
| Monitoring | psutil; LibreHardwareMonitor on Windows, psutil sensors on Linux |
| Storage | SQLite (local, primary); Supabase (optional, cloud) |
| UI | Single-file web dashboard served by FastAPI |
| Tests | `unittest` |

**PyQt was dropped in 2.0.** The desktop UI was one working screen plus two
empty files; the web dashboard is cross-platform, better looking, and reuses
the API and WebSocket that already existed.

---

## 4. Project structure

```
BenchMind/
├── run.py                     CLI entry point
├── api/
│   ├── main.py                FastAPI app: lifespan, job endpoints, WebSocket
│   └── jobs.py                background job manager
├── benchmarks/
│   ├── cpu_test.py            backwards-compatible CPU entry point
│   ├── gpu_test.py            backwards-compatible GPU entry point
│   ├── cpu/
│   │   ├── common.py          measurement core, statistics, scoring
│   │   ├── registry.py        single source of truth for workloads
│   │   ├── payloads.py        deterministic mixed-entropy corpora
│   │   ├── integer.py … interpreter.py   the eight workloads
│   │   │   ├── memory_hierarchy.py  cache-cliff sweep (diagnostic, unscored)
│   ├── _worker.py         spawn-safe worker with context caching
│   │   ├── single_core.py     BLAS isolation, core pinning
│   │   ├── multi_core.py      warm pool, scaling curve
│   │   └── cpu_suite.py       orchestrator
│   └── gpu/
│       ├── devices.py         GPU-only OpenCL discovery
│       ├── kernels.py         verifiable OpenCL sources
│       └── gpu_suite.py       event-profiled multi-workload suite
├── monitoring/
│   ├── telemetry_service.py   centralized sampler
│   ├── temp_reader.py         temperature source + circuit breaker
│   ├── system_info.py         cross-platform identification
│   ├── environment.py         fingerprinting
│   ├── topology.py            P/E cores, SMT siblings, core selection
│   └── validity.py            pre-run gate
├── ai/
│   ├── stability_engine.py    repeatability and sustained performance
│   ├── analysis.py            throttle detection, roofline
│   ├── summary_engine.py      deterministic narrative
│   └── llm_report.py          optional model-written analysis
├── storage/history.py         SQLite history and regression detection
├── db/                        optional Supabase sync (lazy, never fatal)
├── web/index.html             dashboard
├── scripts/                   calibration, repeatability, tool fetch
├── tests/                     81 tests
└── docs/
```

---

## 5. Architecture

```
                 CLI (run.py)        Web dashboard
                      │                    │
                      └────────┬───────────┘
                               ▼
                        FastAPI  +  JobManager
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        validity gate    CPU / GPU suites   TelemetryService
              │                │                │
              └────────────────┼────────────────┘
                               ▼
              throttle · roofline · scaling · stability
                               ▼
                      summary  (+ optional LLM)
                               ▼
                    SQLite history → regression
```

Telemetry is centralized in `TelemetryService`. **Do not create independent
sampling loops.**

---

## 6. Non-negotiable rules

These encode bugs that were found and fixed; reintroducing any of them
reintroduces the bug.

1. **Setup is never inside the timed region.** Split `setup(scale)` from
   `run(ctx)`.
2. **`perf_counter` for performance, `monotonic` for telemetry.** Never mix.
3. **Scale time, not working set.** `scale` changes pass counts, never buffer
   sizes.
4. **Warm the process pool before timing.** Worker startup and NumPy import
   must be paid before `perf_counter` starts.
5. **Single-core and multi-core use the same registry entries at the same
   scale.** Otherwise the blended index is meaningless.
6. **Validation must be able to fail.** Check against an independent
   reference, not just for NaN.
7. **BLAS limited to one thread during single-core.** `threadpoolctl`, always.
7b. **Never pin a benchmark to logical core 0.** It fields interrupts on
    Windows and was responsible for a 37% spread on the shortest workload.
    Core selection goes through `monitoring/topology.py`.
7c. **Never adopt a baseline from noisy calibration data.** The spread gate in
    `scripts/calibrate_baselines.py` enforces this; do not routinely `--force`.
7d. **Never size a cache-resident working set at the cache capacity.** Target
    half of it. Straddling the boundary caused a 15% spread on `integer`.
7e. **Any spread computed across stored runs must filter to valid runs only.**
    The fingerprint excludes power state by design, so battery and mains runs
    share a fingerprint and only the validity verdict separates them.
7f. **Disable index bounds checking on every gather.** `np.take` defaults to
    `mode='raise'`, which costs ~6.3 ns per lookup against ~0.9 ns for the
    gather. BenchMind's indices are always in range by construction. Leaving
    it on compressed the memory sweep's cliff ratio from 5.7x to 2.2x and made
    `branch_heavy` measure validation rather than the prefetcher.
7g. **Watch for implicit allocation in the timed region.** `np.searchsorted(...)`
    and `arr[idx]` allocate their results; only `out=` parameters avoid it.
    Reductions and verification belong in `validate`, not `run`.
8. **Never benchmark a CPU OpenCL runtime as a GPU.** Filter on
   `CL_DEVICE_TYPE_GPU`.
9. **GPU timing uses OpenCL event profiling**, never host wall clock.
10. **No blocking I/O in the telemetry sampler.** Sensor polling lives on its
    own thread behind a circuit breaker.
10b. **Never use `psutil.cpu_freq()` for a live clock on Windows.** It returns
    the registry's nominal base clock, a constant. Clocks come from
    LibreHardwareMonitor. A flat series must be reported as absent, never as
    steady.
10c. **A throttle verdict requires a measured drop in clock or power.**
    Temperature alone is corroborating evidence, never sufficient.
10d. **Warm a GPU by duration, not by one launch.** It needs 100 ms+ to reach
    a steady boost clock, and a short kernel measured during the ramp gives
    both a noisy spread and a median that depends on how warm it started.
10e. **A dependent FMA chain must use a multiplier below 1.** Above 1 it grows
    as y^n and overflows once the loop count is tuned up. This defect has now
    appeared twice, in the CPU and GPU kernels independently.
10f. **EVERY timed thing needs duration-based warmup and a work size that
    clears the 20 ms floor.** CPU repetitions, GPU launches and the memory
    sweep's passes have each had this bug independently. When adding any new
    measurement, size the work from a probe and warm up by elapsed time --
    never by a fixed count.
10g. **Prefer the fastest half to the median for pinned measurements.**
    Interference is one-directional; a central estimator is biased downward.
11. **Never compare across fingerprints, modes or baseline versions.**
12. **Importing anything must never require credentials.**
13. **Never assert on timings in CI** — and that includes tests that only
    look structural. Two hierarchy-sweep tests broke this rule in 2.3.0 and
    flaked on battery power. Extract the sizing logic as a pure function and
    test that; assert on structure and on physical impossibilities, never on
    how fast the machine is.

---

## 7. Current status

**CPU** — complete and calibrated. Eight workloads, setup/run split,
confidence intervals, topology-aware core pinning, warm pool, shared registry,
aligned single/multi work, thread scaling curve. Reference R2 measured with
every category under 1.7% spread.

**GPU** — complete and calibrated. Reference G1 (RTX 3050 6GB Laptop) measured
over five passes with a worst spread of 1.00%, and the FP32:FP64 ratio lands
at 1/59 against Ampere's architectural 1/64. The matrix kernel understates the
hardware and is documented as such rather than quietly improved.

**Telemetry** — decoupled temperature polling, frequency and per-core logging,
sampling-health reporting.

**Analysis** — throttle detection (clock and power based), roofline, scaling,
stability, drift-versus-scatter, regression.

**API and UI** — job-based execution, SSE progress, WebSocket telemetry,
single-file dashboard.

**Storage** — SQLite history with fingerprint-aware regression detection.
Supabase optional.

**Tests** — 140, all passing, none timing-dependent. Verified by running the suite three times consecutively after the 2.3.1 test rewrite.

---

## 8. Current priority

1. Recalibrate `branch_heavy` (`python -m scripts.calibrate_baselines --reps 5`).
   Its stored baseline predates the 2.4.0 gather-mode fix. Other categories
   are unaffected.
2. Run `python run.py repeat --runs 5` and confirm spread is under 2%. Expect
   downward drift on a laptop; raise `--cooldown` if it appears.
3. **Done.** Reference G1 calibrated; worst spread 1.00%.
4. Freeze the CPU implementation; bump `BASELINE_VERSION` on any change after
   that point.
5. RAM latency and bandwidth benchmark.
6. Storage benchmark.
7. Native kernels (cffi or Numba) to reduce dependence on the Python stack.
8. macOS temperature source.
9. Storage benchmark. (Memory is covered by the hierarchy sweep.)

Do not jump ahead unless explicitly asked.

---

## 9. Before making a change

Ask:

- Why is this change needed?
- Does the existing architecture already solve it?
- Does it affect benchmark validity, scoring or telemetry?
- Does it break comparability with stored results? If yes, bump
  `BASELINE_VERSION` and say so in the CHANGELOG.
- Are tests needed? How will repeatability be verified?

Verifying a benchmark change means all of:

1. `python -m unittest discover -s tests`
2. `python run.py bench --mode standard`
3. `python run.py repeat --runs 3`
4. score spread inspected
5. telemetry and temperatures inspected

Code running is not evidence that a benchmark change is correct.

---

## 10. Instructions for AI agents

You are working on an existing project. Inspect before changing.

1. Read this file and `docs/BENCHMARK_SPEC.md` first.
2. Inspect the relevant files before proposing anything.
3. Explain the approach briefly, then implement the smallest coherent change.
4. Run the tests.
5. Report what changed and what remains broken.

Do not replace working implementations with simplified examples. Do not rewrite
from scratch. **If you find a benchmark methodology problem, flag it rather
than silently working around it** — that is how the 1.x defects survived as
long as they did.

This project values benchmark validity and engineering honesty over speed of
implementation.
