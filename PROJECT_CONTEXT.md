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

Version 2.0.0

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
│   │   ├── _worker.py         spawn-safe worker with context caching
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
8. **Never benchmark a CPU OpenCL runtime as a GPU.** Filter on
   `CL_DEVICE_TYPE_GPU`.
9. **GPU timing uses OpenCL event profiling**, never host wall clock.
10. **No blocking I/O in the telemetry sampler.** Temperature polling lives on
    its own thread behind a circuit breaker.
11. **Never compare across fingerprints, modes or baseline versions.**
12. **Importing anything must never require credentials.**
13. **Never assert on timings in CI.** CI runners are noisy VMs; timing
    assertions produce flaky failures that teach people to ignore the suite.

---

## 7. Current status

**CPU** — complete. Eight workloads, setup/run split, confidence intervals,
core pinning, warm pool, shared registry, aligned single/multi work, thread
scaling curve.

**GPU** — rewritten and structurally sound, but **the baselines are untested
placeholders**. Needs a run on real hardware and recalibration.

**Telemetry** — decoupled temperature polling, frequency and per-core logging,
sampling-health reporting.

**Analysis** — throttle detection, roofline, scaling, stability, regression.

**API and UI** — job-based execution, SSE progress, WebSocket telemetry,
single-file dashboard.

**Storage** — SQLite history with fingerprint-aware regression detection.
Supabase optional.

**Tests** — 81, all passing, none timing-dependent.

---

## 8. Current priority

1. Run `python -m scripts.calibrate_baselines --reps 5` on a real physical
   machine and replace reference R1.
2. Run `python run.py repeat --runs 5` and confirm spread is under 2%.
3. Run the GPU suite on the RTX 3050 and calibrate the GPU baselines.
4. Freeze the CPU implementation; bump `BASELINE_VERSION` on any change after
   that point.
5. RAM latency and bandwidth benchmark.
6. Storage benchmark.
7. Native kernels (cffi or Numba) to reduce dependence on the Python stack.
8. macOS temperature source.

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
