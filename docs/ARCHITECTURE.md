# BenchMind Architecture

Companion to `PROJECT_CONTEXT.md`. This file explains *how* the pieces fit;
`docs/BENCHMARK_SPEC.md` explains *what the numbers mean*.

---

## Layers

```
┌──────────────────────────────────────────────────────────┐
│  Presentation    run.py (CLI)   ·   web/index.html        │
├──────────────────────────────────────────────────────────┤
│  Transport       api/main.py (FastAPI)  ·  api/jobs.py    │
├──────────────────────────────────────────────────────────┤
│  Orchestration   cpu_suite.py  ·  gpu_suite.py            │
├──────────────────────────────────────────────────────────┤
│  Measurement     common.py · registry.py · workloads      │
│                  single_core.py · multi_core.py           │
├──────────────────────────────────────────────────────────┤
│  Observation     telemetry_service · temp_reader          │
│                  environment · validity · system_info     │
├──────────────────────────────────────────────────────────┤
│  Interpretation  ai/analysis · stability · summary · llm  │
├──────────────────────────────────────────────────────────┤
│  Persistence     storage/history (SQLite)  ·  db/ (cloud) │
└──────────────────────────────────────────────────────────┘
```

Dependencies point downward only. The measurement layer knows nothing about
FastAPI, and the workloads know nothing about scoring — they return raw work
units and let `common.py` normalize.

---

## The workload contract

Every workload module exposes exactly three functions:

```python
def setup(scale: float) -> ctx      # allocate, seed, build payloads.  UNTIMED
def run(ctx) -> (output, work_units)  # the kernel.                    TIMED
def validate(output) -> bool        # verify against a reference.      UNTIMED
```

`work_units` is in the numerator of the metric (Mops, MFLOP, MB, GFLOP);
`common.py` divides by elapsed seconds to get the raw metric.

`run` must be idempotent against the same `ctx`: calling it twice must do
identical work and produce identical output. Accumulators are reset at the top
of `run`, not carried between repetitions. `tests/test_workloads.py` enforces
this. Without it, repetition 2 does different work from repetition 1 and the
standard deviation is meaningless.

`registry.py` wraps each module in a `WorkloadSpec` carrying display name,
category, profile, metric name, declared bytes moved and arithmetic intensity.
It is the only place workload parameters live, which is what guarantees
single-core and multi-core measure the same thing.

---

## Timing pipeline

```
run_timed_subtest(spec, scale, target_duration, min_reps, max_reps)

  ctx = setup(scale)                         ── untimed, reported separately
  run(ctx)                                   ── untimed warmup
  repeat 5..15 times:
      t0 = perf_counter()
      output, units = run(ctx)               ── the only timed code
      t1 = perf_counter()
      stop once min_reps done and target_duration elapsed
  validate(output)                           ── untimed
  aggregate: median, stdev, 95% CI on per-rep throughput
  score = (raw_metric / baseline) × 1000, or 0 if validation failed
```

---

## Multi-core path

```
WarmPool(n) ────────────────────────────────────────────┐
  create ProcessPoolExecutor(initializer=init_worker)   │  all before
  submit warmup() × 4n, collect distinct PIDs           │  perf_counter
  prepare_workload(): each worker builds and caches ctx │  starts
────────────────────────────────────────────────────────┘
repeat `reps` times:
    t0 = perf_counter()
    submit execute(workload, scale) × (workers × 2)
    gather work_units and validity from every chunk
    t1 = perf_counter()
```

Workers cache contexts in a module-level dict keyed by `(workload, scale)`, so
the second and subsequent chunks reuse the arrays that the first one built.

The pool is created once per suite and reused across all workloads.

---

## Telemetry

Two threads, deliberately.

```
sampler thread (5 Hz)                 temperature thread (1 Hz)
  psutil.cpu_percent()                  get_temperatures()
  psutil.virtual_memory()                 ├── LibreHardwareMonitor HTTP
  psutil.cpu_freq()                       └── psutil.sensors_temperatures()
  read shared temperature slot  ◄─────── write shared slot
  append TelemetrySnapshot
```

The split exists because temperature comes from an external process over HTTP.
With both on one thread, a missing sensor source turned the 0.2 s sampling
interval into an irregular multi-second one without anything visibly failing.
A circuit breaker in `temp_reader.py` stops probing entirely after three
consecutive failures and retries after 30 s.

`sampling_health()` reports the actual interval distribution, so a distorted
timeline is visible rather than silent.

Windows uses `perf_counter` at ~100 ns resolution; the telemetry window is
filtered on `monotonic` and never used for scoring.

---

## Job execution

```
POST /api/benchmark
      │
      ├─ JobManager.submit() ── returns job id in ~3 ms
      │        │
      │        └─ worker thread ── _execute_benchmark(req, progress)
      │                                validity → warmup → CPU → GPU →
      │                                scaling → analysis → history
      │
GET  /api/jobs/{id}            poll
GET  /api/jobs/{id}/events     SSE, closes when the job ends
WS   /ws/live-monitor          telemetry, independent of any job
```

One benchmark at a time, enforced by the manager: two concurrent runs would
contend for the same cores and invalidate both. A second request gets 409.

All job state changes happen under a single lock.

---

## Analysis inputs

| Analysis | Needs |
|---|---|
| throttle detection | `cpu_freq` and `cpu_temp` timelines with `elapsed` |
| roofline | single-thread subtests with declared `arithmetic_intensity` |
| scaling | the thread-count sweep from `run_scaling_curve` |
| stability | history scores, throttle report, cpu timeline |
| regression | SQLite history filtered by fingerprint and mode |

Each analysis degrades gracefully and says what was missing rather than
producing a confident verdict from absent data.

---

## Storage

**SQLite is primary.** Two tables: `runs` (indexed on
`fingerprint_hash, mode, created_at`) and `category_scores`. Raw telemetry
arrays are stripped before storage — a five-minute run at 5 Hz is 1,500 samples
across six series, and keeping every sample forever buys nothing that the
derived summary does not already provide.

`comparable_baseline()` returns the median of the last five **valid** runs on a
matching fingerprint and mode. A median across several runs is a far more
stable reference than the single previous run, which might itself have been a
bad sample.

**Supabase is optional and never fatal.** Every function returns `None` when
unconfigured. Only `valid` runs are uploaded.

---

## Extending

**Adding a CPU workload**

1. Create `benchmarks/cpu/<name>.py` with `setup`, `run`, `validate`.
2. Add a `WorkloadSpec` to `registry.WORKLOADS` with declared
   `bytes_per_run` and `arithmetic_intensity`.
3. Add a baseline to `CATEGORY_BASELINES` measured on the reference machine.
4. Add a corruption test in `tests/test_workloads.py` proving `validate`
   rejects a wrong answer.
5. Bump `BASELINE_VERSION`; update the spec and CHANGELOG.

Both single-core and multi-core pick it up automatically.

**Adding a GPU workload**

1. Add the kernel to `benchmarks/gpu/kernels.py` with a dependent chain so the
   compiler cannot eliminate it.
2. Add a `_bench_*` function using event profiling and CPU-reference
   verification.
3. Register it in `benchmark_device` and add a baseline to `GPU_BASELINES`.
