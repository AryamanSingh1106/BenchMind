# Changelog

## 2.0.0

A methodology overhaul. **Scores from 1.x are not comparable to 2.0 scores.**
Baselines, workloads and scoring all changed; `BASELINE_VERSION` is now
tracked in every result.

### Measurement correctness

- **Setup removed from the timed region.** Every workload split into
  `setup(scale)` and `run(ctx)`. Previously every workload generated its
  arrays and payloads inside the timed function — `floating_point` produced six
  million random numbers on each repetition and charged that to its MFLOPS
  figure.
- **Process pool startup removed from multi-core timing.** The pool is created,
  pinged until every worker reports a distinct PID, and given its workload
  context before `perf_counter` starts. On Windows spawn, worker startup plus
  NumPy import had been a large fraction of the measured window.
- **Single-core and multi-core now run identical work** from a shared registry.
  They previously used different sizes (integer 300k vs 200k, matrix 768 vs
  384) while still being blended 0.4/0.6 into one index.
- **Scale changes duration, not working-set size.** Quick mode used to shrink
  buffers, which moved memory-bound workloads into cache and silently changed
  what they measured.
- **Multi-core reports real statistics.** It previously hardcoded
  `std_dev=0.0` and `stability_pct=100.0` from a single measurement.
- **Minimum 5 repetitions**, up from 3.

### Statistics

- 95% confidence intervals on every subtest, composite and the index, with
  t-distribution critical values and correct propagation through geometric
  means and the weighted sum.
- `scores_are_distinguishable()` — BenchMind will not claim one machine is
  faster than another when the intervals overlap.

### Workloads

- `integer` rewritten as int64 SIMD ALU work. The old pure-Python version
  measured the CPython interpreter.
- New `interpreter` category preserves the old workload, flagged
  `measures_runtime` and excluded from the index.
- `floating_point` resized to be L2-resident and made convergent so FP32 and
  FP64 can be cross-checked.
- `vector_simd` relabelled `memory_bound`, which is what it always was.
- `branch_heavy` rewritten as sort plus binary search plus random gather, and
  renamed "Irregular Control Flow & Random Access". The old version's headline
  operation compiled to branchless SIMD.
- `compression` corpus changed from a repeated pattern to deterministic mixed
  entropy; MD5 calls removed from the timed region.
- `matrix` allocates once and uses `out=`.
- **Validators now verify against independent references** instead of checking
  for NaN. Tests corrupt every output and assert rejection. This caught a real
  bug: `hashing.validate({})` returned True because `None == None`.

### Scoring

- Baselines replaced with measured values from documented reference machine R1.
- Unregistered categories now score 0 instead of falling back to a default
  baseline of 100.

### GPU

- Timed with OpenCL event profiling instead of `time.time()`.
- Arbitrary `/100000` score constant replaced with GFLOPS and GB/s against
  documented baselines.
- Kernel output is read back and verified against a CPU reference; previously
  nothing was read back at all, so a no-op driver would have scored well.
- Only `CL_DEVICE_TYPE_GPU` devices are benchmarked. The old loop iterated every
  OpenCL device, so CPU runtimes were benchmarked and reported as GPUs.
- Four workloads (FP32, FP64, memory bandwidth, matrix) plus separately
  measured host transfer bandwidth.

### Telemetry

- Temperature polling moved to its own slower thread behind a circuit breaker.
  A 1 s blocking HTTP call inside the 5 Hz sampler had been silently destroying
  the sampling interval whenever LibreHardwareMonitor was absent.
- CPU frequency and per-core utilization recorded, which is what makes throttle
  detection possible.
- Sampling interval drift measured and reported.
- psutil sensor fallback on Linux; missing sources report `unavailable` rather
  than 0 °C.

### New capabilities

- Environment fingerprinting; comparisons refused across mismatched stacks.
- Pre-run validity gate.
- Thermal throttle detection.
- Roofline bottleneck classification.
- Thread scaling curve.
- Local SQLite history with regression detection.
- Optional LLM-written report, making the `ai/` folder name honest.
- Web dashboard, CLI, calibration and repeatability scripts.

### Stability

- `calculate_stability` no longer defines stability as
  `100 − std(cpu_utilization)`, which mostly measured BenchMind's own phase
  transitions. Replaced by a weighted blend of repeatability (50%), sustained
  performance (35%) and utilization steadiness (15%). The legacy entry point
  still works.

### API and infrastructure

- `POST /api/benchmark` returns a job id immediately instead of blocking for a
  minute; progress via polling or SSE.
- Lock-protected `JobManager`; concurrent benchmarks rejected with 409.
- Deprecated `@app.on_event` replaced with lifespan.
- WebSocket sends only new samples instead of re-sending the latest on a timer.
- `db/client.py` no longer raises at import time when credentials are missing.
- `requirements.txt` re-encoded as UTF-8; it had been UTF-16 from a PowerShell
  redirect and failed `pip install -r` on Linux and macOS.
- `system_info` works on Linux and macOS; previously Windows-only, and it never
  returned the `os` key that `db/crud.py` read.
- LibreHardwareMonitor binaries removed from the repository and fetched on
  demand, with the MPL-2.0 notice.
- Root-level scratch scripts moved under `scripts/`, where pytest will not
  collect and execute them.
- 81 tests. PyQt UI removed in favour of the web dashboard.

---

## 1.x

Initial implementation: multi-category CPU suite, basic OpenCL GPU test,
centralized telemetry service, FastAPI backend, PyQt screens.
