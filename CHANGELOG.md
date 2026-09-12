# Changelog

## 2.1.1

Baseline set unchanged (`2.1.0`); scores remain comparable to 2.1.0.

### Reference machine replaced: R1 -> R2

The 2.1.0 measurement fixes brought a real i5-13450HX laptop from a 36.92%
worst-case calibration spread down to 1.68%, with five of eight categories
under 0.5%. That machine now becomes reference R2, replacing the shared cloud
instance used for R1.

    integer         2106.52 Mops/sec   spread 0.74%
    floating_point  5137.51 MFLOPS     spread 1.68%
    matrix            46.88 GFLOPS     spread 0.16%
    vector_simd        1.73 GFLOPS     spread 0.45%
    compression       24.84 MB/s       spread 0.14%
    hashing          849.14 MB/s       spread 0.28%
    branch_heavy      32.30 Mops/sec   spread 0.74%
    interpreter       45.06 Mops/sec   spread 0.49%

R1 was a cloud instance with noisy neighbours and unverifiable conditions. R2
is physical hardware whose state can be checked before a pass. Full
specification in docs/BENCHMARK_SPEC.md section 4.1.

Note that `floating_point` also rose from 3,599 to 5,137 MFLOPS between 2.0.2
and 2.1.0 on the same machine. That is not measurement noise: the workload was
previously sized at 2.25 MB against a 2 MB L2, so it was partly DRAM-bound.
Fixing the size made it genuinely faster as well as far more stable.

### The fingerprint no longer hashes the application version

`compute_fingerprint_hash()` included `benchmind_version`, so **every release
produced a new fingerprint and silently orphaned all stored run history** —
even a pure bug-fix release that changed no workload. This was hit in practice
going 2.0.2 -> 2.1.0.

It now hashes `BASELINE_VERSION` instead. Two builds sharing a baseline set
measure the same thing and their scores are comparable; a baseline change
genuinely does invalidate comparisons, and that is now the only case where
history breaks.

### Test bug: BLAS threads were unrestricted

`test_real_workloads_have_useful_repetition_lengths` did not limit BLAS
threads, so `matrix` ran multithreaded — about 264 GFLOPS on a 16-thread
machine against 47 GFLOPS single-threaded. The repetition came in at 10 ms and
the test failed against a configuration the suite never uses. It now wraps in
`threadpool_limits(1, "blas")`, matching `single_core.py`, and also covers
`vector_simd`.

### Elsewhere

- The short-repetition warning now only fires at `scale >= 1.0`. A
  deliberately scaled-down run has short repetitions by construction, so
  warning about it was noise; the test logs were full of it.

## 2.1.0

**Baseline version bumped to `2.1.0`. Scores are not comparable to 2.0.x.**
Two workloads were resized, so their raw metrics mean something different.

Prompted by the 2.0.2 calibration on an i5-13450HX. Fixing core selection had
cleaned up five categories but left three noisy, and the pattern pointed
somewhere specific:

    matrix            0.22%      compression      0.16%     interpreter  0.16%
    vector_simd       1.06%      hashing          1.44%
    floating_point    7.45%  <-- branch_heavy     9.80%  <-- integer  15.57%  <--

### The two worst offenders were balanced on the L2 cache cliff

A Raptor Lake P-core has exactly 2 MB of L2. Measured working sets:

    integer          4 x int64 x 65,536       = 2.00 MB   exactly at the limit
    floating_point   (3 fp32 + 3 fp64) x 65k  = 2.25 MB   over the limit

Sitting precisely on the cliff is the worst possible place to be: any small
change in what else is resident pushes data in and out of L2 and the
measurement swings. It also meant the workload documented as "L2-resident" was
partly DRAM-bound.

- `integer` is now 32,768 elements (1.00 MB), loops raised 120 → 500.
- `floating_point` is now 32,768 elements (1.13 MB), loops raised 150 → 1200.

Loop counts went up rather than down deliberately, so repetition duration
increased. Halving the working set alone would have dropped repetitions to
~13 ms, trading one source of noise for another.

### branch_heavy was allocating 16 MB per repetition, inside the timed region

`np.searchsorted(table, keys)` and `gather_src[gather_idx]` each returned a
fresh 8 MB array on every repetition. The setup/run split was supposed to
eliminate exactly this, but these allocations were implicit in NumPy's return
values rather than visible `np.empty` calls, which is why they survived the
2.0.0 audit.

- Results now go into buffers allocated in `setup`; the gather uses
  `np.take(..., out=)`.
- The binary search is chunked at 64 K so its intermediate stays cache-resident
  and the allocator reuses one block instead of requesting 8 MB each time.
- The sortedness check and four array-wide reductions also ran inside `run`,
  charging verification to the measurement. `run` now returns buffers by
  reference and `validate` does the reductions after the clock stops.
- Validation got stronger as a side effect: it now checks ordering, the exact
  int64 element sum, that every search result is a valid insertion point, and
  the gather sum.

### Warmup is a duration, not a repetition count

One warmup repetition is enough for a 500 ms workload and useless for a 10 ms
one. A laptop CPU bursts to maximum turbo and settles toward its sustained
clock over hundreds of milliseconds, so a short workload with a single warmup
rep sampled a different point on that ramp every pass.

- Warmup now runs until `WARMUP_MIN_SECONDS` (250 ms) has elapsed. In practice
  `integer` now gets ~11 warmup reps where it previously got 1.
- `warmup_time` and `warmup_reps` are reported per subtest.

### Repetitions that are too short are flagged

Below `MIN_USEFUL_REP_SECONDS` (20 ms) a repetition cannot average out
scheduling quanta, interrupts or clock transitions, and no amount of repetition
fixes a structurally noisy measurement.

- `short_rep_warning` is set on the result and logged.
- Calibration marks the category in its output.
- A test asserts every shipped workload clears the floor. It immediately caught
  both resized workloads sitting at ~13 ms on fast hardware, which is how the
  loop counts above were chosen rather than guessed.

### Elsewhere

- Calibration `target_duration` raised 0.8 s → 1.5 s per subtest.
- 110 tests, up from 106.

## 2.0.2

Prompted by the first real calibration run on an i5-13450HX, which produced
spreads that tracked workload duration almost exactly:

    integer          2,019.90 Mops/sec   spread 36.92%   (shortest workload)
    branch_heavy        29.20 Mops/sec   spread 10.74%
    floating_point   3,393.07 MFLOPS     spread  8.38%
    matrix              45.86 GFLOPS     spread  1.18%   (longest workload)

That pattern is the signature of periodic interference being averaged out by
longer runs, and the cause was BenchMind's own core selection.

### Single-core pinning no longer targets core 0

Through 2.0.1, `pinned_to_core()` used `cpu_affinity()[0]` — logical core 0,
which is the worst available choice on Windows for two independent reasons:
it fields a disproportionate share of interrupts and DPCs, and on a hybrid part
it is an SMT sibling sharing a physical core with logical 1.

- New `monitoring/topology.py` detects physical cores, SMT siblings and
  performance-versus-efficiency classes. Windows uses
  `GetLogicalProcessorInformationEx`, where `EfficiencyClass` is the only
  reliable way to tell a P-core from an E-core — core numbering order is not
  guaranteed. Linux reads sysfs `thread_siblings_list` and `cpu_capacity`.
  A count-based fallback exists and is marked low confidence.
- `select_benchmark_core()` prefers a performance core, avoids core 0, and
  takes the highest-numbered candidate.
- The result records which core was used, its type, whether core 0 was avoided,
  the SMT sibling that could not be reserved, and the detection confidence. The
  report says what was done, not what was intended.

### Calibration refuses noisy baselines

`calibrate_baselines.py` now emits nothing when any category exceeds
`--max-spread` (default 5%), listing the offenders worst-first with concrete
remedies. A baseline captured from noisy data permanently miscalibrates its
category and looks authoritative while doing so. The gate logic is a pure
function, `evaluate_spreads()`, tested against the real 36.92% failure above.

The emitted block now annotates each baseline with its spread and pass count,
and the recorded reference metadata includes topology and pinned core.

### Elsewhere

- `.gitattributes` normalises line endings, silencing the CRLF warnings on
  every Windows commit and keeping a Linux checkout byte-identical.
- 106 tests, up from 93.

## 2.0.1

Three fixes, all prompted by the same real run on a Lenovo i5-13450HX laptop
(6 P-cores + 4 E-cores, 16 threads). The 2.0.0 report for that machine said:

    Throttling detected (low confidence). Sustained 100.0% of opening clock
    speed, package temperature rose 6.1 C to a peak of 88.0 C.

That sentence contradicts itself, and tracking down why exposed three separate
defects.

### The clock signal was never real on Windows

`psutil.cpu_freq()` reads the nominal base clock from the registry, not the
live frequency. On the test machine it returned a constant 2400 MHz while the
chip was actually ranging between 800 and 4600 MHz. Every throttle analysis
ever run on Windows was reading a flat line.

- Clocks now come from LibreHardwareMonitor's MSR readings, alongside the
  temperatures already being polled there.
- P-core and E-core clocks are collected separately. Averaging them would make
  a shift in the work split between core types look exactly like throttling.
- CPU package power is now recorded too. On a laptop, PL1 stepping down is
  usually the earliest throttle signal, arriving before the clock visibly
  collapses.
- `clock_source` is recorded on every snapshot. On Windows a psutil-derived
  clock is discarded rather than stored: a constant that looks like a
  measurement is worse than an honest gap.
- Sensor value parsing handles unit suffixes and thousands separators
  (`4,192.5 MHz`, `45.3 W`, `62.0 °C`).

### Throttle detection fired on temperature alone

- A throttle verdict now **requires** a measured drop in clock or package
  power. Temperature alone produces a note, never a verdict.
- `HOT_TEMP_C` raised from 85 to 95. Mobile H- and HX-class parts routinely
  sustain high 80s under all-core load by design; that is the cooling solution
  at its operating point, not a fault.
- A clock series with no variance is treated as **absent**, not as steady.
- The report distinguishes thermal from power throttling, and distinguishes
  "no throttling" from "could not be assessed".

### Standard deviation cannot see drift

Five runs on the test machine gave 2060, 2109, 2081, 1983, 1936. Standard
deviation is 3.54%, which 2.0.0 called "GOOD" — but the scores fall
monotonically after run 2, single-core drops 19% from best to worst, and peak
temperature creeps upward. That is thermal soak across the session, not
measurement noise, and 30 s of cooldown was not clearing it.

- `detect_drift()` separates a systematic trend from scatter using
  least-squares slope for magnitude and Spearman rank correlation against run
  order for direction. Both must clear their threshold.
- Upward drift is reported differently from downward: early slow runs usually
  mean a warmup cost, late slow runs usually mean heat.
- A drifting session is penalised in the repeatability score, however small
  its standard deviation, and `repeat` prints the trend **before** the spread.
- Default cooldown raised from 30 s to 90 s.

### Elsewhere

- The dashboard shows package power, labels a missing clock as "no clock
  source", and refuses to plot a flat synthetic clock line.
- `scripts/fetch_tools.py` tries the current release asset name first. The
  hardcoded `net472` name was stale, and those older builds depend on the
  WinRing0 driver that Windows now blocks by default under the vulnerable
  driver blocklist — the failure mode is silent, with the app running normally
  while every MSR sensor reads `-`.
- 93 tests, up from 81. The new ones reproduce the exact false positive above
  and assert the summary can never contradict itself.

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
