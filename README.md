# BenchMind

A hardware benchmarking and diagnostics suite that tells you what your machine
is good at, not just how big a number it can produce.

What makes it different from a score generator:

- **Every score carries a 95% confidence interval**, and BenchMind refuses to
  call two overlapping scores different.
- **A validity gate** checks background load, battery state, free memory and
  starting temperature before measuring, and marks the run `valid`, `tainted`
  or `invalid`.
- **Thermal and power throttle detection** from real MSR clock, package power
  and temperature: "sustained 87% of opening clock, onset at 94 s, peak 96 °C".
  A verdict requires a measured drop, so a hot-but-healthy laptop is not
  accused of throttling.
- **Drift versus scatter**: five scores falling steadily is a different finding
  from five scores bouncing around a mean, and BenchMind reports them
  separately.
- **Roofline analysis** classifies the machine as compute-limited or
  memory-bandwidth-limited by comparing cache-resident against DRAM-resident
  throughput.
- **Thread scaling curves** that expose SMT gain, hybrid P/E core behaviour,
  and the point where memory bandwidth saturates.
- **Regression detection** against your own machine's history: "down 12% since
  last month" is a far more useful sentence than any leaderboard position.
- **Environment fingerprinting**, because a benchmark written in Python
  measures a software stack as much as silicon, and pretending otherwise makes
  every comparison a lie.

---

## Install

```bash
git clone <your-repo> BenchMind
cd BenchMind
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Optional extras:

```bash
pip install pyopencl            # GPU benchmarking
python -m scripts.fetch_tools   # Windows temperature source
```

Temperature sources: Windows needs LibreHardwareMonitor with its web server on
port 8085 (run it as Administrator, Options → Remote Web Server → Run). Linux
works through psutil automatically. macOS has no source wired up yet; BenchMind
reports "no temperature source" and skips thermal analysis rather than
inventing numbers.

---

## Use

```bash
python run.py check                  # is now a good time to benchmark?
python run.py bench                  # standard run, CPU + GPU
python run.py bench --mode full --scaling
python run.py repeat --runs 5        # repeatability check
python run.py history
python run.py env                    # environment fingerprint
python run.py serve                  # dashboard at localhost:8000/dashboard
```

Start with `check`. On a laptop, running on battery can halve your score, and
`check` will tell you before you waste four minutes.

`repeat` is the most important command in the project. If your score moves more
than about 2% between back-to-back runs, no comparison you make with it means
anything.

---

## What the numbers mean

**1000 points equals the BenchMind reference machine** on that workload. The
reference and its measured raw metrics are documented in
`docs/BENCHMARK_SPEC.md`.

```
CPU Index = 0.4 × single-core + 0.6 × multi-core
```

Both composites are geometric means, so one workload with a huge normalized
score cannot dominate.

Scores are comparable only when the environment fingerprint, the mode and the
baseline version all match. BenchMind enforces this rather than trusting you to
remember it.

The `interpreter` category is reported but **excluded from the index**: it
measures CPython's speed, and moves when you upgrade Python without touching
the hardware. Keeping it visible and excluded is more honest than either
hiding it or pretending it is a hardware metric.

---

## API

```
GET    /dashboard                 web UI
POST   /api/benchmark             start a run, returns a job id immediately
GET    /api/jobs/{id}             poll status and result
GET    /api/jobs/{id}/events      server-sent progress stream
GET    /api/dashboard             latest completed result
GET    /api/system-info           machine and environment
GET    /api/validity              conditions check
GET    /api/history               local run history
WS     /ws/live-monitor           live telemetry
```

Interactive docs at `/docs`.

---

## Development

```bash
python -m unittest discover -s tests
```

110 tests, none of them timing-dependent — CI runners are noisy shared VMs and
asserting on speed there produces flaky failures that teach people to ignore
the suite. Speed is verified by `scripts/repeatability.py` on real hardware.

Before treating any benchmark change as complete:

1. unit tests pass
2. `python run.py bench --mode standard` succeeds
3. `python run.py repeat --runs 3` shows an acceptable spread
4. telemetry and temperatures inspected

Read `PROJECT_CONTEXT.md` and `docs/BENCHMARK_SPEC.md` before changing
anything that affects measurement.

---

## Honest limitations

- BenchMind measures your machine **as a Python compute host**. The matrix
  score is largely your BLAS build; hashing is largely your OpenSSL build.
  Fingerprinting makes scores comparable within a stack, not across stacks.
- The reference machine R1 is a cloud instance and therefore a poor reference.
  Recalibrate against physical hardware with
  `python -m scripts.calibrate_baselines`.
- GPU baselines are untested placeholders. Run the suite on a real GPU and
  recalibrate before trusting GPU scores.
- No RAM latency, storage or network benchmark yet.
- Core pinning targets logical core 0 without topology detection, so on a
  hybrid CPU it may land on a P-core or an E-core.

---

## Licence and third-party notices

LibreHardwareMonitor is **not** bundled. `scripts/fetch_tools.py` downloads it
on demand; it is licensed under MPL-2.0 and its notice is written alongside the
download.
