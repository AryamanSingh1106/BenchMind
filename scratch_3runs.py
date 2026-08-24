import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

from benchmarks.cpu.cpu_suite import run_full_cpu_suite

for run_idx in range(1, 4):
    print(f"\n{'='*60}")
    print(f"RUN {run_idx} / 3")
    print('='*60)
    t0 = time.perf_counter()
    r = run_full_cpu_suite(mode="quick")
    elapsed = time.perf_counter() - t0

    print(f"  Total suite time:    {r['total_suite_time']:.2f}s")
    print(f"  CPU Index:           {r['cpu_index']:.2f}")
    print(f"  Single-Core Score:   {r['single_core_score']:.2f}")
    print(f"  Multi-Core Score:    {r['multi_core_score']:.2f}")
    print(f"  Scoring weights:     SC={r['scoring_weights']['single_core_weight']} MC={r['scoring_weights']['multi_core_weight']}")
    print()
    print("  --- Single-Core Subtests ---")
    sc_subtests = [s for s in r['subtests'] if 'Multi' not in s['name']]
    for s in sc_subtests:
        flag = "✓" if s['validation_passed'] else "✗"
        print(f"  {flag} {s['name']:<40} score={s['score']:>8.2f}  {s['raw_metric_name']}={s['raw_metric_value']:>10.4f}  stability={s['stability_pct']:.1f}%  reps={s['repetitions']}")
    print()
    print("  --- Multi-Core Subtests ---")
    mc_subtests = [s for s in r['subtests'] if 'Multi' in s['name']]
    for s in mc_subtests:
        flag = "✓" if s['validation_passed'] else "✗"
        print(f"  {flag} {s['name']:<40} score={s['score']:>8.2f}  {s['raw_metric_name']}={s['raw_metric_value']:>10.4f}")
    print()
    ts = r['telemetry_summary']
    print(f"  --- Telemetry Summary ---")
    print(f"  Max CPU Temp:        {ts['max_cpu_temp']} °C")
    print(f"  Avg CPU Util:        {ts['avg_cpu_utilization']} %")
    print(f"  Max CPU Util:        {ts['max_cpu_utilization']} %")
    print(f"  Telemetry Samples:   {ts['sample_count']}")

print("\n\nAll 3 runs complete.")
